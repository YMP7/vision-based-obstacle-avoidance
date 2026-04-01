from __future__ import annotations

"""
Pioneer 3 Vision AI Controller
================================
Autonomous obstacle-avoidance controller for the Webots Pioneer 3-DX robot.

Uses:
  • YOLOv8 for real-time object detection
  • MiDaS  for monocular depth estimation
  • Braitenberg-style Gaussian-weighted reactive steering (from lidar reference)

Architecture:
  DepthEstimator       – wraps MiDaS inference with error handling
  ObjectDetector       – wraps YOLO with confidence/class filtering
  NavigationController – Braitenberg smooth steering + YOLO safety override
"""

from controller import Robot
from collections import namedtuple
import logging
import math
import os
import sys

import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg

# ─── Logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("VisionAI")

# ─── Data structures ───────────────────────────────────────────────
Detection = namedtuple("Detection", ["x1", "y1", "x2", "y2", "confidence", "class_id"])


# ====================================================================
#  DepthEstimator
# ====================================================================
class DepthEstimator:
    """Monocular depth estimation using MiDaS.

    MiDaS outputs *inverse* relative depth — higher values indicate
    objects that are **closer** to the camera.
    """

    def __init__(self):
        logger.info("Loading MiDaS model (%s) …", cfg.MIDAS_MODEL_TYPE)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model = self._load_hub_model(cfg.MIDAS_MODEL_TYPE)
        self._model.eval()
        self._model.to(self._device)

        transforms = self._load_hub_model("transforms")
        self._transform = getattr(transforms, cfg.MIDAS_TRANSFORM_TYPE)
        logger.info("MiDaS ready on %s", self._device)

    @staticmethod
    def _load_hub_model(entry: str):
        """Load a MiDaS entry from torch.hub, falling back to local cache."""
        import os
        try:
            return torch.hub.load("intel-isl/MiDaS", entry, trust_repo=True)
        except Exception as exc:
            logger.warning("Network load failed (%s), trying local cache …", exc)
            cache_dir = os.path.join(
                torch.hub.get_dir(), "intel-isl_MiDaS_master"
            )
            if os.path.isdir(cache_dir):
                return torch.hub.load(cache_dir, entry, source="local")
            raise RuntimeError(
                f"MiDaS cache not found at {cache_dir} and network unavailable"
            ) from exc

    def estimate(self, frame: np.ndarray) -> np.ndarray | None:
        """Return a normalised depth map (0–1, higher = closer), or None."""
        try:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_batch = self._transform(img_rgb).to(self._device)

            with torch.no_grad():
                prediction = self._model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            depth = prediction.cpu().numpy()

            # Percentile-clipped normalisation
            d_low = np.percentile(depth, cfg.DEPTH_NORM_LOW_PERCENTILE)
            d_high = np.percentile(depth, cfg.DEPTH_NORM_HIGH_PERCENTILE)
            if d_high - d_low > 1e-6:
                depth = np.clip((depth - d_low) / (d_high - d_low), 0.0, 1.0)
            else:
                depth = np.zeros_like(depth)

            return depth.astype(np.float32)

        except Exception as exc:
            logger.error("Depth estimation failed: %s", exc)
            return None


# ====================================================================
#  ObjectDetector
# ====================================================================
class ObjectDetector:
    """YOLOv8-based object detector with confidence and class filtering."""

    def __init__(self):
        logger.info("Loading YOLO model (%s) …", cfg.YOLO_MODEL_PATH)
        self._model = YOLO(cfg.YOLO_MODEL_PATH)
        logger.info("YOLO ready")

    def detect(self, frame: np.ndarray, img_h: int, img_w: int) -> list[Detection]:
        """Run detection on *frame* and return filtered, bounds-checked results."""
        try:
            results = self._model(frame, verbose=False)
        except Exception as exc:
            logger.error("YOLO inference failed: %s", exc)
            return []

        detections: list[Detection] = []

        for r in results:
            for box, conf_tensor, cls_tensor in zip(
                r.boxes.xyxy, r.boxes.conf, r.boxes.cls
            ):
                conf = float(conf_tensor)
                cls_id = int(cls_tensor)

                if conf < cfg.YOLO_CONFIDENCE_THRESHOLD:
                    continue
                if cfg.OBSTACLE_CLASS_IDS and cls_id not in cfg.OBSTACLE_CLASS_IDS:
                    continue

                coords = box.cpu().numpy()
                x1 = int(np.clip(coords[0], 0, img_w - 1))
                y1 = int(np.clip(coords[1], 0, img_h - 1))
                x2 = int(np.clip(coords[2], 0, img_w - 1))
                y2 = int(np.clip(coords[3], 0, img_h - 1))

                if x2 <= x1 or y2 <= y1:
                    continue

                # Reject detections sitting on the ground (bottom 15% of frame)
                box_center_y = (y1 + y2) / 2
                if box_center_y > img_h * 0.85:
                    continue

                detections.append(Detection(x1, y1, x2, y2, conf, cls_id))

        return detections


# ====================================================================
#  NavigationController
# ====================================================================
class NavigationController:
    """Simple zone-based reactive navigation inspired by the C reference.

    Divides the depth map into LEFT, CENTER, RIGHT zones and makes
    decisive navigation decisions:
      - Path clear     → full forward
      - Obstacle ahead → hard turn away from the closer side
      - Obstacle side  → steer away

    Uses MiDaS depth for direction-aware obstacle detection.
    YOLO provides an additional safety override layer.
    """

    # ── Depth zone thresholds ───────────────────────────────────────
    OBSTACLE_THRESHOLD = 0.08    # depth above floor baseline = obstacle (after median subtraction)
    DANGER_THRESHOLD = 0.15      # depth well above floor = too close, hard turn

    def __init__(self, scan_width: int):
        self.state: str = "FORWARD"
        self._scan_width = scan_width
        logger.info("NavigationController ready (zone-based, scan_width=%d)", scan_width)

    # ── Public API ──────────────────────────────────────────────────

    def decide(
        self,
        detections: list[Detection],
        depth_map: np.ndarray | None,
        frame_width: int,
        step_count: int = 0,
    ) -> tuple[float, float]:
        """Choose motor velocities.

        Priority:
        1. YOLO safety override (hard turn if YOLO detects close object)
        2. Depth zone-based navigation (like C reference but directional)
        3. Full forward (fallback if no depth available)
        """

        # ── Layer 1: YOLO safety override ───────────────────────────
        if detections and depth_map is not None:
            yolo_result = self._check_yolo_safety(
                detections, depth_map, frame_width
            )
            if yolo_result is not None:
                return yolo_result

        # ── Layer 2: Depth zone navigation ──────────────────────────
        if depth_map is not None:
            return self._zone_navigate(depth_map, step_count)

        # ── Fallback: full forward ──────────────────────────────────
        self.state = "FORWARD"
        return self._forward_speed()

    # ── Layer 1: YOLO safety ────────────────────────────────────────

    def _check_yolo_safety(
        self,
        detections: list[Detection],
        depth_map: np.ndarray,
        frame_width: int,
    ) -> tuple[float, float] | None:
        """Hard turn if YOLO detects a very close obstacle."""
        closest_det = None
        closest_depth_val = -1.0

        for det in detections:
            region = depth_map[det.y1 : det.y2, det.x1 : det.x2]
            if region.size == 0:
                continue
            depth_val = float(np.percentile(region, cfg.DEPTH_PERCENTILE))
            if depth_val > closest_depth_val:
                closest_depth_val = depth_val
                closest_det = det

        if closest_det is None:
            return None

        if closest_depth_val >= cfg.DEPTH_CLOSE_THRESHOLD:
            obstacle_cx = (closest_det.x1 + closest_det.x2) / 2
            frame_cx = frame_width / 2
            t = cfg.TURN_SPEED_FACTOR * cfg.MAX_SPEED

            if obstacle_cx > frame_cx:
                self.state = "YOLO_LEFT"
                left_vel, right_vel = -t, t
            else:
                self.state = "YOLO_RIGHT"
                left_vel, right_vel = t, -t

            logger.info(
                "YOLO: cls%d conf=%.0f%% depth=%.2f → %s (L=%.2f R=%.2f)",
                closest_det.class_id, closest_det.confidence * 100,
                closest_depth_val, self.state, left_vel, right_vel,
            )
            return left_vel, right_vel

        return None

    # ── Layer 2: Zone-based depth navigation ────────────────────────

    def _zone_navigate(
        self, depth_map: np.ndarray, step_count: int
    ) -> tuple[float, float]:
        """Simple zone-based navigation like the C reference.

        1. Extract the depth map's horizon strip
        2. Divide into LEFT, CENTER, RIGHT zones
        3. Compute average depth (proximity) for each zone
        4. Make decisive navigation decision
        """
        h, w = depth_map.shape[:2]
        y_top = int(h * cfg.BRAITENBERG_ZONE_TOP)
        y_bot = int(h * cfg.BRAITENBERG_ZONE_BOTTOM)

        strip = depth_map[y_top:y_bot, :]
        if strip.size == 0:
            self.state = "FORWARD"
            return self._forward_speed()

        # Extract 1D scan: max depth per column picks up wall peaks
        # (floor contributes a uniform baseline, walls rise above it)
        scan = np.max(strip, axis=0)  # shape: (w,)

        # Subtract floor baseline (median) — only real obstacles remain
        baseline = float(np.median(scan))
        scan_clean = np.clip(scan - baseline, 0.0, None)

        # Zone averages from obstacle-only signal
        third = w // 3
        left_prox = float(np.mean(scan_clean[:third]))
        center_prox = float(np.mean(scan_clean[third:2*third]))
        right_prox = float(np.mean(scan_clean[2*third:]))

        # ── Decision logic ──────────────────────────────────────────
        fwd = cfg.MAX_SPEED * 0.95
        turn = cfg.MAX_SPEED * 0.5

        if center_prox > self.DANGER_THRESHOLD:
            # Wall dead ahead — hard turn away from closer side
            if left_prox > right_prox:
                self.state = "HARD_RIGHT"
                left_vel, right_vel = turn, -turn
            else:
                self.state = "HARD_LEFT"
                left_vel, right_vel = -turn, turn

        elif center_prox > self.OBSTACLE_THRESHOLD:
            # Obstacle approaching ahead — moderate turn
            if left_prox > right_prox:
                self.state = "TURN_RIGHT"
                left_vel, right_vel = fwd * 0.6, -fwd * 0.2
            else:
                self.state = "TURN_LEFT"
                left_vel, right_vel = -fwd * 0.2, fwd * 0.6

        elif left_prox > self.OBSTACLE_THRESHOLD:
            # Obstacle on left — steer right
            self.state = "STEER_RIGHT"
            left_vel, right_vel = fwd, fwd * 0.3

        elif right_prox > self.OBSTACLE_THRESHOLD:
            # Obstacle on right — steer left
            self.state = "STEER_LEFT"
            left_vel, right_vel = fwd * 0.3, fwd

        else:
            # Path clear — full forward
            self.state = "FORWARD"
            left_vel, right_vel = fwd, fwd

        # Clamp
        left_vel = max(-cfg.MAX_SPEED, min(cfg.MAX_SPEED, left_vel))
        right_vel = max(-cfg.MAX_SPEED, min(cfg.MAX_SPEED, right_vel))

        # Logging
        if step_count % cfg.LOG_INTERVAL == 0:
            logger.info(
                "NAV  L=%.2f C=%.2f R=%.2f → %s (L=%.2f R=%.2f)",
                left_prox, center_prox, right_prox,
                self.state, left_vel, right_vel,
            )

        return left_vel, right_vel

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _forward_speed() -> tuple[float, float]:
        speed = cfg.MAX_SPEED * 0.8
        return speed, speed


# ====================================================================
#  Main loop
# ====================================================================
def main():  # noqa: C901
    """Entry-point: initialise hardware, run the sense → think → act loop."""

    # ── Robot & devices ─────────────────────────────────────────────
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    camera = robot.getDevice(cfg.CAMERA_DEVICE_NAME)
    camera.enable(timestep)
    img_w = camera.getWidth()
    img_h = camera.getHeight()

    left_motor = robot.getDevice(cfg.LEFT_MOTOR_DEVICE_NAME)
    right_motor = robot.getDevice(cfg.RIGHT_MOTOR_DEVICE_NAME)
    left_motor.setPosition(float("inf"))
    right_motor.setPosition(float("inf"))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    # ── AI modules ──────────────────────────────────────────────────
    detector = ObjectDetector()
    depth_estimator = DepthEstimator()
    navigator = NavigationController(scan_width=img_w)

    # ── Frame-skip bookkeeping ──────────────────────────────────────
    step_count = 0
    cached_detections: list[Detection] = []
    cached_depth: np.ndarray | None = None

    logger.info(
        "Controller started  (inference every %d steps, vis=%s)",
        cfg.INFERENCE_INTERVAL,
        cfg.DEBUG_VISUALIZATION,
    )

    # ── Sense → Think → Act ────────────────────────────────────────
    try:
        while robot.step(timestep) != -1:
            # Grab raw camera image
            raw = camera.getImage()
            frame = np.frombuffer(raw, np.uint8).reshape((img_h, img_w, 4))
            frame = frame[:, :, :3]  # drop alpha channel

            # Run heavy inference only every N steps
            if step_count % cfg.INFERENCE_INTERVAL == 0:
                cached_detections = detector.detect(frame, img_h, img_w)
                cached_depth = depth_estimator.estimate(frame)

            # Navigation decision
            left_vel, right_vel = navigator.decide(
                cached_detections, cached_depth, img_w, step_count
            )

            left_motor.setVelocity(left_vel)
            right_motor.setVelocity(right_vel)

            # ── Visualisation (optional) ────────────────────────────
            if cfg.DEBUG_VISUALIZATION:
                vis_frame = frame.copy()
                for det in cached_detections:
                    is_avoiding = navigator.state.startswith("YOLO")
                    color = (0, 0, 255) if is_avoiding else (0, 255, 0)
                    cv2.rectangle(vis_frame, (det.x1, det.y1), (det.x2, det.y2), color, 2)
                    label = f"{det.confidence:.0%} cls{det.class_id}"
                    cv2.putText(
                        vis_frame, label,
                        (det.x1, max(det.y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                    )

                # HUD: state + velocities
                cv2.putText(
                    vis_frame,
                    f"{navigator.state}  L={left_vel:.2f} R={right_vel:.2f}",
                    (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                )

                cv2.imshow("Camera", vis_frame)

                if cached_depth is not None:
                    depth_u8 = (cached_depth * 255).astype(np.uint8)
                    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
                    cv2.imshow("Depth", depth_color)

                cv2.waitKey(1)

            step_count += 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)
        if cfg.DEBUG_VISUALIZATION:
            cv2.destroyAllWindows()
        logger.info("Controller stopped")


# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()