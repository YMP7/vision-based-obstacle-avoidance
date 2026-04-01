"""
Configuration module for the Pioneer 3 Vision AI Controller.

Centralizes all tunable parameters, thresholds, and constants
so they can be adjusted without modifying controller logic.

Reference values from C lidar-based Braitenberg controller + C camera controller.
"""

import math

# ─────────────────────────────────────────────
# Robot Hardware
# ─────────────────────────────────────────────
MAX_SPEED = 5.24                     # rad/s — from C reference controller
CAMERA_DEVICE_NAME = "camera"
LEFT_MOTOR_DEVICE_NAME = "left wheel"
RIGHT_MOTOR_DEVICE_NAME = "right wheel"

# ─────────────────────────────────────────────
# YOLO Object Detection
# ─────────────────────────────────────────────
YOLO_MODEL_PATH = "yolov8n.pt"
YOLO_CONFIDENCE_THRESHOLD = 0.45
OBSTACLE_CLASS_IDS = {0, 1, 2, 3, 5, 7, 13, 56, 57, 62}  # removed 60 (dining table) — floor false positive

# ─────────────────────────────────────────────
# MiDaS Depth Estimation
# ─────────────────────────────────────────────
MIDAS_MODEL_TYPE = "MiDaS_small"
MIDAS_TRANSFORM_TYPE = "small_transform"

# Percentile-based normalisation to clip floor dominance.
DEPTH_NORM_LOW_PERCENTILE = 2
DEPTH_NORM_HIGH_PERCENTILE = 98

DEPTH_PERCENTILE = 75                # for YOLO bounding-box depth check
DEPTH_CLOSE_THRESHOLD = 0.30         # YOLO object too close → hard turn (lowered to react earlier)
DEPTH_SLOW_THRESHOLD = 0.10          # YOLO object moderate → slow down (lowered to react earlier)

# ─────────────────────────────────────────────
# Depth Zone Navigation (horizon strip)
# ─────────────────────────────────────────────
# The depth map horizon strip defines which vertical band of the camera
# is used for obstacle detection. Obstacle/danger thresholds are defined
# in NavigationController class constants.
BRAITENBERG_ZONE_TOP = 0.30          # horizon strip top — above most sky
BRAITENBERG_ZONE_BOTTOM = 0.70       # horizon strip bottom — below this is pure floor

# ─────────────────────────────────────────────
# Navigation
# ─────────────────────────────────────────────
TURN_SPEED_FACTOR = 0.50             # for YOLO emergency hard turns

# ─────────────────────────────────────────────
# Performance
# ─────────────────────────────────────────────
INFERENCE_INTERVAL = 3               # YOLO + MiDaS every N steps

# ─────────────────────────────────────────────
# Debug / Visualisation
# ─────────────────────────────────────────────
DEBUG_VISUALIZATION = True
LOG_INTERVAL = 10                    # log Braitenberg values every N steps
