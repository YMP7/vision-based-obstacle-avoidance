#include <webots/robot.h>
#include <webots/motor.h>
#include <webots/camera.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_SPEED 5.24

typedef struct {
  WbDeviceTag left_motor;
  WbDeviceTag right_motor;
  WbDeviceTag camera;
  int timestep;
} RobotDevices;


/* Initialize robot hardware */
void initialize_robot(RobotDevices *robot) {

  robot->timestep = wb_robot_get_basic_time_step();

  robot->left_motor = wb_robot_get_device("left wheel");
  robot->right_motor = wb_robot_get_device("right wheel");

  wb_motor_set_position(robot->left_motor, INFINITY);
  wb_motor_set_position(robot->right_motor, INFINITY);

  wb_motor_set_velocity(robot->left_motor, 0.0);
  wb_motor_set_velocity(robot->right_motor, 0.0);

  robot->camera = wb_robot_get_device("camera");
  wb_camera_enable(robot->camera, robot->timestep);

  printf("Robot initialized successfully.\n");
}


/* Set robot velocity */
void set_velocity(RobotDevices *robot, double left, double right) {

  if(left > MAX_SPEED) left = MAX_SPEED;
  if(right > MAX_SPEED) right = MAX_SPEED;

  if(left < -MAX_SPEED) left = -MAX_SPEED;
  if(right < -MAX_SPEED) right = -MAX_SPEED;

  wb_motor_set_velocity(robot->left_motor, left);
  wb_motor_set_velocity(robot->right_motor, right);
}


/* Simple prototype vision analysis */
int detect_obstacle(const unsigned char *image, int width, int height) {

  int brightness_sum = 0;
  int pixels = width * height;

  for(int i = 0; i < pixels; i++) {

    int r = image[4*i];
    int g = image[4*i + 1];
    int b = image[4*i + 2];

    int gray = (r + g + b) / 3;

    brightness_sum += gray;
  }

  int avg_brightness = brightness_sum / pixels;

  /* Prototype rule */
  if(avg_brightness < 80)
    return 1;  // obstacle detected

  return 0;
}


/* Main controller loop */
int main() {

  wb_robot_init();

  RobotDevices robot;

  initialize_robot(&robot);

  int width = wb_camera_get_width(robot.camera);
  int height = wb_camera_get_height(robot.camera);

  printf("Camera Resolution: %d x %d\n", width, height);

  while (wb_robot_step(robot.timestep) != -1) {

    const unsigned char *image = wb_camera_get_image(robot.camera);

    if(image == NULL)
      continue;

    int obstacle = detect_obstacle(image, width, height);

    if(obstacle) {

      /* Turn when obstacle detected */
      set_velocity(&robot, -0.5 * MAX_SPEED, 0.5 * MAX_SPEED);

      printf("Obstacle detected - turning\n");

    } else {

      /* Move forward */
      set_velocity(&robot, MAX_SPEED * 0.8, MAX_SPEED * 0.8);

      printf("Path clear - moving forward\n");
    }

  }

  wb_robot_cleanup();

  return 0;
}