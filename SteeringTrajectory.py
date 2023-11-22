from Variables import *
from Functions import *

# Calculate wheel angles
ackermann_angle = compute_ackermann_angle(wheelbase, steering_angle, steering_ratio)

# Calculate trajectories
trajectory_center, trajectory_inner_rear_wheel, trajectory_outer_rear_wheel = compute_ackermann_trajectory(wheelbase, ackermann_angle, speed, time_step, total_time)

# Plotting
x_values_center, y_values_center = zip(*trajectory_center)
x_values_inner_rear, y_values_inner_rear = zip(*trajectory_inner_rear_wheel)
x_values_outer_rear, y_values_outer_rear = zip(*trajectory_outer_rear_wheel)

# plot_wheel_trajectories(x_values_center, y_values_center, x_values_inner_rear, y_values_inner_rear, x_values_outer_rear, y_values_outer_rear)

trajectory_points_3d, trajectory_inner_rear_wheel_3d, trajectory_outer_rear_wheel_3d = convert_points_to_3d(trajectory_center, trajectory_inner_rear_wheel, trajectory_outer_rear_wheel)

image_points, image_points_inner_rear_wheel, image_points_outer_rear_wheel = project_3d_to_2d(trajectory_points_3d, trajectory_inner_rear_wheel_3d, trajectory_outer_rear_wheel_3d, rotation_vector, translation_vector, camera_matrix, dist)

visualize_trajectories(video_path, camera_matrix, dist, image_points, image_points_inner_rear_wheel, image_points_outer_rear_wheel)
