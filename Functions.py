import numpy as np
import matplotlib.pyplot as plt
from Variables import *
import cv2

def compute_ackermann_angle(wheelbase, steering_angle, steering_ratio):
    # Compute the wheel angles
    ackermann_angle = np.radians(steering_angle) / steering_ratio
    # turning_radius = wheelbase / np.tan(ackermann_angle)

    return ackermann_angle

def compute_ackermann_trajectory(wheelbase, ackermann_angle, speed, time_step, total_time):
    trajectory_center = []  # Trajectory of the vehicle (center)
    trajectory_inner_rear_wheel = []  # Trajectory of the inner rear wheel
    trajectory_outer_rear_wheel = []  # Trajectory of the outer rear wheel
    
    # Initial conditions
    x_center, y_center, heading_angle = 0.0, 0.0, np.pi / 2
    x_inner_rear_wheel, y_inner_rear_wheel = 0.0, 0.0
    x_outer_rear_wheel, y_outer_rear_wheel = 0.0, 0.0

    for _ in np.arange(0, total_time, time_step):
        # Angular velocity
        angular_velocity = speed / wheelbase * np.tan(ackermann_angle)

        # Update heading angle
        heading_angle += angular_velocity * time_step

        # Update position of the vehicle (center)
        x_center += speed * np.cos(heading_angle) * time_step
        y_center += speed * np.sin(heading_angle) * time_step

        # Update position of the inner rear wheel
        x_inner_rear_wheel = x_center - 0.5 * track_width * np.sin(heading_angle)
        y_inner_rear_wheel = y_center + 0.5 * track_width * np.cos(heading_angle)

        # Update position of the outer rear wheel
        x_outer_rear_wheel = x_center + 0.5 * track_width * np.sin(heading_angle)
        y_outer_rear_wheel = y_center - 0.5 * track_width * np.cos(heading_angle)
        
        trajectory_center.append((x_center, y_center))
        trajectory_inner_rear_wheel.append((x_inner_rear_wheel, y_inner_rear_wheel))
        trajectory_outer_rear_wheel.append((x_outer_rear_wheel, y_outer_rear_wheel))

    return trajectory_center, trajectory_inner_rear_wheel, trajectory_outer_rear_wheel


def plot_wheel_trajectories(x_values_center, y_values_center, x_values_inner_rear, y_values_inner_rear, x_values_outer_rear, y_values_outer_rear):
 
    # Create the plot
    plt.figure(figsize=(8, 8))

    # Plot trajectories
    plt.plot(x_values_center, y_values_center, label='Vehicle Trajectory (Center)')
    
    plt.plot(x_values_inner_rear, y_values_inner_rear, label='Right Wheel Trajectory')
    plt.plot(x_values_outer_rear, y_values_outer_rear, label='Left Wheel Trajectory')


    # Add labels and legend
    plt.title('Vehicle Trajectory based on Ackermann Steering Model')
    plt.xlabel('X Coordinate (meters)')
    plt.ylabel('Y Coordinate (meters)')
    plt.legend()

    # Display the plot
    plt.grid(True)
    plt.show()


def convert_points_to_3d(trajectory_center, trajectory_inner_rear_wheel, trajectory_outer_rear_wheel):
    
    '''
    Args:
        trajectory_points: Trajectory points in 2D
    Returns:
        trajectory_points_3d: Trajectory points in 3D
    '''
    # Assuming trajectory_points is a list of 2D points (x, y)
    trajectory_center = np.array([(x, -y) for x, y in trajectory_center], dtype=np.float32)
    trajectory_inner_rear_wheel = np.array([(x, -y) for x, y in trajectory_inner_rear_wheel], dtype=np.float32)
    trajectory_outer_rear_wheel = np.array([(x, -y) for x, y in trajectory_outer_rear_wheel], dtype=np.float32)

    # Convert 2D points to 3D points by adding a constant depth
    depth_value = 1.0
    trajectory_points_3d = np.column_stack((trajectory_center, np.full_like(trajectory_center[:, 0], depth_value)))
    trajectory_inner_rear_wheel_3d = np.column_stack((trajectory_inner_rear_wheel, np.full_like(trajectory_inner_rear_wheel[:, 0], depth_value)))
    trajectory_outer_rear_wheel_3d = np.column_stack((trajectory_outer_rear_wheel, np.full_like(trajectory_outer_rear_wheel[:, 0], depth_value)))

    return trajectory_points_3d, trajectory_inner_rear_wheel_3d, trajectory_outer_rear_wheel_3d

def project_3d_to_2d(trajectory_points_3d, trajectory_inner_rear_wheel_3d, trajectory_outer_rear_wheel_3d, rotation_vector, translation_vector, camera_matrix, dist):
    
    '''
    Args:
        trajectory_points_3d: Trajectory points in 3D
        rotation_vector: Rotation vector
        translation_vector: Translation vector
        camera_matrix: Camera matrix
        dist: Distortion coefficients
    Returns:
        trajectory_points: Trajectory points in 2D
    '''
    # Project 3D points to 2D image plane
    image_points, _ = cv2.projectPoints(trajectory_points_3d, rotation_vector, translation_vector, camera_matrix, dist)
    image_points_inner_rear_wheel, _ = cv2.projectPoints(trajectory_inner_rear_wheel_3d, rotation_vector, translation_vector, camera_matrix, dist)
    image_points_outer_rear_wheel, _ = cv2.projectPoints(trajectory_outer_rear_wheel_3d, rotation_vector, translation_vector, camera_matrix, dist)

    return (np.array(image_points), np.array(image_points_inner_rear_wheel), np.array(image_points_outer_rear_wheel))

def visualize_trajectories(video_path, camera_matrix, dist, image_points, image_points_inner_rear_wheel, image_points_outer_rear_wheel):

    '''
    Args:
        video_path: Path to the video file
        camera_matrix: Camera matrix
        dist: Distortion coefficients
        image_points: Trajectory points in 2D
    '''

    # Create a video capture object
    cap = cv2.VideoCapture(video_path)

    while True:
        # Read a single frame from the video
        ret, image = cap.read()

        if not ret:
            break

        # Resize the frame to 960x540
        image = cv2.resize(image, (960, 540))

        # Undistort the image
        h, w = image.shape[:2]
        image = cv2.undistort(image, camera_matrix, dist, None)

        # Draw trajectories on the image
        for pt in image_points.reshape(-1, 2).astype(int):
            pt_shifted = tuple(pt)
            cv2.circle(image, pt_shifted, 2, (255, 0, 0), -1)

        for pt in image_points_inner_rear_wheel.reshape(-1, 2).astype(int):
            pt_shifted = tuple(pt)
            cv2.circle(image, pt_shifted, 2, (0, 255, 0), -1)

        for pt in image_points_outer_rear_wheel.reshape(-1, 2).astype(int):
            pt_shifted = tuple(pt)
            cv2.circle(image, pt_shifted, 2, (0, 0, 255), -1)

        # Display the image with trajectories
        cv2.imshow('Points Visualization', image)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()
