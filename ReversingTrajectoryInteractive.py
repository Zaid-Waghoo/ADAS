import cv2
import numpy as np
from Functions import *
from Variables import *

# Function to update the steering angle based on the trackbar value
def update_steering_angle(value):
    global steering_angle
    steering_angle = value - 360  # Adjust to go from -360 to 360

# Open a video capture object
cap = cv2.VideoCapture('ReversingCamera.mp4')

# Create a window
cv2.namedWindow('Trajectory Visualization')

# Create a trackbar for steering angle control
cv2.createTrackbar('Steering', 'Trajectory Visualization', 360, 720, update_steering_angle)  # Adjust range to 720

steering_angle = 0  # Initial steering angle value

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Resize the frame to 960x540
    frame = cv2.resize(frame, (960, 540))

    # Undistort the image
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist, (w, h), 1, (w, h))
    frame = cv2.undistort(frame, camera_matrix, dist, None)

    # Calculate wheel angles
    ackermann_angle = compute_ackermann_angle(wheelbase, steering_angle, steering_ratio)

    # Calculate trajectories
    trajectory_center, trajectory_inner_rear_wheel, trajectory_outer_rear_wheel = compute_ackermann_trajectory(wheelbase, ackermann_angle, speed, time_step, total_time)

    trajectory_points_3d, trajectory_inner_rear_wheel_3d, trajectory_outer_rear_wheel_3d = convert_points_to_3d(trajectory_center, trajectory_inner_rear_wheel, trajectory_outer_rear_wheel)

    image_points, image_points_inner_rear_wheel, image_points_outer_rear_wheel = project_3d_to_2d(trajectory_points_3d, trajectory_inner_rear_wheel_3d, trajectory_outer_rear_wheel_3d, rotation_vector, translation_vector, camera_matrix, dist)

    # Draw trajectories on the image
    for pt in image_points.reshape(-1, 2).astype(int):
        pt_shifted = tuple(pt)
        cv2.circle(frame, pt_shifted, 2, (255, 0, 0), -1)

    for pt in image_points_inner_rear_wheel.reshape(-1, 2).astype(int):
        pt_shifted = tuple(pt)
        cv2.circle(frame, pt_shifted, 2, (0, 255, 0), -1)
    
    for pt in image_points_outer_rear_wheel.reshape(-1, 2).astype(int):
        pt_shifted = tuple(pt)
        cv2.circle(frame, pt_shifted, 2, (0, 0, 255), -1)
    
    # Draw the steering angle text on the image
    cv2.putText(frame, "Steering Angle: " + str(steering_angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    # Display the frame with trajectories
    cv2.imshow('Trajectory Visualization', frame)

    # Check for the 'q' key to exit the loop
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
