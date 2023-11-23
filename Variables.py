import numpy as np

# Example Usage for Toyota Rav4
wheelbase = 2.69  # Wheelbase length in meters
track_width = 1.59  # Track width in meters
steering_ratio = 14.3  # Steering ratio

# Steering angle parameters
steering_angle = 90 # Steering angle in degrees

# Simulation parameters
time_step = 0.1  # Time interval for simulation
total_time = 5  # Total simulation time
speed = 1  # Speed in meters per second

# Camera parameters
camera_matrix = np.load("camera_matrix.npy")

dist = np.load("distortion_coefficients.npy")

video_path = "ReversingCamera.mp4"

# Camera pose with respect to the center of the rear axle of the vehicle
translation_vector = np.array([0, 0, 0], dtype=np.float32)  # Translation along z-axis
rotation_vector = np.array([np.radians(-70), 0, 0], dtype=np.float32)  # Rotation around x-axis