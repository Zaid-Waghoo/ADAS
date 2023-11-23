# ADAS Project

This project is designed for Advanced Driver Assistance Systems (ADAS) and focuses on visualizing vehicle trajectories based on Ackermann steering.

## Table of Contents
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Running the Code](#running-the-code)

## Requirements
Make sure you have the following installed:
- Python 3.x
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)
- Matplotlib (`pip install matplotlib`)

## Project Structure
- `Variables.py`: Contains project-specific variables.
- `Functions.py`: Includes functions for calculating wheel angles, trajectories, and visualization.
- `SteeringTrajectory.py`: The code snippet performs a trajectory visualization using predefined parameters and functions for calculating vehicle trajectories and visualizing them. The process involves the following steps:
1. **Calculate The Ackermann Angle:**
    - Compute the Ackermann steering angle based on predefined values for wheelbase, steering angle, and steering ratio.

2. **Compute The Trajectories:**
    - Determine trajectories for the center of the rear axle, considering the turning radius from the Instantaneous Center of Curvature (ICC) to the center of the rear axle.
    - Simulate the vehicle's movement by calculating the heading angle, incorporating the vehicle speed in discrete time steps over a specified duration.
    - Update the positions of the vehicle during the simulation. 

3. **Convert Points to 3D:**
   - Transform 2D trajectory points representing the center, inner rear wheel, and outer rear wheel positions to their corresponding 3D representations.
   - Utilize numpy arrays to perform the conversion, adjusting the coordinates and data types accordingly.
   - Introduce a constant depth value to convert 2D points into 3D space, maintaining a consistent depth throughout the trajectories.

4. **Project 3D to 2D:**
   - Utilize the OpenCV library to project 3D trajectory points, including the center, inner rear wheel, and outer rear wheel positions, onto a 2D image plane.
   - Employ the camera parameters such as rotation vector, translation vector, camera matrix, and distortion coefficients to perform the projection.
   - Return the resulting 2D trajectory points for further visualization and analysis.

- `ReversingTrajectoryInteractive.py`: The code snippet makes an interactive trajectory visualization tool that can update the trajectories of the vehicle in real time based on the steering value that can be changed using a slider.  

1. **Interactive Steering Visualization:**
   - Create an interactive visualization using OpenCV to demonstrate the impact of steering angle on vehicle trajectories.
   - Utilize a video file (`ReversingCamera.mp4`) as input to simulate real-time scenarios.
   - Enable user interaction through a trackbar, adjusting the steering angle from -360 to 360 for dynamic trajectory changes.
   - Display the processed frame with overlaid trajectories and real-time steering angle information.


## Running the Code
1. Clone the repository:

```bash
git clone https://github.com/your-username/adas-project.git
cd adas-project

pip install -r requirements.txt

python SteeringTrajectory.py
