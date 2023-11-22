import cv2
import numpy as np

# Define the size of the checkerboard (number of internal corners)
chess_board_size = (7, 9)

# Termination criteria for the corner sub-pixel algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from selected frames
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Create a video capture object
cap = cv2.VideoCapture("CalibrationVideo.mp4")  # Replace "calibration_video.mp4" with the actual video file

# Counter to keep track of frames
frame_count = 0

while True:
    # Read a single frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Skip frames if not the desired frame (every 30 frames)
    if frame_count % 30 != 0:
        continue

    # Resize the frame to 960x540
    frame = cv2.resize(frame, (960, 540))

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chess_board_size, None)

    if ret:
        # Refine the corners to sub-pixel accuracy
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Draw and display the corners
        frame = cv2.drawChessboardCorners(frame, chess_board_size, corners, ret)
        cv2.imshow('Calibration', frame)

        # Prepare object points (0,0,0), (1,0,0), (2,0,0), ..., (6,5,0)
        objp = np.zeros((chess_board_size[0] * chess_board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chess_board_size[0], 0:chess_board_size[1]].T.reshape(-1, 2)

        objpoints.append(objp)
        imgpoints.append(corners)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()

# Perform camera calibration
h, w = gray.shape
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

# Save the calibration parameters to a file
np.save("camera_matrix.npy", mtx)
np.save("distortion_coefficients.npy", dist)

print("Calibration completed. Camera matrix and distortion coefficients saved.")
