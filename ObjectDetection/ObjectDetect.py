import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Instantiate the model
model = YOLO("ObjectDetection/SavedModels/detect/Vehicle_yolov8n2/weights/best.pt")

cap = cv2.VideoCapture("C:/Users/Zaid/Documents/GitHub/ADAS/ReversingCamera.mp4")	

results = model.predict(source="ObjectDetection/datasets/Vehicles-OpenImages-1/valid/images", show=True, save=True, save_txt=True)

cv2.destroyAllWindows()

# while(True):  # Change 'cap' to 'True'
#     ret, frame = cap.read()
    
#     # Check if the video capture was successful
#     if not ret:
#         break
    
#     frame = cv2.resize(frame, (960, 540))

#     results = model.predict(source = frame, show=True)
    
#     # cv2.imshow('frame', results.imgs[0])

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
