import os
import numpy as np
import matplotlib.pyplot as plt
from roboflow import Roboflow
from ultralytics import YOLO

os.makedirs("datasets", exist_ok=True)

from roboflow import Roboflow
rf = Roboflow(api_key="KSnnwF2Lir5l7ni3srs4")
project = rf.workspace("roboflow-gw7yv").project("vehicles-openimages")
dataset = project.version(1).download("yolov8")

model = YOLO("ObjectDetection/YoloModel/yolov8n.pt")

results = model.train(
    data = "ObjectDetection/datasets/Vehicles-OpenImages-1/data.yaml",
    imgsz = 416,
    epochs = 50,
    batch = 16,
    name = "Vehicle_yolov8n",
)