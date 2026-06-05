from ultralytics import YOLO
import cv2
import os

# Load a pretrained model (recommended)
model = YOLO("yolov8n.pt")  # nano model (fastest)


# Train
model.train(
    data="yolo_params.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="wrinkle_detector",
    single_cls=True
)