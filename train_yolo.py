from ultralytics import YOLO
import os

assert os.path.exists("data/yolo.yaml"), "Missing YAML"
assert os.path.exists("data/yolo/images"), "Missing images"
assert os.path.exists("data/yolo/labels"), "Missing labels"

print("Starting YOLO training...")

model = YOLO("yolov8n.pt")

model.train(
    data="data/yolo.yaml",
    epochs=20,
    imgsz=224,
    batch=16,
    device=0  # or "cpu"
)
