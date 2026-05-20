from ultralytics import YOLO
import torch

print("CUDA:", torch.cuda.is_available())

model = YOLO("yolo11n.pt")

model.to("cuda")  # force GPU

model.predict(source=0, show=True)