from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.predict(
    source=0,
    device=0,   # GPU
    show=True
)