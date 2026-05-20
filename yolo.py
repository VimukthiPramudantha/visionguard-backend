from ultralytics import YOLO

# Load latest lightweight model
model = YOLO("yolo11n.pt")

# Run webcam detection
model.predict(
    source=0,
    show=True,
    device=0
)