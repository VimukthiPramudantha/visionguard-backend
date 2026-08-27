import os
import sys

_yolo_model = None
_VEHICLE_CLASSES = {"bicycle", "bus", "car", "motorbike", "truck"}
_BOX_COLORS = {
    "vehicle": (0, 140, 255),   
    "person":  (0, 220, 100),   
}

def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            model_path = r"d:\Projects\VisionGuard\visionguard-backend\runs\detect\combined_train\weights\best.pt"
            if os.path.exists(model_path):
                _yolo_model = YOLO(model_path)
                print(f"[VisionGuard] Model loaded. Classes: {_yolo_model.names}")
            else:
                print(f"YOLO model not found at {model_path}")
                _yolo_model = False
        except ImportError as e:
            print(f"ultralytics import failed: {e}")
            print(f"Python path: {sys.path}")
            print(f"Python executable: {sys.executable}")
            _yolo_model = False
        except Exception as e:
            print(f"Failed to load YOLO: {e}")
            _yolo_model = False
    return _yolo_model if _yolo_model is not False else None

def run_detection(model, frame):
    import cv2

    results = model(
        frame,
        conf=0.45,
        iou=0.35,
        augment=False,
        imgsz=640,
        device=0,
        verbose=False,
    )

    annotated = frame.copy()
    detections = []
    for box in results[0].boxes:
        cls_id   = int(box.cls[0])
        conf_val = float(box.conf[0])
        raw_name = model.names[cls_id]

        label = "vehicle" if raw_name in _VEHICLE_CLASSES else "person"
        color = _BOX_COLORS.get(label, (200, 200, 200))

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        detections.append({
            "label": label,
            "confidence": conf_val,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf_val:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            annotated, text,
            (x1 + 3, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 0, 0), 1, cv2.LINE_AA,
        )

    return annotated, detections
