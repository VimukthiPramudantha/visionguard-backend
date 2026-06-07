# app/ai/detector.py
import cv2
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

class VisionGuardDetector:
    def __init__(self, model_path="models/trained/best.pt", conf=0.25):
        print(f"[*] Attempting to load model: {model_path}")
        self.model_path = model_path
        
        if not Path(model_path).exists():
            print(f"[-] Model NOT found at {model_path}")
            alt = "models/trained/best.pt"
            if Path(alt).exists():
                model_path = alt
                print(f"[+] Using alternative path: {alt}")
        
        self.model = YOLO(model_path)
        self.conf = conf
        self.class_names = self.model.names
        print(f"[+] SUCCESS: Model loaded with {len(self.class_names)} classes")
        print(f"    Classes: {list(self.class_names.values())}")

    def process_images(self, image_dir="samples/images", detected_save_dir="snapshots/detected_vehicles"):
        image_path = Path(image_dir)
        save_path = Path(detected_save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        image_files = list(image_path.glob("*.jpg")) + list(image_path.glob("*.jpeg")) + list(image_path.glob("*.png"))
        
        print(f"\n[+] Found {len(image_files)} images")
        
        for img_file in image_files:
            frame = cv2.imread(str(img_file))
            if frame is None:
                print(f"[-] Could not read {img_file.name}")
                continue

            print(f"\n[*] Testing: {img_file.name} (size: {frame.shape})")
            
            results = self.model.predict(
                source=frame,
                conf=self.conf,
                verbose=True,          # ← More detailed output
                save=False
            )[0]

            detections = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = self.class_names[cls_id]
                detections.append(f"{class_name} ({conf:.3f})")

            print(f"    → Detections: {len(detections)} | Conf={self.conf}")
            if detections:
                print("    → Found:", detections)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = save_path / f"detected_{timestamp}_{img_file.name}"
                annotated = results.plot()
                cv2.imwrite(str(output_file), annotated)
                print(f"    ✅ SAVED: {output_file.name}")
            else:
                print("    → No detections")

if __name__ == "__main__":
    detector = VisionGuardDetector(conf=0.15)   # Very low threshold for debugging
    detector.process_images()