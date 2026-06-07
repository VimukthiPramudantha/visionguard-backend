# app/ai/detector.py
import cv2
import time
import json
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO


class VisionGuardDetector:
    def __init__(
        self,
        model_path="models/trained/best.pt",
        conf=0.4,
        iou=0.45,
        imgsz=640,
        device="0"
    ):
        print(f"[*] Loading YOLO11 model: {model_path}")
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        
        # Get class names from model
        self.class_names = self.model.names
        print(f"[+] Model loaded with {len(self.class_names)} classes: {list(self.class_names.values())}")

    def detect_frame(self, frame):
        """Process single frame and return results + annotated frame"""
        start_time = time.time()
        
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]

        inference_time = time.time() - start_time
        fps = 1 / inference_time if inference_time > 0 else 0

        # Extract detections
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = self.class_names[cls_id]
            
            detections.append({
                "class": class_name,
                "confidence": round(conf, 4),
                "bbox": [x1, y1, x2, y2],
            })

        annotated_frame = results.plot()

        return {
            "detections": detections,
            "annotated_frame": annotated_frame,
            "fps": round(fps, 2),
            "inference_time_ms": round(inference_time * 1000, 2)
        }

    def process_images(self, image_dir="samples/images", save_results=True):
        """Process all images in a directory for batch testing"""
        image_path = Path(image_dir)
        if not image_path.exists():
            print(f"[-] Error: Image directory not found: {image_path}")
            return

        output_dir = Path("runs/detect/image_test")
        output_dir.mkdir(parents=True, exist_ok=True)

        image_files = list(image_path.glob("*.jpg")) + list(image_path.glob("*.jpeg")) + list(image_path.glob("*.png"))
        
        print(f"[+] Found {len(image_files)} images in {image_dir}")
        print(f"[+] Processing with confidence threshold: {self.conf}\n")

        for img_file in image_files:
            print(f"[*] Processing: {img_file.name}")
            
            frame = cv2.imread(str(img_file))
            if frame is None:
                print(f"    [-] Failed to read image: {img_file.name}")
                continue

            result = self.detect_frame(frame)
            
            print(f"    → Detected {len(result['detections'])} vehicles | FPS: {result['fps']}")

            # Save annotated image
            if save_results:
                output_path = output_dir / f"detected_{img_file.name}"
                cv2.imwrite(str(output_path), result["annotated_frame"])
                print(f"    → Saved: {output_path.name}")

            # Print detection details
            for det in result["detections"]:
                print(f"      • {det['class']} ({det['confidence']*100:.1f}%)")

            print("-" * 50)

        print(f"\n[+] Batch processing completed! Results saved to: {output_dir}")


# CLI for easy testing
if __name__ == "__main__":
    detector = VisionGuardDetector(conf=0.35)   # Slightly lower threshold for testing
    
    # === Test on your sample images ===
    detector.process_images(
        image_dir="samples/images", 
        save_results=True
    )