#!/usr/bin/env python
"""
predict.py
YOLO v11 Inference & Real-time Prediction Pipeline.
Supports predictions on camera streams, video files, single images, or directories.
"""
import os
import sys
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="YOLO v11 Prediction Script")
    parser.add_argument("--weights", type=str, default="runs/detect/train/weights/best.pt", 
                        help="Path to trained model weights (.pt)")
    parser.add_argument("--source", type=str, default="0", 
                        help="Input source: '0' or '1' for webcam, image path, video path, or directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (default: 640)")
    parser.add_argument("--show", type=str, default="True", choices=["True", "False"], 
                        help="Display detection window in real-time ('True' or 'False')")
    parser.add_argument("--save", action="store_true", default=True, help="Save annotated results to runs/detect/predict")
    parser.add_argument("--device", type=str, default="0", help="Execution device: '0' (GPU) or 'cpu'")
    args = parser.parse_args()

    print("=" * 60)
    print("         YOLO v11 Inference & Real-time Predictor")
    print("=" * 60)

    # 1. Resolve weight file path
    weights_path = args.weights
    if not os.path.exists(weights_path):
        # Fallback path check
        alt_path = os.path.join("models", "trained", "best.pt")
        if os.path.exists(alt_path):
            weights_path = alt_path
        else:
            print(f"[!] Warning: Could not locate custom weights at: {args.weights}")
            print("[*] Running inference using a base pretrained model 'yolo11n.pt' instead.")
            weights_path = os.path.join("models", "pretrained", "yolo11n.pt")
            if not os.path.exists(weights_path):
                weights_path = "yolo11n.pt"  # base download fallback

    print(f"[*] Loading weight file: {weights_path}")
    try:
        model = YOLO(weights_path)
    except Exception as e:
        print(f"[-] Error: Could not load the YOLO model weights: {e}")
        sys.exit(1)

    # Convert show string argument to boolean
    show_window = True if args.show == "True" else False

    # 2. Parse the source parameter
    source = args.source
    # If source is a digit, convert it to integer for OpenCV camera capture
    if source.isdigit():
        source = int(source)
        print(f"[+] Using webcam index: {source} (Press 'q' in the window to exit)")
    else:
        # Check if source exists in filesystem
        if not os.path.exists(source):
            print(f"[-] Error: Target prediction source file or folder does not exist: {source}")
            sys.exit(1)
        print(f"[+] Using media source path: {os.path.abspath(source)}")

    # 3. Run prediction pipeline
    print(f"[*] Starting predictions: conf={args.conf}, iou={args.iou}, imgsz={args.imgsz}, save={args.save}...")
    print("-" * 60)

    try:
        results = model.predict(
            source=source,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            show=show_window,
            save=args.save,
            device=args.device,
            stream=False  # Set to True for memory efficiency if processing long videos
        )
        
        print("\n" + "-" * 60)
        print("[+] Prediction process finished successfully!")
        if args.save:
            # Safely fetch the output directory of the prediction run
            output_dir = getattr(results[0], "save_dir", "runs/detect/predict")
            print(f"[+] Output annotated files saved to: {os.path.abspath(output_dir)}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n[!] Prediction run interrupted by user.")
    except Exception as e:
        print(f"\n[-] Prediction pipeline failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
