#!/usr/bin/env python
"""
train.py - Updated for better stability on GTX 1650
"""
import os
import sys
import argparse
import shutil
import torch
from ultralytics import YOLO

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_OUTPUT = os.path.join(_PROJECT_ROOT, "runs", "detect")


def check_paths():
    yaml_path = os.path.join("dataset", "data.yaml")
    if not os.path.exists(yaml_path):
        print(f"[-] Error: data.yaml not found at {os.path.abspath(yaml_path)}")
        sys.exit(1)
    
    pretrained_dir = os.path.join("models", "pretrained")
    os.makedirs(pretrained_dir, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="YOLO v11 Vehicle Detection Trainer")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs (default: 100)")
    parser.add_argument("--batch", type=int, default=6, help="Batch size - lower for GTX 1650 (recommended 4-8)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--name", type=str, default="train", help="Experiment name")
    args = parser.parse_args()

    print("=" * 70)
    print("           VisionGuard - YOLO11 Vehicle Training")
    print("=" * 70)

    check_paths()

    # Device
    if args.device == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"
        if device == 0:
            print(f"[+] GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = args.device

    # Model path
    model_path = os.path.join("models", "pretrained", "yolo11n.pt")
    if not os.path.exists(model_path):
        model_path = "yolo11n.pt"  

    print(f"[*] Using model: {model_path}")
    model = YOLO(model_path)

    yaml_config = os.path.join("dataset", "data.yaml")
    print(f"[*] Dataset: {yaml_config}")

    try:
        results = model.train(
            data=yaml_config,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            name=args.name,
            project=_DEFAULT_OUTPUT,
            patience=20,
            plots=True,
            save=True,
            exist_ok=True,
            pretrained=True,
            augment=True,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
        )

        best_run_path = os.path.join(_DEFAULT_OUTPUT, args.name, "weights", "best.pt")
        dest_path = os.path.join("models", "trained", "best.pt")
        
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy(best_run_path, dest_path)
        print(f"[+] Best model saved to: {dest_path}")

    except Exception as e:
        print(f"[-] Training error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()