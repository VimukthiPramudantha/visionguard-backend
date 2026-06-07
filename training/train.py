#!/usr/bin/env python
"""
train.py
YOLO v11 Vehicle Detection Training Pipeline.
Auto-detects CUDA GPU (GeForce GTX 1650) and handles paths gracefully.
"""
import os
import sys
import argparse
import shutil
import torch
from ultralytics import YOLO

# Absolute path to the project root (parent of this script's directory)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_OUTPUT = os.path.join(_PROJECT_ROOT, "runs", "detect")

def check_paths():
    """Verify that dataset and data.yaml exist before starting."""
    yaml_path = os.path.join("dataset", "data.yaml")
    if not os.path.exists(yaml_path):
        print(f"[-] Error: Dataset config file not found at: {os.path.abspath(yaml_path)}")
        print("Please ensure you run this script from the project root folder.")
        sys.exit(1)
        
    pretrained_dir = os.path.join("models", "pretrained")
    pretrained_model = os.path.join(pretrained_dir, "yolo11n.pt")
    if not os.path.exists(pretrained_model):
        print(f"[!] Warning: Pretrained weight not found at {pretrained_model}")
        print("[*] YOLO will automatically download it to the current directory.")
        # Create directory if missing
        os.makedirs(pretrained_dir, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="YOLO v11 Training Pipeline")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--imgsz", type=int, default=640, help="Target image size (default: 640)")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Execution device: '0', 'cpu', or 'auto'")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    parser.add_argument("--project", type=str, default=_DEFAULT_OUTPUT, help="Absolute root folder for training output (default: <project_root>/runs/detect)")
    parser.add_argument("--name", type=str, default="train", help="Sub-folder name inside --project (default: train)")
    args = parser.parse_args()

    print("=" * 60)
    print("           YOLO v11 Vehicle Detection Trainer")
    print("=" * 60)
    
    # 1. Verify environment paths
    check_paths()

    # 2. Configure training device (GPU auto-detection)
    if args.device == "auto":
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"[+] GPU Acceleration Detected: {device_name}")
            device = 0
        else:
            print("[!] No GPU acceleration found. Falling back to CPU training.")
            device = "cpu"
    else:
        device = args.device
        print(f"[*] Manually configured execution device: {device}")

    # Determine weight file path
    pretrained_dir = os.path.join("models", "pretrained")
    model_path = os.path.join(pretrained_dir, "yolo11n.pt")
    
    # Fallback to local yolo11n.pt if pretrained weight dir download happens outside
    if not os.path.exists(model_path) and os.path.exists("yolo11n.pt"):
        model_path = "yolo11n.pt"

    print(f"[*] Loading YOLO v11 model: {model_path}")
    model = YOLO(model_path)

    # 3. Start model training
    yaml_config = os.path.join("dataset", "data.yaml")
    print(f"[*] Loading Dataset config: {yaml_config}")
    print(f"[*] Hyperparameters: Epochs={args.epochs}, Batch={args.batch}, Imgsz={args.imgsz}, LR={args.lr0}")
    print("[-] Training is starting. Press Ctrl+C to abort.")
    print("-" * 60)
    
    try:
        abs_project = os.path.abspath(args.project)
        print(f"[*] Training output directory: {os.path.join(abs_project, args.name)}")
        results = model.train(
            data=yaml_config,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            lr0=args.lr0,
            resume=args.resume,
            project=abs_project,
            name=args.name,
            plots=True
        )
        print("[+] Training completed successfully!")
        
        # 4. Copy best.pt weights to models/trained/best.pt as a permanent backup
        best_run_path = os.path.join(abs_project, args.name, "weights", "best.pt")
        dest_trained_dir = os.path.join("models", "trained")
        dest_best_path = os.path.join(dest_trained_dir, "best.pt")
        
        if os.path.exists(best_run_path):
            os.makedirs(dest_trained_dir, exist_ok=True)
            shutil.copy(best_run_path, dest_best_path)
            print(f"[+] Permanent backup created successfully at: {dest_best_path}")
            
            # If the model has downloaded to root, move it to models/pretrained to clean workspace
            if os.path.exists("yolo11n.pt"):
                shutil.move("yolo11n.pt", os.path.join(pretrained_dir, "yolo11n.pt"))
        else:
            print("[!] Warning: Could not locate 'best.pt' in training weights output directory.")
            
    except KeyboardInterrupt:
        print("\n[!] Training manually interrupted by user.")
    except Exception as e:
        print(f"\n[-] Critical training error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
