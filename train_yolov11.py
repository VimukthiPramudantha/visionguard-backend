#!/usr/bin/env python
"""
train.py  —  VisionGuard YOLOv11 Trainer
==========================================
Dataset  : Pedestrian and Vehicle (6 classes)
Classes  : bicycle, bus, car, motorbike, person, truck
Images   : ~1142 (train / val / test split)
Hardware : CUDA GPU (auto-detected, optimised for GTX 1650)

Usage:
    py train_yolov11.py                              # defaults (batch=6, yolo11n)
    py train_yolov11.py --batch 8 --epochs 150
    py train_yolov11.py --model yolo11s.pt --resume
    py train_yolov11.py --device cpu                 # force CPU

Requirements:
    py -m pip install ultralytics
    py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from ultralytics import YOLO

_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_DEFAULT_OUT  = str(_SCRIPT_DIR / "runs" / "detect")


def check_environment() -> None:
    """Print Python / PyTorch / CUDA info and abort if packages are missing."""
    print("=" * 70)
    print("           VisionGuard - YOLO11 Vehicle Training")
    print("=" * 70)

    py = sys.version_info
    print(f"\n[*] Python     : {py.major}.{py.minor}.{py.micro}")
    print(f"[*] PyTorch    : {torch.__version__}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            vram = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
            print(f"[+] CUDA GPU {i} : {name}  ({vram:.1f} GB VRAM)")
    else:
        print("[-] CUDA       : NOT available — training will run on CPU (slow!)")

    try:
        import ultralytics
        print(f"[*] Ultralytics: {ultralytics.__version__}")
    except ImportError:
        print("\n[-] ultralytics not installed. Run:\n    py -m pip install ultralytics")
        sys.exit(1)

    print()


def check_paths() -> Path:
    """Validate dataset YAML exists and ensure model directories are present."""
    yaml_path = _SCRIPT_DIR / "dataset" / "data.yaml"

    if not yaml_path.exists():
        print(f"[-] Error: data.yaml not found at {yaml_path}")
        sys.exit(1)

    import yaml
    with open(yaml_path) as fh:
        cfg = yaml.safe_load(fh)

    print(f"[*] Classes ({cfg['nc']}) : {cfg['names']}")
    base = yaml_path.parent
    for split in ("train", "val", "test"):
        rel = cfg.get(split, "")
        if not rel:
            continue
        p = (base / rel).resolve()
        exists = p.exists()
        count  = len(list(p.glob("*.*"))) if exists else 0
        status = f"OK  ({count} images)" if exists else "MISSING"
        print(f"[{'+'if exists else '-'}] {split:<5} split : {p}  [{status}]")

    print()

    (_SCRIPT_DIR / "models" / "pretrained").mkdir(parents=True, exist_ok=True)
    (_SCRIPT_DIR / "models" / "trained").mkdir(parents=True, exist_ok=True)

    return yaml_path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VisionGuard YOLO11 Vehicle Detection Trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        choices=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"],
        help="Pretrained weights (nano=fastest, xlarge=most accurate)",
    )
    parser.add_argument("--epochs",   type=int,   default=100,   help="Training epochs")
    parser.add_argument("--batch",    type=int,   default=6,     help="Batch size (GTX 1650: recommend 4-8)")
    parser.add_argument("--imgsz",    type=int,   default=640,   help="Input image size")
    parser.add_argument("--workers",  type=int,   default=4,     help="DataLoader workers")
    parser.add_argument("--patience", type=int,   default=20,    help="Early-stop patience epochs")
    parser.add_argument("--lr0",      type=float, default=0.01,  help="Initial learning rate")
    parser.add_argument("--lrf",      type=float, default=0.01,  help="Final LR fraction of lr0")
    parser.add_argument("--device",   type=str,   default="auto",help="Device: '0', 'cpu', or 'auto'")
    parser.add_argument("--name",     type=str,   default="train",help="Experiment name")
    parser.add_argument("--resume",   action="store_true",       help="Resume from last checkpoint")
    parser.add_argument("--cache",    action="store_true",       help="Cache images in RAM (needs 4+ GB free)")
    return parser.parse_args()


def main() -> None:
    check_environment()
    args = parse_args()

    yaml_path = check_paths()

    if args.device == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            print(f"[+] GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = args.device
    print(f"[*] Device : {device}\n")

    local_weights = _SCRIPT_DIR / "models" / "pretrained" / args.model
    model_path = str(local_weights) if local_weights.exists() else args.model
    print(f"[*] Model  : {model_path}")

    model = YOLO(model_path)

    print(f"[*] Dataset: {yaml_path}")
    print(f"[*] Epochs : {args.epochs}  |  Batch : {args.batch}  |  ImgSz : {args.imgsz}\n")

    try:
        print("--- Training Started ---\n")
        t0 = time.time()

        results = model.train(
            data        = str(yaml_path),
            epochs      = args.epochs,
            imgsz       = args.imgsz,
            batch       = args.batch,
            workers     = args.workers,
            device      = device,
            lr0         = args.lr0,
            lrf         = args.lrf,
            patience    = args.patience,
            name        = args.name,
            project     = _DEFAULT_OUT,
            resume      = args.resume,
            cache       = args.cache,
            pretrained  = True,
            augment     = True,
            degrees     = 10.0,
            translate   = 0.1,
            scale       = 0.5,
            fliplr      = 0.5,
            flipud      = 0.0,
            mosaic      = 1.0,
            hsv_h       = 0.015,
            hsv_s       = 0.7,
            hsv_v       = 0.4,
            save        = True,
            save_period = 10,
            plots       = True,
            exist_ok    = True,
            verbose     = True,
        )

        elapsed = time.time() - t0
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)

        print("\n" + "=" * 70)
        print("  Training Complete!")
        print(f"  Total time  : {h:02d}h {m:02d}m {s:02d}s")
        print(f"  Results dir : {_DEFAULT_OUT}/{args.name}/")
        print("=" * 70)

        best_src = Path(_DEFAULT_OUT) / args.name / "weights" / "best.pt"
        best_dst = _SCRIPT_DIR / "models" / "trained" / "best.pt"

        if best_src.exists():
            shutil.copy(best_src, best_dst)
            print(f"[+] Best model saved to : {best_dst}")
        else:
            print(f"[-] Warning: best.pt not found at {best_src}")

    except Exception as e:
        print(f"[-] Training error: {e}")
        sys.exit(1)

    print("\n--- Running Evaluation on Test Split ---")
    try:
        val_results = model.val(
            data   = str(yaml_path),
            split  = "test",
            imgsz  = args.imgsz,
            device = device,
            plots  = True,
        )

        if hasattr(val_results, "box"):
            b = val_results.box
            print("\n--- Test Set Metrics ---")
            print(f"  mAP@50    : {b.map50:.4f}")
            print(f"  mAP@50-95 : {b.map:.4f}")
            print(f"  Precision : {b.mp:.4f}")
            print(f"  Recall    : {b.mr:.4f}")
    except Exception as e:
        print(f"[-] Evaluation error: {e}")

    best_dst = _SCRIPT_DIR / "models" / "trained" / "best.pt"
    if best_dst.exists():
        print(f"\n[*] Best weights : {best_dst}")
        print("    Export ONNX      : yolo export model=models/trained/best.pt format=onnx")
        print("    Export TensorRT  : yolo export model=models/trained/best.pt format=engine")

    print("\nDone. ✓\n")


if __name__ == "__main__":
    main()
