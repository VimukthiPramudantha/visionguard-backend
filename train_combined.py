#!/usr/bin/env python
"""
train_combined.py  —  VisionGuard Combined Dataset Trainer
===========================================================
Merges dataset/ and dataset2/ into a single combined dataset, then retrains
the current model on it so the final weights contain knowledge from BOTH datasets.

Class Mapping
-------------
  Unified (6 classes): bicycle, bus, car, motorbike, person, truck
  Dataset 1 (6):       bicycle(0), bus(1), car(2), motorbike(3), person(4), truck(5)
                       -> already aligned, no remapping needed
  Dataset 2 (5):       bus(0), car(1), motorcycle(2), person(3), truck(4)
                       ->  0->1, 1->2, 2->3, 3->4, 4->5  (bicycle stays absent)

Usage:
    py train_combined.py                            # defaults
    py train_combined.py --epochs 150 --batch 8
    py train_combined.py --no-merge                 # skip merge, only train
    py train_combined.py --merge-only               # only merge, skip training
    py train_combined.py --device cpu

Requirements:
    py -m pip install ultralytics pyyaml
"""

import argparse
import os
import re
import shutil
import sys
import time
from pathlib import Path

import torch
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent

DATASET1_DIR = _SCRIPT_DIR / "dataset"
DATASET2_DIR = _SCRIPT_DIR / "dataset2"
COMBINED_DIR = _SCRIPT_DIR / "dataset_combined"
COMBINED_YAML = COMBINED_DIR / "data.yaml"

TRAINED_WEIGHTS = _SCRIPT_DIR / "models" / "trained" / "best.pt"
BASE_WEIGHTS    = _SCRIPT_DIR / "yolo11n.pt"

_DEFAULT_OUT = str(_SCRIPT_DIR / "runs" / "detect")

# ─────────────────────────────────────────────────────────────────────────────
# Unified class definition
# ─────────────────────────────────────────────────────────────────────────────
UNIFIED_CLASSES = ["bicycle", "bus", "car", "motorbike", "person", "truck"]

# Dataset 2 class index -> Unified class index
# ds2: bus(0), car(1), motorcycle(2), person(3), truck(4)
DS2_REMAP = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def print_banner() -> None:
    print("=" * 70)
    print("   VisionGuard — Combined Dataset Trainer (YOLOv11)")
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
        print("[-] ultralytics not installed. Run:  py -m pip install ultralytics")
        sys.exit(1)
    print()


def count_files(directory: Path, pattern: str = "*.*") -> int:
    """Count files matching a glob pattern in a directory."""
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


def remap_label_file(src: Path, dst: Path, remap: dict) -> None:
    """
    Copy a YOLO label file from src to dst, remapping class indices.
    Each line format: <class_id> <cx> <cy> <w> <h>
    """
    lines_out = []
    with open(src, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls_id = int(parts[0])
            new_cls = remap.get(cls_id, cls_id)
            parts[0] = str(new_cls)
            lines_out.append(" ".join(parts))
    with open(dst, "w") as fh:
        fh.write("\n".join(lines_out) + "\n")


def copy_label_unchanged(src: Path, dst: Path) -> None:
    """Copy a label file without remapping."""
    shutil.copy2(src, dst)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset merge
# ─────────────────────────────────────────────────────────────────────────────

def merge_split(
    ds1_split_dir: Path,
    ds2_split_dir: Path,
    out_split_dir: Path,
    ds2_remap: dict,
    split_name: str,
) -> tuple:
    """
    Merge a single split (train / valid / test) from both datasets.
    Returns (count_ds1, count_ds2) of images copied.
    """
    img_out = out_split_dir / "images"
    lbl_out = out_split_dir / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    count_ds1 = 0
    count_ds2 = 0

    # ── Dataset 1: copy images + labels unchanged ──────────────────────────
    ds1_img_dir = ds1_split_dir / "images"
    ds1_lbl_dir = ds1_split_dir / "labels"

    if ds1_img_dir.exists():
        for img_src in sorted(ds1_img_dir.iterdir()):
            if img_src.suffix.lower() not in image_exts:
                continue
            new_name = f"ds1_{img_src.name}"
            shutil.copy2(img_src, img_out / new_name)

            lbl_src = ds1_lbl_dir / (img_src.stem + ".txt")
            if lbl_src.exists():
                copy_label_unchanged(lbl_src, lbl_out / f"ds1_{lbl_src.name}")
            else:
                # Create empty label file so YOLO doesn't complain
                (lbl_out / f"ds1_{img_src.stem}.txt").touch()

            count_ds1 += 1

    # ── Dataset 2: copy images + remap labels ─────────────────────────────
    ds2_img_dir = ds2_split_dir / "images"
    ds2_lbl_dir = ds2_split_dir / "labels"

    if ds2_img_dir.exists():
        for img_src in sorted(ds2_img_dir.iterdir()):
            if img_src.suffix.lower() not in image_exts:
                continue
            new_name = f"ds2_{img_src.name}"
            shutil.copy2(img_src, img_out / new_name)

            lbl_src = ds2_lbl_dir / (img_src.stem + ".txt")
            if lbl_src.exists():
                remap_label_file(lbl_src, lbl_out / f"ds2_{lbl_src.name}", ds2_remap)
            else:
                (lbl_out / f"ds2_{img_src.stem}.txt").touch()

            count_ds2 += 1

    print(
        f"   [{split_name}] ds1={count_ds1} images  |  ds2={count_ds2} images  "
        f"->  {count_ds1 + count_ds2} total"
    )
    return count_ds1, count_ds2


def write_combined_yaml() -> None:
    """Write the unified data.yaml for the combined dataset."""
    yaml_content = f"""# Combined dataset — dataset + dataset2
# Unified class list ({len(UNIFIED_CLASSES)} classes)
train: train/images
val:   valid/images
test:  test/images

nc: {len(UNIFIED_CLASSES)}
names: {UNIFIED_CLASSES}
"""
    with open(COMBINED_YAML, "w") as fh:
        fh.write(yaml_content)
    print(f"[+] Wrote {COMBINED_YAML}")


def merge_datasets(force: bool = False) -> None:
    """
    Merge dataset/ and dataset2/ into dataset_combined/.
    Skips if already done, unless force=True.
    """
    print("\n--- Dataset Merge ---")

    if COMBINED_DIR.exists() and not force:
        train_count = count_files(COMBINED_DIR / "train" / "images")
        print(
            f"[*] dataset_combined/ already exists ({train_count} train images). "
            "Skipping merge. Use --force-merge to re-run."
        )
        return

    if COMBINED_DIR.exists():
        print(f"[*] Removing existing {COMBINED_DIR} ...")
        shutil.rmtree(COMBINED_DIR)

    COMBINED_DIR.mkdir(parents=True)

    # Validate source datasets
    for ds_dir, name in [(DATASET1_DIR, "dataset"), (DATASET2_DIR, "dataset2")]:
        if not ds_dir.exists():
            print(f"[-] Error: {name}/ not found at {ds_dir}")
            sys.exit(1)

    print(f"[*] Merging '{DATASET1_DIR.name}' + '{DATASET2_DIR.name}' -> '{COMBINED_DIR.name}/'")
    print(f"[*] Unified classes: {UNIFIED_CLASSES}")
    print(f"[*] Dataset-2 class remap: {DS2_REMAP}\n")

    total_ds1, total_ds2 = 0, 0
    for split in ("train", "valid", "test"):
        c1, c2 = merge_split(
            ds1_split_dir=DATASET1_DIR / split,
            ds2_split_dir=DATASET2_DIR / split,
            out_split_dir=COMBINED_DIR / split,
            ds2_remap=DS2_REMAP,
            split_name=split,
        )
        total_ds1 += c1
        total_ds2 += c2

    write_combined_yaml()

    print(
        f"\n[+] Merge complete! "
        f"Total: ds1={total_ds1}  ds2={total_ds2}  combined={total_ds1 + total_ds2} images\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def pick_weights(args):
    """
    Decide which weights to start from:
    1. User-specified --model (custom path or YOLO hub name)
    2. Existing trained best.pt (fine-tune on top of current model)
    3. Default yolo11n.pt pretrained weights
    """
    if args.model:
        return args.model

    if TRAINED_WEIGHTS.exists():
        print(f"[+] Found existing trained model: {TRAINED_WEIGHTS}")
        print("    Will fine-tune on the combined dataset.")
        return str(TRAINED_WEIGHTS)

    if BASE_WEIGHTS.exists():
        print(f"[*] Using base pretrained weights: {BASE_WEIGHTS}")
        return str(BASE_WEIGHTS)

    print("[*] Downloading yolo11n.pt from Ultralytics hub ...")
    return "yolo11n.pt"


def run_training(args) -> None:
    """Load weights and train on the combined dataset."""
    if not COMBINED_YAML.exists():
        print(f"[-] Error: {COMBINED_YAML} not found. Run merge first (remove --no-merge).")
        sys.exit(1)

    device = (0 if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"[*] Device     : {device}")

    weights = pick_weights(args)
    print(f"[*] Weights    : {weights}")
    print(f"[*] Dataset    : {COMBINED_YAML}")
    print(f"[*] Epochs     : {args.epochs}  |  Batch : {args.batch}  |  ImgSz : {args.imgsz}\n")

    model = YOLO(weights)

    print("--- Training Started ---\n")
    t0 = time.time()

    try:
        model.train(
            data        = str(COMBINED_YAML),
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
    except Exception as e:
        print(f"[-] Training error: {e}")
        sys.exit(1)

    elapsed = time.time() - t0
    h, rem = divmod(int(elapsed), 3600)
    m, s   = divmod(rem, 60)
    print("\n" + "=" * 70)
    print("  Training Complete!")
    print(f"  Total time  : {h:02d}h {m:02d}m {s:02d}s")
    print(f"  Results dir : {_DEFAULT_OUT}/{args.name}/")
    print("=" * 70)

    # Copy best.pt -> models/trained/best.pt
    best_src = Path(_DEFAULT_OUT) / args.name / "weights" / "best.pt"
    dest_dir = _SCRIPT_DIR / "models" / "trained"
    dest_dir.mkdir(parents=True, exist_ok=True)
    best_dst = dest_dir / "best.pt"

    if best_src.exists():
        # Backup previous model
        if best_dst.exists():
            backup = dest_dir / "best_prev.pt"
            shutil.copy2(best_dst, backup)
            print(f"[*] Previous model backed up : {backup}")
        shutil.copy2(best_src, best_dst)
        print(f"[+] Best model saved to      : {best_dst}")
    else:
        print(f"[-] Warning: best.pt not found at {best_src}")

    # ── Evaluation ────────────────────────────────────────────────────────
    print("\n--- Running Evaluation on Test Split ---")
    try:
        val_results = model.val(
            data   = str(COMBINED_YAML),
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

    if best_dst.exists():
        print(f"\n[*] Best weights : {best_dst}")
        print("    Export ONNX     : yolo export model=models/trained/best.pt format=onnx")
        print("    Export TensorRT : yolo export model=models/trained/best.pt format=engine")

    print("\nDone. ✓\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="VisionGuard — Combined Dataset Trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Dataset merge options ──────────────────────────────────────────────
    merge_group = parser.add_argument_group("Dataset Merge Options")
    merge_group.add_argument(
        "--no-merge",
        action="store_true",
        help="Skip dataset merge (use existing dataset_combined/ as-is)",
    )
    merge_group.add_argument(
        "--merge-only",
        action="store_true",
        help="Only merge datasets, do not start training",
    )
    merge_group.add_argument(
        "--force-merge",
        action="store_true",
        help="Force re-merge even if dataset_combined/ already exists",
    )

    # ── Training options ───────────────────────────────────────────────────
    train_group = parser.add_argument_group("Training Options")
    train_group.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Path to weights to start from. "
            "Defaults to models/trained/best.pt (if it exists) then yolo11n.pt"
        ),
    )
    train_group.add_argument("--epochs",   type=int,   default=100,    help="Training epochs")
    train_group.add_argument("--batch",    type=int,   default=6,      help="Batch size")
    train_group.add_argument("--imgsz",    type=int,   default=640,    help="Input image size")
    train_group.add_argument("--workers",  type=int,   default=4,      help="DataLoader workers")
    train_group.add_argument("--patience", type=int,   default=20,     help="Early-stop patience")
    train_group.add_argument("--lr0",      type=float, default=0.005,  help="Initial learning rate (lower for fine-tuning)")
    train_group.add_argument("--lrf",      type=float, default=0.01,   help="Final LR fraction of lr0")
    train_group.add_argument("--device",   type=str,   default="auto", help="Device: '0', 'cpu', or 'auto'")
    train_group.add_argument("--name",     type=str,   default="combined_train", help="Experiment name")
    train_group.add_argument("--resume",   action="store_true", help="Resume from last checkpoint")
    train_group.add_argument("--cache",    action="store_true", help="Cache images in RAM")

    return parser.parse_args()


def main() -> None:
    print_banner()
    args = parse_args()

    # ── Step 1: Merge ──────────────────────────────────────────────────────
    if not args.no_merge:
        merge_datasets(force=args.force_merge)
    else:
        print("[*] Skipping merge (--no-merge flag set).\n")

    if args.merge_only:
        print("[*] --merge-only flag set. Stopping before training.")
        return

    # ── Step 2: Train ─────────────────────────────────────────────────────
    run_training(args)


if __name__ == "__main__":
    main()
