#!/usr/bin/env python
"""
validate.py
YOLO v11 Model Validation & Evaluation pipeline.
Runs evaluation on validation or test dataset splits and displays clean metrics.
"""
import os
import sys
import argparse
from ultralytics import YOLO
from tabulate import tabulate

def main():
    parser = argparse.ArgumentParser(description="YOLO v11 Model Validation")
    parser.add_argument("--weights", type=str, default="runs/detect/train/weights/best.pt", 
                        help="Path to trained model weights (.pt)")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], 
                        help="Dataset split to run validation on: 'val' or 'test'")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for validation (default: 640)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for validation (default: 8)")
    parser.add_argument("--device", type=str, default="0", help="Execution device: '0' (GPU) or 'cpu'")
    args = parser.parse_args()

    print("=" * 60)
    print(f"         YOLO v11 Model Evaluation Tool ({args.split.upper()} Split)")
    print("=" * 60)

    # 1. Resolve weight file path
    weights_path = args.weights
    if not os.path.exists(weights_path):
        # Check alternative location
        alt_path = os.path.join("models", "trained", "best.pt")
        if os.path.exists(alt_path):
            weights_path = alt_path
        else:
            print(f"[-] Error: Could not locate weight file at path: {args.weights}")
            print("    Please run training first, or specify the correct weight file using --weights.")
            sys.exit(1)

    print(f"[*] Loading model weights from: {weights_path}")
    model = YOLO(weights_path)

    # 2. Run Validation
    yaml_config = os.path.join("dataset", "data.yaml")
    if not os.path.exists(yaml_config):
        print(f"[-] Error: Dataset config data.yaml not found at: {yaml_config}")
        sys.exit(1)
        
    print(f"[*] Executing validation split={args.split}, imgsz={args.imgsz}, device={args.device}...")
    print("-" * 60)

    try:
        # Run validation
        results = model.val(
            data=yaml_config,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            plots=True
        )
        
        # 3. Print a beautiful tabular metric log to console
        print("\n" + "=" * 65)
        print("                 CLASS-WISE EVALUATION SUMMARY")
        print("=" * 65)
        
        # Class mapping names
        class_names = model.names
        
        headers = ["Class ID", "Class Name", "Precision", "Recall", "mAP50", "mAP50-95"]
        table_data = []
        
        # Pull metrics from results object
        # Class-wise metrics
        for i, class_name in class_names.items():
            # index mappings
            p = results.results_dict.get(f"metrics/precision({class_name})", results.class_result(i)[0])
            r = results.results_dict.get(f"metrics/recall({class_name})", results.class_result(i)[1])
            map50 = results.results_dict.get(f"metrics/mAP50({class_name})", results.class_result(i)[2])
            map95 = results.results_dict.get(f"metrics/mAP50-95({class_name})", results.class_result(i)[3])
            
            table_data.append([
                i,
                class_name,
                f"{p:.4f}",
                f"{r:.4f}",
                f"{map50:.4f}",
                f"{map95:.4f}"
            ])
            
        # Add a row for Overall dataset performance
        table_data.append([
            "-",
            "ALL CLASSES",
            f"{results.results_dict['metrics/precision(B)']:.4f}",
            f"{results.results_dict['metrics/recall(B)']:.4f}",
            f"{results.results_dict['metrics/mAP50(B)']:.4f}",
            f"{results.results_dict['metrics/mAP50-95(B)']:.4f}"
        ])
        
        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
        print("\n[+] Validation plots and confusion matrices saved to:")
        print(f"    {results.save_dir}")
        print("=" * 65)
        
    except Exception as e:
        print(f"\n[-] Validation run encountered a crash: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
