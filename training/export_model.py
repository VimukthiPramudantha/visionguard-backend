#!/usr/bin/env python
"""
export_model.py
YOLO v11 Model Conversion & Deployment Export Pipeline.
Exports PyTorch weights (.pt) to ONNX, TensorRT, OpenVINO, TFLite, etc.
"""
import os
import sys
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="YOLO v11 Model Export Pipeline")
    parser.add_argument("--weights", type=str, default="runs/detect/train/weights/best.pt", 
                        help="Path to trained model weights (.pt)")
    parser.add_argument("--format", type=str, default="onnx", 
                        choices=["onnx", "engine", "openvino", "tflite", "pb", "torchscript"], 
                        help="Export format target (default: 'onnx')")
    parser.add_argument("--imgsz", type=int, default=640, help="Export input image resolution width/height")
    parser.add_argument("--half", action="store_true", help="FP16 half-precision quantization (recommended for GPU)")
    parser.add_argument("--int8", action="store_true", help="INT8 quantization (recommended for CPU/Edge)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for compile/export ('cpu' or '0')")
    args = parser.parse_args()

    print("=" * 60)
    print("           YOLO v11 Model Export & Conversion Utility")
    print("=" * 60)

    # 1. Resolve weight file path
    weights_path = args.weights
    if not os.path.exists(weights_path):
        # Fallback path check
        alt_path = os.path.join("models", "trained", "best.pt")
        if os.path.exists(alt_path):
            weights_path = alt_path
        else:
            print(f"[-] Error: Weight file not found at: {args.weights}")
            print("    Please run training first, or specify the correct weight file using --weights.")
            sys.exit(1)

    print(f"[*] Loading PyTorch weights: {weights_path}")
    try:
        model = YOLO(weights_path)
    except Exception as e:
        print(f"[-] Error loading model: {e}")
        sys.exit(1)

    # Format descriptions for informative printing
    format_info = {
        "onnx": "Open Neural Network Exchange (ONNX) - Ideal for CPU/Web deployment.",
        "engine": "NVIDIA TensorRT Engine (.engine) - Ultra high-speed GPU acceleration.",
        "openvino": "Intel OpenVINO - Highly optimized for Intel CPUs/iGPUs.",
        "tflite": "TensorFlow Lite (.tflite) - Ideal for mobile and edge devices.",
        "pb": "TensorFlow Frozen Graph (.pb) - Standard tensorflow model format.",
        "torchscript": "TorchScript - High-performance deployment inside C++ environments."
    }

    selected_format_desc = format_info.get(args.format, args.format)
    print(f"[*] Export Target Format: {args.format.upper()}")
    print(f"[*] Description: {selected_format_desc}")
    print(f"[*] Resolution: {args.imgsz}x{args.imgsz}")
    print(f"[*] Quantization options: Half-precision={args.half}, INT8-quantization={args.int8}")
    print("[-] Executing model compilation. Please wait...")
    print("-" * 60)

    try:
        # Run export pipeline
        exported_path = model.export(
            format=args.format,
            imgsz=args.imgsz,
            half=args.half,
            int8=args.int8,
            device=args.device
        )
        
        print("\n" + "-" * 60)
        print("[+] Model export completed successfully!")
        print(f"[+] Output compiled weight saved to: {os.path.abspath(exported_path)}")
        print("=" * 60)

    except Exception as e:
        print(f"\n[-] Model export failed with error: {e}")
        print("[-] Note: Certain exports like TensorRT (engine) require correct CUDA/CUDNN GPU setups.")
        sys.exit(1)

if __name__ == "__main__":
    main()
