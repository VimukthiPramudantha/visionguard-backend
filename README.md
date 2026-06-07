# VisionGuard AI - YOLO v11 Vehicle Detection Project

Welcome to the backend workspace for **VisionGuard AI**. This environment is fully configured for training, evaluating, deploying, and validating a state-of-the-art **YOLO v11** computer vision model optimized for high-speed vehicle detection.

This workspace supports **8 primary classes** of vehicles:
`Bicycle`, `Bus`, `Car`, `Jeepney`, `Motorcycle`, `Tricycle`, `Truck`, `Van`

---

## Key Features

* **GPU Auto-Acceleration:** Automatically leverages your **NVIDIA GeForce GTX 1650** GPU (`CUDA` 11.8) for fast, hardware-accelerated training.
* **Integrity Validation:** Integrated dataset checker to scan coordinates and visually overlay boxes before training.
* **Live Prediction Feed:** Flexible inference pipeline that supports prediction overlays on **webcams, videos, or image folders**.
* **Edge Deployment Ready:** Model conversion tools to export your weights to **ONNX, TensorRT, OpenVINO, and TFLite**.
* **Auto-Labeler Pipeline:** Bootstraps unannotated datasets by auto-detecting and generating labels using pre-trained weights.

---

## Project Directory Structure

```text
visionguard-backend/
│
├── dataset/                    # Vehicle dataset directory (Roboflow structured)
│   ├── train/                  # Training set (images and label txt files)
│   ├── valid/                  # Validation set (images and label txt files)
│   ├── test/                   # Test set (images and label txt files)
│   └── data.yaml               # Dataset metadata & relative path definitions
│
├── models/                     # Saved model weight snapshots
│   ├── pretrained/             # Pretrained base weights (yolo11n.pt)
│   └── trained/                # Permanent backup of your best customized model (best.pt)
│
├── training/                   # Core training & deployment execution scripts
│   ├── train.py                # Parameterized YOLO v11 model training pipeline
│   ├── validate.py             # Advanced precision & recall performance evaluation
│   ├── predict.py              # Inference pipeline for webcams, videos, and images
│   └── export_model.py         # Converts model weights to ONNX/TensorRT formats
│
├── utils/                      # Specialized database & preprocessing utilities
│   ├── dataset_checker.py      # Scan coordinates and visually overlay boxes
│   ├── image_resizer.py        # Scale image dimension preserving bounding boxes
│   ├── video_frame_extractor.py# Slice raw video feeds into standalone frames
│   └── annotation_tools.py     # Pre-trained auto-labeler & XML to YOLO converter
│
├── venv/                       # Standardized local Python virtual environment
├── requirements.txt            # System dependencies tracker
└── README.md                   # Workspace deployment handbook
```

---

## Installation & Environment Setup

This workspace comes with a fully structured virtual environment (`venv`). To activate the environment and check your packages:

### 1. Activate Virtual Environment
```powershell
# In PowerShell (run from visionguard-backend folder)
.\venv\Scripts\Activate.ps1
```

### 2. Verify Environment & GPU Acceleration
Run the automated check utility to verify PyTorch correctly detects your **GeForce GTX 1650 GPU**:
```powershell
.\venv\Scripts\python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```
*Expected Output: `CUDA Available: True | GPU: NVIDIA GeForce GTX 1650`*

### 3. Reinstall Dependencies (If needed)
If you ever need to restore dependencies:
```powershell
.\venv\Scripts\pip install -r requirements.txt
```

---

## Step 1: Verify Dataset Integrity

Before starting model training, run the dataset checker. This script guarantees your annotations are structurally sound, calculates dataset statistics, and opens a visual verification window to ensure the bounding boxes align correctly:

```powershell
# 1. Run console analysis (checks train, valid, test subsets and outputs distributions)
.\venv\Scripts\python utils/dataset_checker.py

# 2. Run visual interactive mode (cv2 window displays bboxes, press SPACE for next, 'q' to close)
.\venv\Scripts\python utils/dataset_checker.py --visualize --subset train
```

---

## Step 2: Train the YOLO v11 Model

Our `train.py` script automatically configures GPU acceleration, monitors learning loss, and saves progress logs into `runs/detect/train`. Once training succeeds, it copies your final weights to `models/trained/best.pt` automatically.

```powershell
# Start training with recommended parameters (50 epochs, batch size 8, size 640)
.\venv\Scripts\python training/train.py --epochs 50 --batch 8 --imgsz 640

# Customize learning rate or override execution device manually to CPU
.\venv\Scripts\python training/train.py --epochs 10 --batch 16 --lr0 0.005 --device cpu

# Resume an interrupted training session from the last checkpoint
.\venv\Scripts\python training/train.py --resume
```

---

## Step 3: Run Model Validation

Evaluate the accuracy and metrics of your trained model on either the `val` or `test` splits. This script compiles a beautiful class-wise grid illustrating Precision, Recall, and mean Average Precision (mAP) values:

```powershell
# Evaluate the best model weights on the validation dataset split
.\venv\Scripts\python training/validate.py --weights runs/detect/train/weights/best.pt --split val

# Evaluate on the test dataset split at an custom image size
.\venv\Scripts\python training/validate.py --weights models/trained/best.pt --split test --imgsz 640
```

---

## Step 4: Run Real-time Predictions (Inference)

Verify how your model performs in real-time. The prediction script handles images, folder paths, video files, or active live webcams:

```powershell
# 1. Run live vehicle detection using your default Webcam stream (index 0)
.\venv\Scripts\python training/predict.py --weights models/trained/best.pt --source 0

# 2. Predict on a single image file or a video
.\venv\Scripts\python training/predict.py --weights models/trained/best.pt --source samples/images/test.jpg
.\venv\Scripts\python training/predict.py --weights models/trained/best.pt --source samples/videos/traffic.mp4

# 3. Predict on a folder of images with custom confidence thresholds
.\venv\Scripts\python training/predict.py --weights models/trained/best.pt --source samples/images/ --conf 0.40 --save
```
*Results are saved inside `runs/detect/predict/`.*

---

## Step 5: Export for Deployment

To compile your model for edge platforms, web applications, or native device apps, run the exporter utility:

```powershell
# Export model to standard ONNX format (Ideal for CPU inference & Web)
.\venv\Scripts\python training/export_model.py --weights models/trained/best.pt --format onnx

# Export to high-speed NVIDIA TensorRT Engine (requires correct CUDA setup on device)
.\venv\Scripts\python training/export_model.py --weights models/trained/best.pt --format engine --half

# Export to TensorFlow Lite (for mobile/edge devices)
.\venv\Scripts\python training/export_model.py --weights models/trained/best.pt --format tflite
```

---

## Additional Preprocessing Utilities

### 1. Slice Videos into Training Images
To extract frame captures from a traffic recording to expand your image database:
```powershell
.\venv\Scripts\python utils/video_frame_extractor.py --source samples/videos/traffic.mp4 --output samples/images --interval 30 --prefix raw_slice
```
*Saves 1 image for every 30 frames (approximately 1 frame per second for a 30fps video).*

### 2. Annotation Auto-Labeler
Auto-annotate a directory of raw images with bounding boxes using a pretrained model to avoid tedious hand-labeling hours:
```powershell
.\venv\Scripts\python utils/annotation_tools.py autolabel --src samples/images --dest samples/labels --conf 0.30
```

### 3. XML to YOLO Annotation Converter
Convert standard Pascal VOC XML files to the required normalized YOLO text format:
```powershell
.\venv\Scripts\python utils/annotation_tools.py voc2yolo --src samples/xml --dest samples/labels --classes "Bicycle,Bus,Car,Jeepney,Motorcycle,Tricycle,Truck,Van"
```

---

## Development Guidelines

* **Keep coordinates normalized:** YOLO coordinates must always be in the normalized bounding box format (`class_id x_center y_center width height`), where values are floats bounded between `0.0` and `1.0`.
* **Hardware optimization:** Do not manually set `--device cpu` in training unless your GPU is experiencing memory limitations. Using the GPU accelerates training speed by **10x to 50x**.
* **Model weight backup:** Always verify that `models/trained/best.pt` has been backed up successfully after running your training runs.
