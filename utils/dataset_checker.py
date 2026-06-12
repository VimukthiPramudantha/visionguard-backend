#!/usr/bin/env python
"""
dataset_checker.py
Comprehensive YOLO Dataset Checker.
Verifies annotation integrity, reports class counts, and lets you visually inspect annotations.
"""
import os
import sys
import argparse
import random
import yaml
import cv2

def load_dataset_config(yaml_path):
    """Load class names from data.yaml."""
    if not os.path.exists(yaml_path):
        print(f"[-] Warning: Dataset config not found at: {yaml_path}")
        return None
    try:
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[-] Error loading YAML: {e}")
        return None

def verify_subset(dataset_dir, subset_name, class_names):
    """Run checks on a dataset subset (e.g., train, valid, test)."""
    images_dir = os.path.join(dataset_dir, subset_name, "images")
    labels_dir = os.path.join(dataset_dir, subset_name, "labels")

    print(f"\n[*] Checking subset: {subset_name.upper()}")
    print(f"    Images directory: {images_dir}")
    print(f"    Labels directory: {labels_dir}")

    if not os.path.exists(images_dir):
        print(f"    [-] Skipped: Images directory does not exist.")
        return None

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    label_files = [f for f in os.listdir(labels_dir) if f.lower().endswith('.txt')] if os.path.exists(labels_dir) else []

    total_images = len(image_files)
    total_labels = len(label_files)
    
    print(f"    [+] Found {total_images} images and {total_labels} label files.")

    # Stat counters
    missing_labels = 0
    corrupt_labels = 0
    empty_labels = 0
    total_boxes = 0
    class_counts = {i: 0 for i in range(len(class_names))} if class_names else {}

    # Run check per image
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        label_file = base_name + ".txt"
        label_path = os.path.join(labels_dir, label_file)

        if not os.path.exists(label_path):
            missing_labels += 1
            continue

        if os.path.getsize(label_path) == 0:
            empty_labels += 1
            continue

        # Parse labels
        try:
            with open(label_path, "r") as f:
                lines = f.readlines()
                
            for line_idx, line in enumerate(lines):
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) != 5:
                    corrupt_labels += 1
                    continue
                
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                
                # Check coordinate normalization (0.0 to 1.0)
                if any(x < 0.0 or x > 1.0 for x in coords):
                    print(f"    [!] Warning: Coordinate out of bounds in {label_file}: '{line.strip()}'")
                    corrupt_labels += 1
                    continue

                total_boxes += 1
                if class_names and class_id in class_counts:
                    class_counts[class_id] += 1
                elif class_names:
                    # Class index outside defined range in yaml
                    if class_id not in class_counts:
                        class_counts[class_id] = 1
                    else:
                        class_counts[class_id] += 1

        except Exception:
            corrupt_labels += 1

    # Report results
    print(f"    [+] Analysis Results:")
    print(f"        - Missing Label files (background images): {missing_labels}")
    print(f"        - Empty Label files: {empty_labels}")
    print(f"        - Corrupt annotations: {corrupt_labels}")
    print(f"        - Total annotated bounding boxes: {total_boxes}")
    
    if class_names and total_boxes > 0:
        print("        - Class Distribution:")
        for cid, name in class_names.items():
            count = class_counts.get(cid, 0)
            pct = (count / total_boxes) * 100 if total_boxes > 0 else 0
            print(f"          * {name:<12} (ID {cid}): {count:>5} ({pct:.1f}%)")
            
    return {
        "images": total_images,
        "boxes": total_boxes,
        "class_counts": class_counts
    }

def visualize_annotations(dataset_dir, subset_name, yaml_config):
    """Draw bounding boxes on images and show them using OpenCV."""
    images_dir = os.path.join(dataset_dir, subset_name, "images")
    labels_dir = os.path.join(dataset_dir, subset_name, "labels")

    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print("[-] Error: Images or labels directory not found for visualization.")
        return

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print("[-] No images found to visualize.")
        return

    class_names = yaml_config.get("names", {}) if yaml_config else {}
    if isinstance(class_names, list):
        class_names = {i: name for i, name in enumerate(class_names)}

    # Color palette for classes (8 classes)
    colors = [
        (255, 0, 0),    # Bicycle - Blue
        (0, 255, 0),    # Bus - Green
        (0, 0, 255),    # Car - Red
        (255, 255, 0),  # Jeepney - Cyan
        (255, 0, 255),  # Motorcycle - Magenta
        (0, 255, 255),  # Tricycle - Yellow
        (128, 0, 255),  # Truck - Orange
        (0, 128, 255)   # Van - Orange-Yellow
    ]

    print("\n" + "=" * 60)
    print("      DATASET VISUALIZER - PRESS SPACE FOR NEXT, 'Q' TO QUIT")
    print("=" * 60)
    
    # Shuffle to see random samples
    random.shuffle(image_files)

    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        label_file = base_name + ".txt"
        
        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, label_file)

        if not os.path.exists(label_path):
            continue

        image = cv2.imread(img_path)
        if image is None:
            continue

        h, w, _ = image.shape

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            x_center, y_center, bbox_w, bbox_h = [float(x) for x in parts[1:]]

            # Denormalize coordinates to pixels
            xmin = int((x_center - bbox_w / 2) * w)
            ymin = int((y_center - bbox_h / 2) * h)
            xmax = int((x_center + bbox_w / 2) * w)
            ymax = int((y_center + bbox_h / 2) * h)

            # Pick class properties
            color = colors[class_id % len(colors)]
            class_name = class_names.get(class_id, f"Class {class_id}")
            
            # Draw bbox
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            
            # Draw label background and text
            label_text = f"{class_name}"
            tf = max(1, int(2 * 0.5))  # Font thickness
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, tf)
            cv2.rectangle(image, (xmin, ymin - th - 5), (xmin + tw, ymin), color, -1)
            cv2.putText(image, label_text, (xmin, ymin - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), tf)

        # Scale down if too large
        display_img = image
        max_display_w = 1000
        if w > max_display_w:
            scale = max_display_w / w
            display_img = cv2.resize(image, (max_display_w, int(h * scale)))

        cv2.imshow("Dataset Verification Tool", display_img)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    cv2.destroyAllWindows()
    print("[*] Visualizer closed.")

def main():
    parser = argparse.ArgumentParser(description="YOLO v11 Dataset Checker & Visualizer")
    parser.add_argument("--dataset", type=str, default="dataset/data.yaml", help="Path to data.yaml config")
    parser.add_argument("--visualize", action="store_true", help="Launch OpenCV interactive visualization")
    parser.add_argument("--subset", type=str, default="train", choices=["train", "valid", "test"], 
                        help="Subset to visualize if --visualize is enabled")
    args = parser.parse_args()

    print("=" * 60)
    print("           YOLO v11 Dataset Integrity Checker")
    print("=" * 60)

    # 1. Load config
    yaml_config = load_dataset_config(args.dataset)
    class_names = {}
    if yaml_config and "names" in yaml_config:
        names_node = yaml_config["names"]
        if isinstance(names_node, list):
            class_names = {i: name for i, name in enumerate(names_node)}
        elif isinstance(names_node, dict):
            class_names = {int(k): v for k, v in names_node.items()}
        print(f"[+] Loaded dataset classes: {list(class_names.values())}")
    
    # Resolve physical dataset root
    dataset_dir = "dataset"
    if yaml_config and "path" in yaml_config:
        # If absolute path is set
        path_val = yaml_config["path"]
        if os.path.exists(path_val):
            dataset_dir = path_val

    # 2. Run analysis
    subsets = ["train", "valid", "test"]
    overall_stats = {"images": 0, "boxes": 0}
    
    for subset in subsets:
        res = verify_subset(dataset_dir, subset, class_names)
        if res:
            overall_stats["images"] += res["images"]
            overall_stats["boxes"] += res["boxes"]

    print("\n" + "=" * 60)
    print("                      OVERALL STATS")
    print("=" * 60)
    print(f"[+] Total Dataset Images: {overall_stats['images']}")
    print(f"[+] Total Annotations (BBoxes): {overall_stats['boxes']}")
    print("=" * 60)

    # 3. Handle visualization
    if args.visualize:
        visualize_annotations(dataset_dir, args.subset, yaml_config)

if __name__ == "__main__":
    main()
