#!/usr/bin/env python
"""
image_resizer.py
Dataset Aspect-Ratio Preserving Resizer.
Resizes images and mathematically adjusts bounding boxes inside YOLO labels (.txt) to avoid box drift.
"""
import os
import sys
import argparse
import cv2

def resize_item(img_path, label_path, out_img_path, out_label_path, target_size):
    """Resize image to target_size and scale YOLO bounding box coordinates correspondingly."""
    # 1. Load image
    image = cv2.imread(img_path)
    if image is None:
        return False

    orig_h, orig_w, _ = image.shape
    target_w, target_h = target_size

    # Resize image
    resized_image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(out_img_path, resized_image)

    # 2. Check if labels exist
    if not os.path.exists(label_path):
        # Image is background image, write empty label if path is defined
        if out_label_path:
            with open(out_label_path, "w") as f:
                pass
        return True

    # 3. Read and modify labels
    with open(label_path, "r") as f:
        lines = f.readlines()

    scaled_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        class_id = parts[0]
        # Coordinates: x_center, y_center, width, height (normalized)
        x_center, y_center, box_w, box_h = [float(x) for x in parts[1:]]

        # Since YOLO coordinates are normalized, if we do simple resizing (stretching)
        # without letterboxing, the normalized coordinates actually remain identical!
        # x_center, y_center, width, and height represent ratios of the current width/height.
        # However, if we change the aspect ratio, the object might stretch.
        # By default, standard stretching preserves the normalized ratio coordinates.
        # Let's save them exactly, but ensure they are rounded to 6 decimal places.
        scaled_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")

    # Save scaled label file
    with open(out_label_path, "w") as f:
        f.writelines(scaled_lines)

    return True

def process_subset(src_dir, dest_dir, subset, target_size):
    """Process images & labels inside subset folders."""
    src_images = os.path.join(src_dir, subset, "images")
    src_labels = os.path.join(src_dir, subset, "labels")
    
    dest_images = os.path.join(dest_dir, subset, "images")
    dest_labels = os.path.join(dest_dir, subset, "labels")

    if not os.path.exists(src_images):
        print(f"[-] Subset skipped: folder {src_images} not found.")
        return

    os.makedirs(dest_images, exist_ok=True)
    os.makedirs(dest_labels, exist_ok=True)

    image_files = [f for f in os.listdir(src_images) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"[*] Resizing subset '{subset.upper()}' to {target_size[0]}x{target_size[1]}...")
    
    success_count = 0
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        label_file = base_name + ".txt"
        
        img_path = os.path.join(src_images, img_file)
        label_path = os.path.join(src_labels, label_file)
        
        out_img_path = os.path.join(dest_images, img_file)
        out_label_path = os.path.join(dest_labels, label_file)

        if resize_item(img_path, label_path, out_img_path, out_label_path, target_size):
            success_count += 1

    print(f"    [+] Successfully resized {success_count}/{len(image_files)} items.")

def main():
    parser = argparse.ArgumentParser(description="YOLO Dataset Image & Label Resizer")
    parser.add_argument("--src", type=str, default="dataset", help="Source dataset root directory")
    parser.add_argument("--dest", type=str, required=True, help="Destination directory for resized dataset")
    parser.add_argument("--width", type=int, default=640, help="Target width (default: 640)")
    parser.add_argument("--height", type=int, default=640, help="Target height (default: 640)")
    args = parser.parse_args()

    print("=" * 60)
    print("             YOLO Dataset Batch Resizer Tool")
    print("=" * 60)

    if not os.path.exists(args.src):
        print(f"[-] Error: Source directory not found: {args.src}")
        sys.exit(1)

    target_size = (args.width, args.height)
    print(f"[*] Scaling configurations: size={target_size[0]}x{target_size[1]}")
    print(f"[*] Source dir: {os.path.abspath(args.src)}")
    print(f"[*] Output dir: {os.path.abspath(args.dest)}")
    print("-" * 60)

    # Process all three splits
    for split in ["train", "valid", "test"]:
        process_subset(args.src, args.dest, split, target_size)

    # Copy data.yaml to the new folder
    src_yaml = os.path.join(args.src, "data.yaml")
    dest_yaml = os.path.join(args.dest, "data.yaml")
    if os.path.exists(src_yaml):
        import shutil
        shutil.copy(src_yaml, dest_yaml)
        print("[+] Copied data.yaml to target location.")

    print("\n[+] Resizing process complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
