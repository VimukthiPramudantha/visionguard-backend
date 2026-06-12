#!/usr/bin/env python
"""
annotation_tools.py
YOLO Annotation Helpers & Converter Pipeline.
Includes an Auto-Labeler (pre-labels new raw images using a pretrained model)
and a Pascal VOC XML-to-YOLO annotation converter.
"""
import os
import sys
import argparse
import xml.etree.ElementTree as ET
from ultralytics import YOLO

def run_auto_labeler(weights, images_dir, output_dir, conf):
    """Detect objects in a directory of unannotated images and write YOLO label txt files."""
    if not os.path.exists(images_dir):
        print(f"[-] Error: Target images directory not found: {images_dir}")
        return False

    # Resolve weight path
    weights_path = weights
    if not os.path.exists(weights_path):
        alt_path = os.path.join("models", "pretrained", "yolo11n.pt")
        if os.path.exists(alt_path):
            weights_path = alt_path
        else:
            weights_path = "yolo11n.pt"  # fallback base download

    print(f"[*] Loading pre-labeling model weights: {weights_path}")
    try:
        model = YOLO(weights_path)
    except Exception as e:
        print(f"[-] Error loading model: {e}")
        return False

    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"[-] No unannotated images found in directory: {images_dir}")
        return False

    print(f"[*] Found {len(image_files)} images to process. Confidence Threshold: {conf}")
    print("[-] Auto-labeling in progress...")
    print("-" * 60)

    count = 0
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        
        # Predict
        try:
            results = model.predict(source=img_path, conf=conf, verbose=False)
            result = results[0]
            
            # Label path
            base_name = os.path.splitext(img_file)[0]
            txt_path = os.path.join(output_dir, f"{base_name}.txt")
            
            w, h = result.orig_shape[1], result.orig_shape[0]
            label_lines = []
            
            # Pull boxes
            for box in result.boxes:
                # Class index
                class_id = int(box.cls[0].item())
                # Box coordinates (normalized xywh format)
                xywh_norm = box.xywhn[0].tolist()
                
                label_lines.append(f"{class_id} {xywh_norm[0]:.6f} {xywh_norm[1]:.6f} {xywh_norm[2]:.6f} {xywh_norm[3]:.6f}\n")
                
            # Write file (even if empty, representing background)
            with open(txt_path, "w") as f:
                f.writelines(label_lines)
                
            count += 1
            if count % 20 == 0:
                print(f"    [+] Processed {count}/{len(image_files)} images...")
                
        except Exception as e:
            print(f"    [-] Failed to auto-label image {img_file}: {e}")

    print(f"\n[+] Auto-labeling complete! Generated {count} label files inside: {output_dir}")
    return True

def convert_voc_to_yolo(xml_dir, output_dir, classes_list):
    """Convert Pascal VOC XML files to YOLO label txt files."""
    if not os.path.exists(xml_dir):
        print(f"[-] Error: Target XML directory not found: {xml_dir}")
        return False

    # Create mapping
    class_map = {name.lower(): i for i, name in enumerate(classes_list)}
    print(f"[*] Map configuration loaded: {class_map}")

    os.makedirs(output_dir, exist_ok=True)
    xml_files = [f for f in os.listdir(xml_dir) if f.lower().endswith('.xml')]

    if not xml_files:
        print(f"[-] No XML files found in directory: {xml_dir}")
        return False

    print(f"[*] Found {len(xml_files)} VOC XML annotation files to convert...")
    
    count = 0
    skipped_count = 0
    
    for xml_file in xml_files:
        xml_path = os.path.join(xml_dir, xml_file)
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Resolve image size
            size_node = root.find("size")
            if size_node is None:
                skipped_count += 1
                continue
                
            w = int(size_node.find("width").text)
            h = int(size_node.find("height").text)
            
            if w == 0 or h == 0:
                skipped_count += 1
                continue

            base_name = os.path.splitext(xml_file)[0]
            txt_path = os.path.join(output_dir, f"{base_name}.txt")
            
            label_lines = []
            for obj in root.findall("object"):
                name = obj.find("name").text.strip().lower()
                if name not in class_map:
                    continue
                    
                class_id = class_map[name]
                
                # Bounding box
                bndbox = obj.find("bndbox")
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)
                
                # Math calculation for YOLO normalized center-x, center-y, width, height
                x_center = (xmin + xmax) / 2.0 / w
                y_center = (ymin + ymax) / 2.0 / h
                box_w = (xmax - xmin) / w
                box_h = (ymax - ymin) / h
                
                label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")
                
            with open(txt_path, "w") as f:
                f.writelines(label_lines)
                
            count += 1
            
        except Exception as e:
            print(f"    [-] Conversion failed for file {xml_file}: {e}")
            skipped_count += 1

    print(f"[+] Conversion complete! Successfully converted {count} files. Skipped: {skipped_count}.")
    return True

def main():
    parser = argparse.ArgumentParser(description="YOLO Workspace Annotation Utility Toolkit")
    subparsers = parser.add_subparsers(dest="command", help="Command utilities")

    # Subparser for Auto-Labeler
    al_parser = subparsers.add_parser("autolabel", help="Auto-generate bounding boxes using a pre-trained model")
    al_parser.add_argument("--weights", type=str, default="models/pretrained/yolo11n.pt", help="Path to pre-trained weights")
    al_parser.add_argument("--src", type=str, required=True, help="Folder containing raw unannotated images")
    al_parser.add_argument("--dest", type=str, required=True, help="Destination folder to save generated label files (.txt)")
    al_parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold to consider a prediction (default: 0.35)")

    # Subparser for VOC-to-YOLO Converter
    voc_parser = subparsers.add_parser("voc2yolo", help="Convert Pascal VOC XML annotations to YOLO TXT format")
    voc_parser.add_argument("--src", type=str, required=True, help="Folder containing VOC XML files")
    voc_parser.add_argument("--dest", type=str, required=True, help="Destination folder to save YOLO label files (.txt)")
    voc_parser.add_argument("--classes", type=str, required=True, 
                            help="Comma-separated list of target class names in the correct class index order")

    args = parser.parse_args()

    print("=" * 60)
    print("          YOLO Workspace Annotation Helper Toolkit")
    print("=" * 60)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "autolabel":
        run_auto_labeler(args.weights, args.src, args.dest, args.conf)
        
    elif args.command == "voc2yolo":
        # Parse classes comma string
        classes_list = [c.strip() for c in args.classes.split(",") if c.strip()]
        if not classes_list:
            print("[-] Error: Please specify a valid comma-separated list of class names via --classes.")
            sys.exit(1)
        convert_voc_to_yolo(args.src, args.dest, classes_list)

    print("=" * 60)

if __name__ == "__main__":
    main()
