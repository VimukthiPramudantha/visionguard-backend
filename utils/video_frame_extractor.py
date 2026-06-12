#!/usr/bin/env python
"""
video_frame_extractor.py
Video Dataset Generation Tool.
Slices video streams (.mp4, .avi, etc.) into high-quality JPEG training images at a customizable frame interval.
"""
import os
import sys
import argparse
import cv2

def extract_frames(video_path, output_dir, frame_interval, prefix):
    """Extract frames from a single video file."""
    if not os.path.exists(video_path):
        print(f"[-] Error: Video file not found at: {video_path}")
        return False

    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[-] Error: Could not open video file: {video_path}")
        return False

    os.makedirs(output_dir, exist_ok=True)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    
    print(f"[*] Extracting: '{video_filename}'")
    print(f"    FPS: {fps:.2f} | Total Frames: {total_frames}")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame if interval condition met
        if frame_count % frame_interval == 0:
            out_filename = f"{prefix}_{video_filename}_frame_{frame_count:06d}.jpg"
            out_path = os.path.join(output_dir, out_filename)
            
            # Save frame
            cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            saved_count += 1

        frame_count += 1
        
        # Show progress
        if frame_count % 100 == 0:
            sys.stdout.write(f"\r    Progress: {frame_count}/{total_frames} frames processed...")
            sys.stdout.flush()

    cap.release()
    print(f"\n    [+] Successfully saved {saved_count} frames to: {output_dir}")
    return True

def main():
    parser = argparse.ArgumentParser(description="YOLO Video Frame Extractor Utility")
    parser.add_argument("--source", type=str, required=True, help="Path to video file or directory containing videos")
    parser.add_argument("--output", type=str, default="samples/images", help="Output directory to save frames (default: samples/images)")
    parser.add_argument("--interval", type=int, default=30, 
                        help="Frame slice interval (e.g. 30 saves 1 frame every 30 frames / approx. 1 second of video)")
    parser.add_argument("--prefix", type=str, default="extract", help="Filename prefix for saved images (default: 'extract')")
    args = parser.parse_args()

    print("=" * 60)
    print("             YOLO Video Frame Slicing Utility")
    print("=" * 60)
    print(f"[*] Target source: {args.source}")
    print(f"[*] Target output dir: {args.output}")
    print(f"[*] Slice interval: Every {args.interval} frames")
    print("-" * 60)

    # Resolve paths
    source_path = args.source
    if not os.path.exists(source_path):
        print(f"[-] Error: Source target does not exist: {source_path}")
        sys.exit(1)

    # Process files
    if os.path.isdir(source_path):
        video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.wmv')
        video_files = [f for f in os.listdir(source_path) if f.lower().endswith(video_extensions)]
        
        if not video_files:
            print(f"[-] No valid video files found in directory: {source_path}")
            sys.exit(1)

        print(f"[+] Found {len(video_files)} video files in folder.")
        for v_file in video_files:
            v_path = os.path.join(source_path, v_file)
            extract_frames(v_path, args.output, args.interval, args.prefix)
    else:
        extract_frames(source_path, args.output, args.interval, args.prefix)

    print("\n[+] Slicing pipeline completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
