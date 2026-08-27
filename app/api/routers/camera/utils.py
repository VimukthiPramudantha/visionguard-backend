import os
import cv2
import numpy as np
import time as _time
from datetime import datetime
from app.api.routers.camera.state import _zone_snapshot_cooldowns, ZONE_SNAPSHOT_COOLDOWN_SECS

def _denormalize_zone(zone_points, frame_w, frame_h):
    return np.array(
        [[int(p["x"] * frame_w), int(p["y"] * frame_h)] for p in zone_points],
        dtype=np.int32,
    )

def check_zone_intrusion(detections, zone_points, frame_w, frame_h):
    if not zone_points or len(zone_points) < 3:
        return []

    poly = _denormalize_zone(zone_points, frame_w, frame_h).reshape((-1, 1, 2))
    intruders = []
    for det in detections:
        cx = (det["x1"] + det["x2"]) // 2
        cy = (det["y1"] + det["y2"]) // 2
        dist = cv2.pointPolygonTest(poly, (float(cx), float(cy)), False)
        if dist >= 0:  
            intruders.append(det)
    return intruders

def save_intrusion_snapshot(frame, camera_id):
    now = _time.time()
    last = _zone_snapshot_cooldowns.get(camera_id, 0)
    if now - last < ZONE_SNAPSHOT_COOLDOWN_SECS:
        return None  

    _zone_snapshot_cooldowns[camera_id] = now

    dt = datetime.now()
    time_folder = dt.strftime("%H-%M-%S")
    date_file = dt.strftime("%Y-%m-%d")

    backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    save_dir = os.path.join(backend_root, "snapshots", "Detect", camera_id, time_folder)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{date_file}.jpg")
    cv2.imwrite(save_path, frame)
    print(f"[VisionGuard] Intrusion snapshot saved → {save_path}")
    return save_path

def draw_zone_overlay(frame, zone_points):
    if not zone_points or len(zone_points) < 3:
        return frame
    h, w = frame.shape[:2]
    poly = _denormalize_zone(zone_points, w, h)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [poly], (0, 200, 255, 80))
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    cv2.polylines(frame, [poly], True, (0, 200, 255), 2, cv2.LINE_AA)
    return frame
