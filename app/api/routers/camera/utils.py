import cv2
import numpy as np
import time as _time
from datetime import datetime
from app.api.routers.camera.state import _zone_snapshot_cooldowns, ZONE_SNAPSHOT_COOLDOWN_SECS
from app.core.db_service import upload_snapshot, save_detection_event

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
    if intruders:
        print(f"[VisionGuard] Zone check: {len(intruders)}/{len(detections)} "
              f"object(s) inside restricted area")
    return intruders

def save_intrusion_snapshot(frame, camera_id, intruders=None, user_id=None):

    now = _time.time()
    last = _zone_snapshot_cooldowns.get(camera_id, 0)
    if now - last < ZONE_SNAPSHOT_COOLDOWN_SECS:
        return None  

    print(f"[VisionGuard] Intrusion detected on camera {camera_id} — "
          f"{len(intruders or [])} intruder(s), uploading snapshot...")

    snapshot_url = upload_snapshot(frame, camera_id)

    if not snapshot_url:
        print(f"[VisionGuard] Snapshot upload FAILED for camera {camera_id} — "
              "cooldown NOT set so next frame will retry")
        return None
    _zone_snapshot_cooldowns[camera_id] = now

    if not user_id:
        from app.core.db_service import get_camera_by_id
        cam = get_camera_by_id(camera_id)
        if cam:
            user_id = cam.get("user_id")

    if not user_id:
        try:
            from app.core.supabase import supabase
            users_res = supabase.table("users").select("id").limit(1).execute()
            if users_res.data:
                user_id = users_res.data[0]["id"]
        except Exception as err:
            print(f"[VisionGuard] Failed to fetch fallback user from DB: {err}")

    if not user_id:
        user_id = "00000000-0000-0000-0000-000000000000"
        print(f"[VisionGuard] Warning: Using default fallback system user_id for camera {camera_id}")

    if intruders and snapshot_url:
        for intruder in intruders:
            save_detection_event(
                camera_id=camera_id,
                detection_type=intruder.get("label", "unknown"),
                confidence=intruder.get("confidence", 0.0),
                snapshot_url=snapshot_url,
                user_id=user_id,
            )
        print(f"[VisionGuard] ✓ Alert(s) created for {len(intruders)} intruder(s) on camera {camera_id}")
    elif snapshot_url:
        save_detection_event(
            camera_id=camera_id,
            detection_type="unknown",
            confidence=0.0,
            snapshot_url=snapshot_url,
            user_id=user_id,
        )
        print(f"[VisionGuard] ✓ Alert created (unknown type) for camera {camera_id}")

    return snapshot_url

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
