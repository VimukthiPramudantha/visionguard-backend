from typing import List, Dict
from datetime import datetime
import json
from app.schemas.camera import Camera
from app.core.db_service import get_all_cameras as db_get_all_cameras, sync_hardcoded_cameras, HARDCODED_CAMERAS

_in_memory_cameras = {}
active_feeds = set()
_camera_zones: Dict[str, list] = {}
_zone_snapshot_cooldowns: Dict[str, float] = {}
ZONE_SNAPSHOT_COOLDOWN_SECS = 600  

_hardcoded_synced = False

def _ensure_hardcoded_synced():
    """Sync hardcoded cameras to DB on first call."""
    global _hardcoded_synced
    if not _hardcoded_synced:
        try:
            sync_hardcoded_cameras()
            _hardcoded_synced = True
        except Exception as e:
            print(f"[VisionGuard] Failed to sync hardcoded cameras: {e}")

def get_detected_cameras() -> List[Camera]:
    _ensure_hardcoded_synced()

    detected = []

    try:
        db_cameras = db_get_all_cameras()
        for row in db_cameras:
            zone_points = row.get("zone_points", [])
            if isinstance(zone_points, str):
                try:
                    zone_points = json.loads(zone_points)
                except (json.JSONDecodeError, TypeError):
                    zone_points = []

            cam = Camera(
                id=row["id"],
                name=row["name"],
                type=row["type"],
                url=row.get("url"),
                status=row.get("status", "online"),
                last_active=row.get("updated_at", datetime.utcnow().isoformat()),
                location=row.get("location"),
                user_id=row.get("user_id"),
                zone_points=zone_points if zone_points else None,
                created_at=row.get("created_at"),
            )
            detected.append(cam)

            if zone_points and cam.id not in _camera_zones:
                _camera_zones[cam.id] = zone_points

    except Exception as e:
        print(f"[VisionGuard] Failed to fetch cameras from DB: {e}")
        for hc in HARDCODED_CAMERAS:
            detected.append(Camera(
                id=hc["id"],
                name=hc["name"],
                type=hc["type"],
                url=hc.get("url"),
                status=hc.get("status", "online"),
                last_active=datetime.utcnow().isoformat(),
                location=hc.get("location"),
            ))

    db_ids = {c.id for c in detected}
    for cam_id, cam in _in_memory_cameras.items():
        if cam_id not in db_ids:
            detected.append(cam)

    return detected

