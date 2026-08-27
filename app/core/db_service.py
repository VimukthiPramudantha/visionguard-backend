# app/core/db_service.py
"""
Supabase DB + Storage service layer for cameras and detection events.
"""
import cv2
import json
from datetime import datetime, timezone
from app.core.supabase import supabase

STORAGE_BUCKET = "detection-snapshots"


def upsert_camera(camera_data: dict) -> dict:
    """Insert or update a camera record in the cameras table."""
    camera_data["updated_at"] = datetime.now(timezone.utc).isoformat()
    if "created_at" not in camera_data:
        camera_data["created_at"] = datetime.now(timezone.utc).isoformat()

    response = supabase.table("cameras").upsert(
        camera_data, on_conflict="id"
    ).execute()

    if response.data:
        return response.data[0]
    return camera_data


def get_all_cameras() -> list:
    """Fetch all cameras from the cameras table."""
    response = supabase.table("cameras").select("*").execute()
    return response.data or []


def get_camera_by_id(camera_id: str) -> dict | None:
    """Fetch a single camera by ID."""
    response = (
        supabase.table("cameras")
        .select("*")
        .eq("id", camera_id)
        .execute()
    )
    if response.data:
        return response.data[0]
    return None


def save_zone_points(camera_id: str, points: list) -> None:
    """Update zone_points for a camera in the database."""
    supabase.table("cameras").update({
        "zone_points": json.dumps(points),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", camera_id).execute()


def delete_zone_points(camera_id: str) -> None:
    """Clear zone_points for a camera in the database."""
    supabase.table("cameras").update({
        "zone_points": json.dumps([]),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", camera_id).execute()



HARDCODED_CAMERAS = [
    {
        "id": "cctv_1",
        "name": "Camera 01",
        "type": "cctv",
        "url": "CCTV/Cam01",
        "status": "online",
        "location": "Front Gate",
    },
    {
        "id": "cctv_8",
        "name": "Test Cam 01",
        "type": "cctv",
        "url": "CCTV/TestCam01",
        "status": "online",
        "location": "Mark For Vehicle",
    },
    {
        "id": "cctv_9",
        "name": "Test Cam 02",
        "type": "cctv",
        "url": "CCTV/TestCam02",
        "status": "online",
        "location": "Mark For Human",
    },
]


def sync_hardcoded_cameras() -> None:
    """Ensure hardcoded CCTV cameras exist in the database.
    Uses upsert so existing records are updated without losing zone_points."""
    for cam in HARDCODED_CAMERAS:
        existing = get_camera_by_id(cam["id"])
        if existing:
            supabase.table("cameras").update({
                "name": cam["name"],
                "type": cam["type"],
                "url": cam["url"],
                "status": cam["status"],
                "location": cam["location"],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", cam["id"]).execute()
        else:
            cam_data = {
                **cam,
                "zone_points": json.dumps([]),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            supabase.table("cameras").insert(cam_data).execute()

    print("[VisionGuard] Hardcoded cameras synced to database")

def upload_snapshot(frame, camera_id: str) -> str | None:
    """Encode frame to JPEG and upload to Supabase Storage.
    Returns the public URL of the uploaded snapshot, or None on failure."""
    try:
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("[VisionGuard] Failed to encode snapshot frame")
            return None

        image_bytes = buffer.tobytes()
        dt = datetime.now(timezone.utc)
        file_path = f"{camera_id}/{dt.strftime('%Y-%m-%d')}/{dt.strftime('%H-%M-%S')}_{dt.strftime('%f')}.jpg"

        supabase.storage.from_(STORAGE_BUCKET).upload(
            path=file_path,
            file=image_bytes,
            file_options={"content-type": "image/jpeg"},
        )

        public_url = supabase.storage.from_(STORAGE_BUCKET).get_public_url(file_path)
        print(f"[VisionGuard] Snapshot uploaded → {public_url}")
        return public_url

    except Exception as e:
        print(f"[VisionGuard] Snapshot upload failed: {e}")
        return None

def save_detection_event(
    camera_id: str,
    detection_type: str,
    confidence: float,
    snapshot_url: str | None,
    user_id: str | None = None,
) -> dict | None:
    """Insert a detection event into the detection_events table."""
    try:
        event = {
            "camera_id": camera_id,
            "detection_type": detection_type,
            "confidence": round(confidence, 4),
            "snapshot_url": snapshot_url,
            "detected_at": datetime.now(timezone.utc).isoformat(),
        }
        if user_id:
            event["user_id"] = user_id

        response = supabase.table("detection_events").insert(event).execute()
        if response.data:
            return response.data[0]
        return None
    except Exception as e:
        print(f"[VisionGuard] Failed to save detection event: {e}")
        return None
