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
    user_id: str,
) -> dict | None:
    try:
        if not user_id:
            raise ValueError("user_id is required to save detection events")

        event = {
            "camera_id": camera_id,
            "detection_type": detection_type,
            "confidence": round(confidence, 4),
            "snapshot_url": snapshot_url,
            "user_id": user_id,
            "detected_at": datetime.now(timezone.utc).isoformat(),
        }

        response = supabase.table("detection_events").insert(event).execute()
        if response.data:
            saved_event = response.data[0]

            alert_user_id = user_id
            if not alert_user_id:
                cam = get_camera_by_id(camera_id)
                if cam:
                    alert_user_id = cam.get("user_id")

            if alert_user_id:
                create_alert(
                    user_id=alert_user_id,
                    camera_id=camera_id,
                    detection_event_id=saved_event.get("id"),
                    snapshot_url=snapshot_url,
                    detection_type=detection_type,
                    confidence=round(confidence, 4),
                )

            return saved_event
        return None
    except Exception as e:
        print(f"[VisionGuard] Failed to save detection event: {e}")
        return None


# ── Alert helpers ────────────────────────────────────────────────

def create_alert(
    user_id: str,
    camera_id: str,
    detection_event_id: str | None = None,
    snapshot_url: str | None = None,
    detection_type: str | None = None,
    confidence: float | None = None,
) -> dict | None:
    try:
        alert = {
            "user_id": user_id,
            "camera_id": camera_id,
            "status": "unread",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if detection_event_id:
            alert["detection_event_id"] = detection_event_id
        if snapshot_url:
            alert["snapshot_url"] = snapshot_url
        if detection_type:
            alert["detection_type"] = detection_type
        if confidence is not None:
            alert["confidence"] = confidence

        response = supabase.table("alerts").insert(alert).execute()
        if response.data:
            print(f"[VisionGuard] Alert created for user {user_id} (camera {camera_id})")
            return response.data[0]
        return None
    except Exception as e:
        print(f"[VisionGuard] Failed to create alert: {e}")
        return None


def get_alerts_for_user(user_id: str) -> list:
    try:
        response = (
            supabase.table("alerts")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )
        return response.data or []
    except Exception as e:
        print(f"[VisionGuard] Failed to fetch alerts: {e}")
        return []


def get_unread_alert_count(user_id: str) -> int:
    try:
        response = (
            supabase.table("alerts")
            .select("id", count="exact")
            .eq("user_id", user_id)
            .eq("status", "unread")
            .execute()
        )
        return response.count or 0
    except Exception as e:
        print(f"[VisionGuard] Failed to count unread alerts: {e}")
        return 0


def mark_alert_read(alert_id: str) -> dict | None:
    """Update an alert's status to 'read'."""
    try:
        response = (
            supabase.table("alerts")
            .update({"status": "read"})
            .eq("id", alert_id)
            .execute()
        )
        if response.data:
            return response.data[0]
        return None
    except Exception as e:
        print(f"[VisionGuard] Failed to mark alert as read: {e}")
        return None
