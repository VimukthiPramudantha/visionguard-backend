# app/api/routers/camera_routes.py
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from app.core.supabase import supabase

router = APIRouter()

def detect_and_sync_usb_cameras():
    try:
        import cv2
    except ImportError:
        print("OpenCV (cv2) is not installed. Skipping local camera check.")
        return

    try:
        # Detect connected USB cameras
        connected_indices = []
        for index in range(3):  # Check indices 0, 1, 2
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if hasattr(cv2, 'CAP_DSHOW') else cv2.VideoCapture(index)
            if cap.isOpened():
                connected_indices.append(index)
                cap.release()
        
        # Get existing cameras from database
        db_response = supabase.table("cameras").select("*").execute()
        existing_cameras = db_response.data or []
        
        # Map existing cameras by their URL (e.g. "0", "1") for USB type
        existing_usb_map = {cam["url"]: cam for cam in existing_cameras if cam.get("type") == "usb"}
        
        detected_urls = [str(idx) for idx in connected_indices]
        
        # 1. Update/Insert detected cameras
        for idx in connected_indices:
            url_str = str(idx)
            if url_str in existing_usb_map:
                cam = existing_usb_map[url_str]
                if cam.get("status") != "online":
                    supabase.table("cameras").update({
                        "status": "online",
                        "last_active": datetime.utcnow().isoformat()
                    }).eq("id", cam["id"]).execute()
            else:
                new_cam = {
                    "name": f"Integrated Camera {idx}" if idx == 0 else f"USB Camera {idx}",
                    "type": "usb",
                    "url": url_str,
                    "location": "Local Host",
                    "status": "online",
                    "last_active": datetime.utcnow().isoformat()
                }
                supabase.table("cameras").insert(new_cam).execute()
        
        # 2. Update offline cameras (type 'usb' but not detected)
        for url_str, cam in existing_usb_map.items():
            if url_str not in detected_urls:
                if cam.get("status") != "offline":
                    supabase.table("cameras").update({
                        "status": "offline"
                    }).eq("id", cam["id"]).execute()
                    
    except Exception as e:
        print(f"Error detecting/syncing USB cameras: {e}")

class Camera(BaseModel):
    id: str
    name: str
    type: str          # "rtsp", "usb", "ip"
    url: Optional[str] = None
    status: str        # "online", "offline", "error"
    last_active: Optional[str] = None
    location: Optional[str] = None

class CameraCreate(BaseModel):
    name: str
    type: str
    url: Optional[str] = None
    location: Optional[str] = None

@router.get("/cameras", response_model=List[Camera])
async def get_all_cameras():
    """Get all cameras"""
    detect_and_sync_usb_cameras()
    response = supabase.table("cameras").select("*").order("created_at", desc=True).execute()
    return response.data or []


@router.post("/cameras", response_model=Camera)
async def add_camera(camera: CameraCreate):
    """Add new camera"""
    new_camera = {
        "name": camera.name,
        "type": camera.type,
        "url": camera.url,
        "location": camera.location,
        "status": "offline",
        "last_active": datetime.utcnow().isoformat()
    }

    response = supabase.table("cameras").insert(new_camera).execute()

    if not response.data:
        raise HTTPException(status_code=500, detail="Failed to add camera")

    return response.data[0]


@router.get("/cameras/{camera_id}")
async def get_camera(camera_id: str):
    response = supabase.table("cameras").select("*").eq("id", camera_id).execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="Camera not found")
    return response.data[0]