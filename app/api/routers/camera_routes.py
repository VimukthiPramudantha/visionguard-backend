# app/api/routers/camera_routes.py
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

router = APIRouter()

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

# A simple list of manually added cameras stored in memory
_in_memory_cameras = {}

def get_detected_cameras() -> List[Camera]:
    detected = []
    
    # 1. Try to scan physical USB cameras using OpenCV
    try:
        import cv2
        for index in range(3):  # Check indices 0, 1, 2
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if hasattr(cv2, 'CAP_DSHOW') else cv2.VideoCapture(index)
            if cap.isOpened():
                cam_id = f"usb_{index}"
                detected.append(Camera(
                    id=cam_id,
                    name=f"Integrated Camera {index}" if index == 0 else f"USB Camera {index}",
                    type="usb",
                    url=str(index),
                    status="online",
                    last_active=datetime.utcnow().isoformat(),
                    location="Local Host"
                ))
                cap.release()
    except Exception as e:
        print(f"Error scanning local webcams: {e}")
        
    # 2. Add any manually added cameras (stored in memory)
    for cam_id, cam in _in_memory_cameras.items():
        if not any(d.id == cam_id for d in detected):
            detected.append(cam)
            
    # 3. Fallback to at least one simulated camera if absolutely nothing is connected
    if not detected:
        detected.append(Camera(
            id="simulated_0",
            name="Integrated Camera 0 (Simulated)",
            type="usb",
            url="0",
            status="online",
            last_active=datetime.utcnow().isoformat(),
            location="Simulated Environment"
        ))
        
    return detected

@router.get("/cameras", response_model=List[Camera])
async def get_all_cameras():
    """Get all cameras"""
    return get_detected_cameras()

@router.post("/cameras", response_model=Camera)
async def add_camera(camera: CameraCreate):
    """Add new camera to memory (no database)"""
    cam_id = f"custom_{len(_in_memory_cameras) + 1}"
    new_cam = Camera(
        id=cam_id,
        name=camera.name,
        type=camera.type,
        url=camera.url or "0",
        location=camera.location or "Custom Location",
        status="online",
        last_active=datetime.utcnow().isoformat()
    )
    _in_memory_cameras[cam_id] = new_cam
    return new_cam

@router.get("/cameras/{camera_id}", response_model=Camera)
async def get_camera(camera_id: str):
    cameras = get_detected_cameras()
    for cam in cameras:
        if cam.id == camera_id:
            return cam
    raise HTTPException(status_code=404, detail="Camera not found")