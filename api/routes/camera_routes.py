# app/api/routes/camera_routes.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import cv2
from typing import List, Dict

router = APIRouter(prefix="/cameras", tags=["Cameras"])

class Camera(BaseModel):
    id: int
    name: str
    type: str 
    url: str
    status: str = "offline"

class AddIPCamera(BaseModel):
    name: str
    rtsp_url: str

# In-memory storage for added IP cameras
ip_cameras_db: List[Dict] = []

@router.get("/local-webcams", response_model=List[Dict])
async def get_local_webcams():
    """Detect all connected webcams on the PC"""
    cameras = []
    # Test indices 0 to 4 to find connected webcams
    for i in range(5): 
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append({
                "id": i,
                "name": f"Webcam {i}",
                "type": "webcam",
                "url": f"webcam:{i}",
                "status": "online"
            })
            cap.release()
    return cameras


@router.post("/ip", response_model=Camera)
async def add_ip_camera(camera: AddIPCamera):
    """Add a new IP Camera (RTSP)"""
    if not camera.rtsp_url.startswith(("rtsp://", "http://", "https://")):
        raise HTTPException(status_code=400, detail="Invalid RTSP/URL format")
    
    new_id = 1000 + len(ip_cameras_db)
    new_camera = {
        "id": new_id,
        "name": camera.name,
        "type": "ip",
        "url": camera.rtsp_url,
        "status": "online" 
    }
    ip_cameras_db.append(new_camera)
    return new_camera


@router.get("/all", response_model=List[Camera])
async def get_all_cameras():
    """Return both local + added IP cameras"""
    local = await get_local_webcams()
    return local + ip_cameras_db