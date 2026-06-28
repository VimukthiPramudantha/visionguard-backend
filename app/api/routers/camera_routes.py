# app/api/routers/camera_routes.py
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from app.core.supabase import supabase

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

@router.get("/cameras", response_model=List[Camera])
async def get_all_cameras():
    """Get all cameras"""
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