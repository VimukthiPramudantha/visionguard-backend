from pydantic import BaseModel
from typing import List, Optional

class Camera(BaseModel):
    id: str
    name: str
    type: str  
    url: Optional[str] = None
    status: str
    last_active: Optional[str] = None
    location: Optional[str] = None
    user_id: Optional[str] = None
    zone_points: Optional[List[dict]] = None
    created_at: Optional[str] = None

class CameraCreate(BaseModel):
    name: str
    type: str
    url: Optional[str] = None
    location: Optional[str] = None
    user_id: Optional[str] = None

class ZonePoint(BaseModel):
    x: float
    y: float

class ZonePayload(BaseModel):
    points: List[ZonePoint]
