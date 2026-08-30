# app/schemas/alert.py
from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class AlertResponse(BaseModel):
    id: str
    user_id: str
    camera_id: str
    detection_event_id: Optional[str] = None
    snapshot_url: Optional[str] = None
    detection_type: Optional[str] = None
    confidence: Optional[float] = None
    status: str = "unread"
    created_at: Optional[str] = None

    class Config:
        from_attributes = True
