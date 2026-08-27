from typing import List, Dict
from datetime import datetime
from app.schemas.camera import Camera

_in_memory_cameras = {}
active_feeds = set()
_camera_zones: Dict[str, list] = {}
_zone_snapshot_cooldowns: Dict[str, float] = {}
ZONE_SNAPSHOT_COOLDOWN_SECS = 10

def get_detected_cameras() -> List[Camera]:
    detected = []
    
    detected.append(Camera(
        id="cctv_1",
        name="Camera 01",
        type="cctv",
        url="CCTV/Cam01",
        status="online",
        last_active=datetime.utcnow().isoformat(),
        location="Front Gate"
    ))
    
    detected.append(Camera(
        id="cctv_8",
        name="Test Cam 01",
        type="cctv",
        url="CCTV/TestCam01",
        status="online",
        last_active=datetime.utcnow().isoformat(),
        location="Mark For Vehicle"
    ))
    
    detected.append(Camera(
        id="cctv_9",
        name="Test Cam 02",
        type="cctv",
        url="CCTV/TestCam02",
        status="online",
        last_active=datetime.utcnow().isoformat(),
        location="Mark For Human"
    ))
    
    for cam_id, cam in _in_memory_cameras.items():
        if not any(d.id == cam_id for d in detected):
            detected.append(cam)
            
    return detected
