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
active_feeds = set()

def get_detected_cameras() -> List[Camera]:
    detected = []
    
    # 1. Try to scan physical USB cameras using OpenCV
    try:
        import cv2
        for index in range(3):  # Check indices 0, 1, 2
            cam_id = f"usb_{index}"
            # If this camera is currently streaming, it is definitely online/connected
            if cam_id in active_feeds:
                detected.append(Camera(
                    id=cam_id,
                    name=f"Integrated Camera {index}" if index == 0 else f"USB Camera {index}",
                    type="usb",
                    url=str(index),
                    status="online",
                    last_active=datetime.utcnow().isoformat(),
                    location="Local Host"
                ))
                continue

            # Otherwise, test-open it using default backend (no CAP_DSHOW to avoid exceptions)
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
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

@router.get("/cameras/{camera_id}/feed")
async def get_camera_feed(camera_id: str):
    from fastapi.responses import StreamingResponse
    import cv2
    import numpy as np
    import time

    cameras = get_detected_cameras()
    camera = next((c for c in cameras if c.id == camera_id), None)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    is_simulated = (camera.id == "simulated_0")

    def gen_frames():
        active_feeds.add(camera_id)
        try:
            if is_simulated:
                width, height = 640, 480
                while True:
                    img = np.zeros((height, width, 3), dtype=np.uint8)
                    t_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cv2.putText(img, "VisionGuard Feed", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                    cv2.putText(img, "SIMULATED LIVE", (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (14, 165, 233), 2)
                    cv2.putText(img, t_str, (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (34, 197, 94), 2)
                    
                    y = int(240 + 80 * np.sin(time.time() * 2))
                    cv2.line(img, (50, y), (590, y), (0, 0, 255), 2)
                    
                    ret, buffer = cv2.imencode('.jpg', img)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    time.sleep(0.04)  # ~25 FPS
            else:
                try:
                    val = int(camera.url)
                except ValueError:
                    val = camera.url
                
                cap = cv2.VideoCapture(val)
                if not cap.isOpened():
                    # Fallback to simulated pattern if opening webcam fails
                    width, height = 640, 480
                    while True:
                        img = np.zeros((height, width, 3), dtype=np.uint8)
                        t_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        cv2.putText(img, camera.name, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                        cv2.putText(img, "CAMERA UNREACHABLE", (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (239, 68, 68), 2)
                        cv2.putText(img, t_str, (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 116, 139), 2)
                        
                        ret, buffer = cv2.imencode('.jpg', img)
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                        time.sleep(0.04)
                else:
                    try:
                        while True:
                            success, frame = cap.read()
                            if not success:
                                break
                            ret, buffer = cv2.imencode('.jpg', frame)
                            if not ret:
                                continue
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    finally:
                        cap.release()
        finally:
            active_feeds.discard(camera_id)

    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")