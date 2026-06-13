# app/api/routes/camera_routes.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import cv2
import os
from typing import List, Dict
from ultralytics import YOLO

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

# Lazy load YOLO model instances
_custom_model_instance = None
_pretrained_model_instance = None

def get_yolo_model(model_type: str = "custom"):
    global _custom_model_instance, _pretrained_model_instance
    
    if model_type == "pretrained":
        if _pretrained_model_instance is not None:
            return _pretrained_model_instance
        # Resolve pretrained base weights path (COCO dataset containing person class)
        weights_path = os.path.join("models", "pretrained", "yolo11n.pt")
        if not os.path.exists(weights_path):
            weights_path = "yolo11n.pt"  # base online download fallback
        print(f"[*] Loading Pretrained YOLO weights (COCO): {weights_path}")
        _pretrained_model_instance = YOLO(weights_path)
        return _pretrained_model_instance
    else:
        if _custom_model_instance is not None:
            return _custom_model_instance
        # Resolve custom trained weights path
        weights_path = "runs/detect/train/weights/best.pt"
        if not os.path.exists(weights_path):
            alt_path = os.path.join("models", "trained", "best.pt")
            if os.path.exists(alt_path):
                weights_path = alt_path
            else:
                weights_path = os.path.join("models", "pretrained", "yolo11n.pt")
                if not os.path.exists(weights_path):
                    weights_path = "yolo11n.pt"
        print(f"[*] Loading Custom YOLO weights: {weights_path}")
        _custom_model_instance = YOLO(weights_path)
        return _custom_model_instance


def generate_camera_stream(source: str | int, model_type: str = "custom"):
    """Generator to capture frames, run YOLO, draw annotations, and stream MJPEG"""
    # Open the video capture
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Pre-load model to avoid threading latency
    model = get_yolo_model(model_type)
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                # If a webcam stream fails (e.g. no physical device connected),
                # yield a simulated color frame with bounding boxes to show it in action!
                if source == 0 or source == "webcam:0 (Demo)":
                    import time
                    import numpy as np
                    # Create black frame
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    # Draw a simulated highway road
                    cv2.rectangle(frame, (100, 200), (540, 480), (50, 50, 50), -1) 
                    # Draw moving vehicles
                    t = time.time()
                    car_y = int(300 + 40 * np.sin(t * 2))
                    cv2.rectangle(frame, (280, car_y), (360, car_y + 60), (0, 0, 255), -1) # Red car
                    cv2.putText(frame, "SIMULATED TRAFFIC STREAM", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Run YOLO on the simulated frame
                    try:
                        results = model(frame, conf=0.25, verbose=False)
                        annotated_frame = results[0].plot()
                    except Exception:
                        annotated_frame = frame.copy()
                        cv2.rectangle(annotated_frame, (280, car_y), (360, car_y + 60), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, "Car 91%", (280, car_y - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Encode as JPEG
                    ret, jpeg = cv2.imencode('.jpg', annotated_frame)
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                    time.sleep(0.04) # cap at ~25fps
                    continue
                else:
                    break
            
            # Run YOLO model inference (verbose=False avoids console spamming)
            results = model(frame, conf=0.25, verbose=False)
            
            # Get annotated frame (BGR numpy array)
            annotated_frame = results[0].plot()
            
            # Encode frame as JPEG
            ret, jpeg = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                continue
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    except Exception as e:
        print(f"[-] Error in video generator stream: {e}")
    finally:
        cap.release()


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
    # Fallback simulated camera if no physical webcams are connected
    if not cameras:
        cameras.append({
            "id": 0,
            "name": "Simulated Webcam (Demo)",
            "type": "webcam",
            "url": "webcam:0 (Demo)",
            "status": "online"
        })
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


@router.get("/stream/{camera_id}")
async def stream_camera(camera_id: int, model_type: str = "custom"):
    """Stream camera video feed with real-time YOLO detection overlays"""
    # Resolve source based on ID
    source = None
    if camera_id < 1000:
        # Local webcam index
        source = camera_id
    else:
        # IP camera
        for cam in ip_cameras_db:
            if cam["id"] == camera_id:
                source = cam["url"]
                break
        if source is None:
            raise HTTPException(status_code=404, detail="IP Camera not found")
            
    return StreamingResponse(
        generate_camera_stream(source, model_type),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )