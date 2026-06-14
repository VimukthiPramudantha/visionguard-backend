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
_custom_model = None
_pretrained_model = None

def load_yolo_models():
    """Load both custom and pretrained YOLO models into memory"""
    global _custom_model, _pretrained_model
    
    # 1. Load Pretrained YOLO (for person detection)
    if _pretrained_model is None:
        weights_path = os.path.join("models", "pretrained", "yolo11n.pt")
        if not os.path.exists(weights_path):
            weights_path = "yolo11n.pt"
        print(f"[*] Loading Pretrained YOLO weights (COCO): {weights_path}")
        _pretrained_model = YOLO(weights_path)
        
    # 2. Load Custom YOLO (for custom vehicle detection)
    if _custom_model is None:
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
        _custom_model = YOLO(weights_path)
        
    return _custom_model, _pretrained_model


def generate_camera_stream(source: str | int):
    """Generator to capture frames, run dual YOLO inference (custom + person), and stream MJPEG"""
    # Open the video capture
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Pre-load both models
    custom_model, pretrained_model = load_yolo_models()
    
    # Keep track of detection intervals
    import time
    last_detect_time = 0.0
    cached_custom_results = None
    cached_pretrained_results = None
    
    try:
        while True:
            success, frame = cap.read()
            current_time = time.time()
            
            if not success:
                # If a webcam stream fails (e.g. no physical device connected),
                # yield a simulated color frame with bounding boxes to show it in action!
                if source == 0 or source == "webcam:0 (Demo)":
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
                    
                    # Run YOLO on the simulated frame once every 5 seconds
                    if current_time - last_detect_time >= 5.0:
                        try:
                            cached_custom_results = custom_model(frame, conf=0.25, verbose=False)
                            cached_pretrained_results = pretrained_model(frame, classes=[0], conf=0.25, verbose=False)
                            last_detect_time = current_time
                        except Exception as e:
                            print(f"[-] YOLO inference error: {e}")
                    
                    # Overlay cached boxes if available
                    annotated_frame = frame.copy()
                    if cached_custom_results is not None:
                        try:
                            annotated_frame = cached_custom_results[0].plot(img=annotated_frame)
                        except Exception:
                            annotated_frame = cached_custom_results[0].plot()
                            
                    if cached_pretrained_results is not None:
                        try:
                            annotated_frame = cached_pretrained_results[0].plot(img=annotated_frame)
                        except Exception:
                            pass
                    
                    # Encode as JPEG
                    ret, jpeg = cv2.imencode('.jpg', annotated_frame)
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                    time.sleep(0.04) # cap at ~25fps
                    continue
                else:
                    break
            
            # Run YOLO model inference once every 5 seconds (both custom vehicles and pretrained persons)
            if current_time - last_detect_time >= 5.0:
                cached_custom_results = custom_model(frame, conf=0.25, verbose=False)
                # classes=[0] filters pretrained COCO model to ONLY detect 'person'
                cached_pretrained_results = pretrained_model(frame, classes=[0], conf=0.25, verbose=False)
                last_detect_time = current_time
            
            # Overlay custom vehicle detection annotations
            annotated_frame = frame.copy()
            if cached_custom_results is not None:
                try:
                    annotated_frame = cached_custom_results[0].plot(img=annotated_frame)
                except Exception:
                    annotated_frame = cached_custom_results[0].plot()
            
            # Overlay pretrained person detection annotations
            if cached_pretrained_results is not None:
                try:
                    annotated_frame = cached_pretrained_results[0].plot(img=annotated_frame)
                except Exception:
                    pass
            
            # Encode frame as JPEG
            ret, jpeg = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                continue
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            
            # Tiny sleep to regulate throughput in real streams
            time.sleep(0.01)
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
        "status": "online"  # Mark added IP cameras as online for demo purposes
    }
    ip_cameras_db.append(new_camera)
    return new_camera


@router.get("/all", response_model=List[Camera])
async def get_all_cameras():
    """Return both local + added IP cameras"""
    local = await get_local_webcams()
    return local + ip_cameras_db


@router.get("/stream/{camera_id}")
async def stream_camera(camera_id: int):
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
        generate_camera_stream(source),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )