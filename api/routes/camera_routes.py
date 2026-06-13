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


def generate_camera_stream(source: str | int, camera_id: int, detect: bool = True):
    """Generator to capture frames, run dual YOLO inference (custom + person), and stream MJPEG"""
    import time
    import numpy as np

    # Pre-load both models if detection is enabled
    custom_model, pretrained_model = None, None
    if detect:
        try:
            custom_model, pretrained_model = load_yolo_models()
        except Exception as e:
            print(f"[-] YOLO load error: {e}")

    # Keep track of detection intervals
    last_detect_time = 0.0
    cached_custom_results = None
    cached_pretrained_results = None
    
    # Countdown and snapshot states
    first_detection_time = None
    snapshot_taken = False
    
    cap = None
    last_retry_time = 0.0
    is_demo = (source == "webcam:0 (Demo)")
    
    try:
        while True:
            current_time = time.time()
            success = False
            frame = None
            
            # If not a demo stream, try to connect to the video source
            if not is_demo:
                if cap is None or not cap.isOpened():
                    if current_time - last_retry_time >= 5.0:
                        last_retry_time = current_time
                        print(f"[*] Attempting to connect to camera source: {source}")
                        if cap is not None:
                            cap.release()
                        try:
                            cap = cv2.VideoCapture(source)
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        except Exception as e:
                            print(f"[-] VideoCapture error for {source}: {e}")
                
                if cap is not None and cap.isOpened():
                    try:
                        success, frame = cap.read()
                    except Exception as e:
                        print(f"[-] Frame read error: {e}")
                        success = False

            # If reading failed or it's a demo stream
            if not success or frame is None:
                # Check if we should render the simulated traffic stream (for local webcam 0 fallback or demo)
                if is_demo or source == 0:
                    # Create simulated traffic frame
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    # Draw a simulated highway road
                    cv2.rectangle(frame, (100, 200), (540, 480), (50, 50, 50), -1) 
                    # Draw moving vehicles
                    t = time.time()
                    car_y = int(300 + 40 * np.sin(t * 2))
                    cv2.rectangle(frame, (280, car_y), (360, car_y + 60), (0, 0, 255), -1) # Red car
                    cv2.putText(frame, "SIMULATED TRAFFIC STREAM", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    annotated_frame = frame.copy()

                    if detect:
                        # Run YOLO on the simulated frame once every 5 seconds
                        if custom_model is not None and pretrained_model is not None:
                            if current_time - last_detect_time >= 5.0:
                                try:
                                    cached_custom_results = custom_model(frame, conf=0.25, verbose=False)
                                    cached_pretrained_results = pretrained_model(frame, classes=[0], conf=0.25, verbose=False)
                                    last_detect_time = current_time
                                except Exception as e:
                                    print(f"[-] YOLO inference error: {e}")
                        
                        # Overlay cached boxes if available
                        if cached_custom_results is not None:
                            try:
                                annotated_frame = cached_custom_results[0].plot(img=annotated_frame)
                            except Exception:
                                try:
                                    annotated_frame = cached_custom_results[0].plot()
                                except Exception:
                                    pass
                                    
                        if cached_pretrained_results is not None:
                            try:
                                annotated_frame = cached_pretrained_results[0].plot(img=annotated_frame)
                            except Exception:
                                pass
                                    
                        # Simulated stream countdown tracking
                        something_detected = False
                        if cached_custom_results is not None:
                            something_detected = something_detected or len(cached_custom_results[0].boxes) > 0
                        if cached_pretrained_results is not None:
                            something_detected = something_detected or len(cached_pretrained_results[0].boxes) > 0
                            
                        if something_detected:
                            if first_detection_time is None:
                                first_detection_time = current_time
                                snapshot_taken = False
                            
                            elapsed = current_time - first_detection_time
                            remaining = 60.0 - elapsed
                            
                            if remaining > 0:
                                msg = f"Alert: Object persists. Snap in {int(remaining)}s"
                                cv2.putText(annotated_frame, msg, (20, 80), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            else:
                                if not snapshot_taken:
                                    os.makedirs("snap", exist_ok=True)
                                    timestamp = int(time.time())
                                    snap_path = os.path.join("snap", f"snap_cam{camera_id}_{timestamp}.jpg")
                                    cv2.imwrite(snap_path, annotated_frame)
                                    print(f"[+] Snapshot saved: {snap_path}")
                                    snapshot_taken = True
                                    first_detection_time = current_time  # Reset timer to snap again if stays
                                cv2.putText(annotated_frame, "SNAPSHOT CAPTURED!", (20, 80), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        else:
                            first_detection_time = None
                            snapshot_taken = False
                    
                    # Encode as JPEG
                    ret, jpeg = cv2.imencode('.jpg', annotated_frame)
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                    time.sleep(0.04) # cap at ~25fps
                    continue
                else:
                    # For failed IP cameras or other sources, generate a status/offline placeholder frame
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    for y in range(480):
                        color_val = int(20 + (y / 480) * 15)
                        frame[y, :] = [color_val + 5, color_val, color_val]
                    
                    cv2.rectangle(frame, (0, 0), (640, 60), (40, 30, 30), -1)
                    cv2.putText(frame, "VISIONGUARD SECURITY FEED", (20, 38), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 255), 2, cv2.LINE_AA)
                    
                    pulse = int(120 + 20 * np.sin(current_time * 3))
                    cv2.circle(frame, (320, 240), pulse, (164, 96, 10), 1)
                    cv2.circle(frame, (320, 240), 10, (164, 96, 10), -1)
                    
                    status_text = "CAMERA OFFLINE / CONNECTING"
                    if isinstance(source, str) and source.startswith("http"):
                        hint_text = f"URL: {source}"
                        tip_text = "Tip: Verify IP address & port (e.g. http://ip:port/video)"
                    else:
                        hint_text = f"Source ID: {source}"
                        tip_text = "Tip: Check webcam connection or device permissions"
                    
                    cv2.putText(frame, status_text, (130, 220), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 240), 2, cv2.LINE_AA)
                    cv2.putText(frame, hint_text, (60, 280), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
                    cv2.putText(frame, tip_text, (60, 310), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)
                    
                    ret, jpeg = cv2.imencode('.jpg', frame)
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                    time.sleep(0.1) # low frame rate placeholder to save CPU
                    continue
            
            annotated_frame = frame.copy()

            if detect:
                # Run YOLO model inference once every 5 seconds (both custom vehicles and pretrained persons)
                if current_time - last_detect_time >= 5.0:
                    if custom_model is not None and pretrained_model is not None:
                        cached_custom_results = custom_model(frame, conf=0.25, verbose=False)
                        # classes=[0] filters pretrained COCO model to ONLY detect 'person'
                        cached_pretrained_results = pretrained_model(frame, classes=[0], conf=0.25, verbose=False)
                    last_detect_time = current_time
                
                # Overlay custom vehicle detection annotations
                if cached_custom_results is not None:
                    try:
                        annotated_frame = cached_custom_results[0].plot(img=annotated_frame)
                    except Exception:
                        try:
                            annotated_frame = cached_custom_results[0].plot()
                        except Exception:
                            pass
                
                # Overlay pretrained person detection annotations
                if cached_pretrained_results is not None:
                    try:
                        annotated_frame = cached_pretrained_results[0].plot(img=annotated_frame)
                    except Exception:
                        pass
                
                # Check if something is currently detected (custom vehicles or persons)
                something_detected = False
                if cached_custom_results is not None:
                    something_detected = something_detected or len(cached_custom_results[0].boxes) > 0
                if cached_pretrained_results is not None:
                    something_detected = something_detected or len(cached_pretrained_results[0].boxes) > 0
                    
                if something_detected:
                    if first_detection_time is None:
                        first_detection_time = current_time
                        snapshot_taken = False
                    
                    elapsed = current_time - first_detection_time
                    remaining = 60.0 - elapsed
                    
                    if remaining > 0:
                        msg = f"Alert: Object persists. Snap in {int(remaining)}s"
                        cv2.putText(annotated_frame, msg, (20, 80), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        if not snapshot_taken:
                            os.makedirs("snap", exist_ok=True)
                            timestamp = int(time.time())
                            snap_path = os.path.join("snap", f"snap_cam{camera_id}_{timestamp}.jpg")
                            cv2.imwrite(snap_path, annotated_frame)
                            print(f"[+] Snapshot saved: {snap_path}")
                            snapshot_taken = True
                            first_detection_time = current_time  # Reset timer
                        cv2.putText(annotated_frame, "SNAPSHOT CAPTURED!", (20, 80), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    first_detection_time = None
                    snapshot_taken = False
            
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
        if cap is not None:
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
async def stream_camera(camera_id: int, detect: bool = True):
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
        generate_camera_stream(source, camera_id, detect),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )