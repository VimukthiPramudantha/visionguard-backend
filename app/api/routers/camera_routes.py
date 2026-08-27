from fastapi import APIRouter, HTTPException, status, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import asyncio
import cv2
import numpy as np
import base64

router = APIRouter()

class Camera(BaseModel):
    id: str
    name: str
    type: str  
    url: Optional[str] = None
    status: str
    last_active: Optional[str] = None
    location: Optional[str] = None

class CameraCreate(BaseModel):
    name: str
    type: str
    url: Optional[str] = None
    location: Optional[str] = None

_in_memory_cameras = {}
active_feeds = set()

def get_detected_cameras() -> List[Camera]:
    detected = []
    
    detected.append(Camera( id="cctv_1", name="Camera 01", type="cctv", url="CCTV/Cam01", status="online", last_active=datetime.utcnow().isoformat(), location="Front Gate" ))

    # detected.append(Camera( id="cctv_2", name="Camera 02", type="cctv", url="CCTV/Cam02", status="online", last_active=datetime.utcnow().isoformat(), location="Main Hall" ))

    # detected.append(Camera( id="cctv_3", name="Camera 03", type="cctv", url="CCTV/Cam03", status="online", last_active=datetime.utcnow().isoformat(), location="Backyard"))

    # detected.append(Camera( id="cctv_4", name="Camera 04", type="cctv", url="CCTV/Cam04", status="online", last_active=datetime.utcnow().isoformat(), location="Road" ))

    # detected.append(Camera( id="cctv_5", name="Camera 05", type="cctv", url="CCTV/Cam05", status="online", last_active=datetime.utcnow().isoformat(),location="Parking" ))

    # detected.append(Camera( id="cctv_6", name="Camera 06", type="cctv", url="CCTV/Cam06", status="online", last_active=datetime.utcnow().isoformat(), location="Traffic" ))

    # detected.append(Camera( id="cctv_7",name="Camera 07", type="cctv", url="CCTV/Cam07", status="online", last_active=datetime.utcnow().isoformat(), location="Human" ))
    
    detected.append(Camera( id="cctv_8",name="Test Cam 01", type="cctv", url="CCTV/TestCam01", status="online", last_active=datetime.utcnow().isoformat(), location="Mark For Vehicle" ))

    detected.append(Camera( id="cctv_9",name="Test Cam 02", type="cctv", url="CCTV/TestCam02", status="online", last_active=datetime.utcnow().isoformat(), location="Mark For Human" ))
    

        
    for cam_id, cam in _in_memory_cameras.items():
        if not any(d.id == cam_id for d in detected):
            detected.append(cam)
            
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

_yolo_model = None

_VEHICLE_CLASSES = {"bicycle", "bus", "car", "motorbike", "truck"}

_BOX_COLORS = {
    "vehicle": (0, 140, 255),   
    "person":  (0, 220, 100),   
}


def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            model_path = r"d:\Projects\VisionGuard\visionguard-backend\runs\detect\combined_train\weights\best.pt"
            import os
            if os.path.exists(model_path):
                _yolo_model = YOLO(model_path)
                print(f"[VisionGuard] Model loaded. Classes: {_yolo_model.names}")
            else:
                print(f"YOLO model not found at {model_path}")
                _yolo_model = False
        except ImportError as e:
            import sys
            print(f"ultralytics import failed: {e}")
            print(f"Python path: {sys.path}")
            print(f"Python executable: {sys.executable}")
            _yolo_model = False
        except Exception as e:
            print(f"Failed to load YOLO: {e}")
            _yolo_model = False
    return _yolo_model if _yolo_model is not False else None


def run_detection(model, frame):

    import cv2

    results = model(
        frame,
        conf=0.45,
        iou=0.35,
        augment=False,
        imgsz=640,
        device=0,
        verbose=False,
    )

    annotated = frame.copy()
    for box in results[0].boxes:
        cls_id   = int(box.cls[0])
        conf_val = float(box.conf[0])
        raw_name = model.names[cls_id]

        label = "vehicle" if raw_name in _VEHICLE_CLASSES else "person"
        color = _BOX_COLORS.get(label, (200, 200, 200))

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf_val:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            annotated, text,
            (x1 + 3, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 0, 0), 1, cv2.LINE_AA,
        )

    return annotated


@router.get("/cameras/{camera_id}/feed")
def get_camera_feed(camera_id: str):
    from fastapi.responses import StreamingResponse
    import cv2
    import numpy as np
    import time
    import os

    cameras = get_detected_cameras()
    camera = next((c for c in cameras if c.id == camera_id), None)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    is_simulated = (camera.id == "simulated_0")
    is_cctv = camera.id.startswith("cctv_")

    def gen_frames():
        active_feeds.add(camera_id)
        try:
            model = get_yolo_model()
            if is_cctv:
                backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
                cam_path = os.path.join(backend_root, camera.url)
                
                if os.path.exists(cam_path) and os.path.isdir(cam_path):
                    video_files = sorted([
                        os.path.join(cam_path, f)
                        for f in os.listdir(cam_path)
                        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))
                    ], key=lambda x: x.lower())
                else:
                    video_files = []
                
                if not video_files:
                    width, height = 640, 480
                    while True:
                        img = np.zeros((height, width, 3), dtype=np.uint8)
                        t_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        cv2.putText(img, camera.name, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                        cv2.putText(img, "NO VIDEO FILES FOUND", (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (239, 68, 68), 2)
                        cv2.putText(img, t_str, (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 116, 139), 2)
                        
                        ret, buffer = cv2.imencode('.jpg', img)
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                        time.sleep(0.04)
                
                video_index = 0
                while True:
                    video_path = video_files[video_index]
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        video_index = (video_index + 1) % len(video_files)
                        time.sleep(1)
                        continue
                    
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if fps <= 0 or np.isnan(fps):
                        fps = 25.0
                    delay = 1.0 / fps
                    
                    try:
                        while True:
                            start_time = time.time()
                            success, frame = cap.read()
                            if not success:
                                break
                            
                            if model:
                                try:
                                    frame = run_detection(model, frame)
                                except Exception as e:
                                    print(f"Error during YOLO inference (CCTV): {e}")

                            ret, buffer = cv2.imencode('.jpg', frame)
                            if not ret:
                                continue
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                            
                            elapsed = time.time() - start_time
                            sleep_time = max(0.001, delay - elapsed)
                            time.sleep(sleep_time)
                    finally:
                        cap.release()
                    
                    video_index = (video_index + 1) % len(video_files)

            elif is_simulated:
                width, height = 640, 480
                while True:
                    img = np.zeros((height, width, 3), dtype=np.uint8)
                    t_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cv2.putText(img, "VisionGuard Feed", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                    cv2.putText(img, "SIMULATED LIVE", (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (14, 165, 233), 2)
                    cv2.putText(img, t_str, (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (34, 197, 94), 2)
                    
                    y = int(240 + 80 * np.sin(time.time() * 2))
                    cv2.line(img, (50, y), (590, y), (0, 0, 255), 2)
                    
                    if model:
                        try:
                            img = run_detection(model, img)
                        except Exception as e:
                            print(f"Error during YOLO inference (Simulated): {e}")

                    ret, buffer = cv2.imencode('.jpg', img)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    time.sleep(0.04)  
            else:
                try:
                    val = int(camera.url)
                except ValueError:
                    val = camera.url
                
                if isinstance(val, int):
                    cap = cv2.VideoCapture(val, cv2.CAP_MSMF)
                else:
                    cap = cv2.VideoCapture(val)
                    
                if not cap.isOpened():
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
                            
                            if model:
                                try:
                                    frame = run_detection(model, frame)
                                except Exception as e:
                                    print(f"Error during YOLO inference (USB): {e}")

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


@router.websocket("/cameras/{camera_id}/ws")
async def websocket_camera_feed(websocket: WebSocket, camera_id: str):

    await websocket.accept()
    
    cameras = get_detected_cameras()
    camera = next((c for c in cameras if c.id == camera_id), None)
    if not camera:
        await websocket.close(code=1008, reason="Camera not found")
        return

    is_simulated = (camera.id == "simulated_0")
    is_cctv = camera.id.startswith("cctv_")
    
    active_feeds.add(camera_id)
    cap = None
    
    async def receive_messages():
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass

    receiver_task = asyncio.create_task(receive_messages())

    try:
        model = get_yolo_model()
        
        if is_cctv:
            import os
            backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            cam_path = os.path.join(backend_root, camera.url)
            if os.path.exists(cam_path) and os.path.isdir(cam_path):
                video_files = sorted([
                    os.path.join(cam_path, f)
                    for f in os.listdir(cam_path)
                    if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))
                ], key=lambda x: x.lower())
            else:
                video_files = []
            
            if not video_files:
                while not receiver_task.done():
                    img = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(img, "NO VIDEO FILES FOUND", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    _, buffer = cv2.imencode('.jpg', img)
                    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                    await websocket.send_text(jpg_as_text)
                    await asyncio.sleep(0.04)
            
            video_index = 0
            while not receiver_task.done():
                video_path = video_files[video_index]
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    video_index = (video_index + 1) % len(video_files)
                    await asyncio.sleep(1)
                    continue
                
                fps = cap.get(cv2.CAP_PROP_FPS)
                delay = 1.0 / (fps if fps > 0 else 25.0)
                
                try:
                    while not receiver_task.done():
                        t0 = asyncio.get_event_loop().time()
                        success, frame = cap.read()
                        if not success:
                            break
                        
                        if model:
                            try:
                                frame = run_detection(model, frame)
                            except Exception:
                                pass
                        
                        ret, buffer = cv2.imencode('.jpg', frame)
                        if ret:
                            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                            await websocket.send_text(jpg_as_text)
                        
                        elapsed = asyncio.get_event_loop().time() - t0
                        sleep_time = max(0.001, delay - elapsed)
                        await asyncio.sleep(sleep_time)
                finally:
                    cap.release()
                
                video_index = (video_index + 1) % len(video_files)
                
        elif is_simulated:
            while not receiver_task.done():
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                t_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(img, "VisionGuard WS Feed", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                cv2.putText(img, "SIMULATED LIVE WS", (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (14, 165, 233), 2)
                cv2.putText(img, t_str, (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (34, 197, 94), 2)
                
                if model:
                    try:
                        img = run_detection(model, img)
                    except Exception:
                        pass
                
                ret, buffer = cv2.imencode('.jpg', img)
                if ret:
                    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                    await websocket.send_text(jpg_as_text)
                await asyncio.sleep(0.04)
                
        else:
            try:
                val = int(camera.url)
            except ValueError:
                val = camera.url
            
            cap = cv2.VideoCapture(val, cv2.CAP_MSMF) if isinstance(val, int) else cv2.VideoCapture(val)
            if not cap.isOpened():
                while not receiver_task.done():
                    img = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(img, "CAMERA UNREACHABLE", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    _, buffer = cv2.imencode('.jpg', img)
                    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                    await websocket.send_text(jpg_as_text)
                    await asyncio.sleep(0.04)
            
            while not receiver_task.done():
                success, frame = cap.read()
                if not success:
                    await asyncio.sleep(0.01)
                    continue
                
                if model:
                    try:
                        frame = run_detection(model, frame)
                    except Exception:
                        pass
                
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                    await websocket.send_text(jpg_as_text)
                await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        print(f"[WS] Client disconnected from camera {camera_id}")
    except Exception as e:
        print(f"[WS] Error streaming camera {camera_id}: {e}")
    finally:
        receiver_task.cancel()
        if cap and cap.isOpened():
            cap.release()
        active_feeds.discard(camera_id)