from fastapi import APIRouter, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from typing import List, Optional, Dict
from datetime import datetime
import asyncio
import cv2
import numpy as np
import base64
import os
import time
import json

from app.schemas.camera import Camera, CameraCreate, ZonePayload
from app.api.routers.camera.state import (
    _in_memory_cameras,
    active_feeds,
    _camera_zones,
    _zone_snapshot_cooldowns,
    get_detected_cameras
)
from app.api.routers.camera.detection import get_yolo_model, run_detection
from app.api.routers.camera.utils import check_zone_intrusion, save_intrusion_snapshot, draw_zone_overlay

router = APIRouter()

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

@router.post("/cameras/{camera_id}/zone")
async def set_camera_zone(camera_id: str, payload: ZonePayload):
    cameras = get_detected_cameras()
    if not any(c.id == camera_id for c in cameras):
        raise HTTPException(status_code=404, detail="Camera not found")
    _camera_zones[camera_id] = [p.model_dump() for p in payload.points]
    return {"status": "ok", "camera_id": camera_id, "points": _camera_zones[camera_id]}

@router.get("/cameras/{camera_id}/zone")
async def get_camera_zone(camera_id: str):
    zone = _camera_zones.get(camera_id)
    return {"camera_id": camera_id, "points": zone}

@router.delete("/cameras/{camera_id}/zone")
async def delete_camera_zone(camera_id: str):
    _camera_zones.pop(camera_id, None)
    return {"status": "ok", "camera_id": camera_id}

@router.get("/cameras/{camera_id}/feed")
def get_camera_feed(camera_id: str):
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
                backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
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
                            
                            detections = []
                            if model:
                                try:
                                    frame, detections = run_detection(model, frame)
                                except Exception as e:
                                    print(f"Error during YOLO inference (CCTV): {e}")

                            zone = _camera_zones.get(camera_id)
                            if zone and detections:
                                h, w = frame.shape[:2]
                                intruders = check_zone_intrusion(detections, zone, w, h)
                                if intruders:
                                    save_intrusion_snapshot(frame, camera_id)
                            if zone:
                                frame = draw_zone_overlay(frame, zone)

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
                            img, _ = run_detection(model, img)
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
                            
                            detections = []
                            if model:
                                try:
                                    frame, detections = run_detection(model, frame)
                                except Exception as e:
                                    print(f"Error during YOLO inference (USB): {e}")

                            zone = _camera_zones.get(camera_id)
                            if zone and detections:
                                h, w = frame.shape[:2]
                                intruders = check_zone_intrusion(detections, zone, w, h)
                                if intruders:
                                    save_intrusion_snapshot(frame, camera_id)
                            if zone:
                                frame = draw_zone_overlay(frame, zone)

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
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue

                msg_type = msg.get("type")
                if msg_type == "set_zone":
                    points = msg.get("points", [])
                    if points and len(points) >= 3:
                        _camera_zones[camera_id] = points
                        print(f"[VisionGuard] Zone set for camera {camera_id}: {len(points)} points")
                        try:
                            await websocket.send_text(json.dumps({
                                "type": "zone_set_ack",
                                "camera_id": camera_id,
                                "points": points,
                            }))
                        except Exception:
                            pass
                elif msg_type == "clear_zone":
                    _camera_zones.pop(camera_id, None)
                    _zone_snapshot_cooldowns.pop(camera_id, None)
                    print(f"[VisionGuard] Zone cleared for camera {camera_id}")
                    try:
                        await websocket.send_text(json.dumps({
                            "type": "zone_clear_ack",
                            "camera_id": camera_id,
                        }))
                    except Exception:
                        pass
        except WebSocketDisconnect:
            pass

    receiver_task = asyncio.create_task(receive_messages())

    try:
        model = get_yolo_model()
        
        if is_cctv:
            backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
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
                        
                        detections = []
                        if model:
                            try:
                                frame, detections = run_detection(model, frame)
                            except Exception:
                                pass

                        zone = _camera_zones.get(camera_id)
                        if zone and detections:
                            h, w = frame.shape[:2]
                            intruders = check_zone_intrusion(detections, zone, w, h)
                            if intruders:
                                saved = save_intrusion_snapshot(frame, camera_id)
                                if saved:
                                    try:
                                        await websocket.send_text(json.dumps({
                                            "type": "zone_alert",
                                            "camera_id": camera_id,
                                            "intruders": [
                                                {"label": i["label"], "confidence": round(i["confidence"], 2)}
                                                for i in intruders
                                            ],
                                            "snapshot_path": saved,
                                            "timestamp": datetime.now().isoformat(),
                                        }))
                                    except Exception:
                                        pass
                        if zone:
                            frame = draw_zone_overlay(frame, zone)
                        
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
                cv2.putText(img, "VisionGuard WS Feed", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                cv2.putText(img, "SIMULATED LIVE WS", (50, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (14, 165, 233), 2)
                cv2.putText(img, t_str, (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (34, 197, 94), 2)
                
                if model:
                    try:
                        img, _ = run_detection(model, img)
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
                
                detections = []
                if model:
                    try:
                        frame, detections = run_detection(model, frame)
                    except Exception:
                        pass

                zone = _camera_zones.get(camera_id)
                if zone and detections:
                    h, w = frame.shape[:2]
                    intruders = check_zone_intrusion(detections, zone, w, h)
                    if intruders:
                        saved = save_intrusion_snapshot(frame, camera_id)
                        if saved:
                            try:
                                await websocket.send_text(json.dumps({
                                    "type": "zone_alert",
                                    "camera_id": camera_id,
                                    "intruders": [
                                        {"label": i["label"], "confidence": round(i["confidence"], 2)}
                                        for i in intruders
                                    ],
                                    "snapshot_path": saved,
                                    "timestamp": datetime.now().isoformat(),
                                }))
                            except Exception:
                                pass
                if zone:
                    frame = draw_zone_overlay(frame, zone)
                
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
