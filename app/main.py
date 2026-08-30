# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.api.routers.auth_router import router as auth_router
from app.api.routers.face_recognition import router as face_router

try:
    from app.api.routers.camera import router as camera_router
except ImportError:
    try:
        from app.api.routers.camera_routes import router as camera_router
    except ImportError:
        try:
            from app.api.camera_routes import router as camera_router
        except ImportError:
            camera_router = None
            print("camera_router not found - skipping for now")

from app.api.routers.alerts_router import router as alerts_router

app = FastAPI(title="VisionGuard")

from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(face_router)
app.include_router(alerts_router, prefix="/alerts", tags=["alerts"])

if camera_router:
    app.include_router(camera_router, tags=["camera"])
else:
    print("Camera routes skipped (will add later)")

@app.get("/")
async def root():
    return {"message": "VisionGuard API is running successfully!"}

@app.on_event("startup")
async def startup_event():
    try:
        from app.core.db_service import sync_hardcoded_cameras
        sync_hardcoded_cameras()
    except Exception as e:
        print(f"[VisionGuard] Startup camera sync failed: {e}")

print("VisionGuard Backend started successfully!")