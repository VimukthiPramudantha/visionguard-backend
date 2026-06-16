# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Import routers
from app.api.routers.auth_router import router as auth_router
from app.api.routers.camera_routes import router as camera_router

app = FastAPI(title="VisionGuard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(camera_router, tags=["camera"])   # kept as requested

@app.get("/")
async def root():
    return {"message": "VisionGuard API is running 🚀"}