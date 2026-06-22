# main.py
from fastapi import FastAPI
from app.api.routers.auth_router import router as auth_router
from app.middleware import init_middleware

try:
    from app.api.routers.camera_routes import router as camera_router
except ImportError:
    try:
        from app.api.camera_routes import router as camera_router
    except ImportError:
        camera_router = None
        print("camera_router not found - skipping for now")

app = FastAPI(title="VisionGuard")

init_middleware(app)

app.include_router(auth_router, prefix="/auth", tags=["auth"])

if camera_router:
    app.include_router(camera_router, tags=["camera"])
else:
    print("Camera routes skipped (will add later)")

@app.get("/")
async def root():
    return {"message": "VisionGuard API is running successfully!"}

print("VisionGuard Backend started successfully!")