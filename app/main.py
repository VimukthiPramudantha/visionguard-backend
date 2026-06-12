from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import sys

# Ensure backend root is in python path to resolve 'api' module
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from api.routes.camera_routes import router as camera_router

app = FastAPI(title="VisionGuard")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(camera_router)

# Set up templates directory
templates_dir = ROOT / "app" / "templates"
templates = Jinja2Templates(directory=str(templates_dir) if templates_dir.exists() else "app/templates")

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    template_path = Path("app/templates/camera_dashboard.html")
    if not template_path.exists():
        return HTMLResponse("<h1>VisionGuard Camera Dashboard</h1><p>Template not found.</p>")
    return templates.TemplateResponse("camera_dashboard.html", {"request": {}})