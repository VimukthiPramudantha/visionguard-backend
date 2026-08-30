# app/api/routers/auth_router.py
from fastapi import APIRouter, HTTPException, status
from app.schemas.user import UserCreate, UserLogin, UserResponse
from app.core.supabase import supabase
import bcrypt
from datetime import datetime

router = APIRouter()

def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    try:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except Exception:
        return False

@router.post("/register", response_model=UserResponse)
async def register(user: UserCreate):
    
    existing = supabase.table("users").select("id").eq("email", user.email.lower()).execute()
    
    if existing.data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    hashed_password = hash_password(user.password)

    new_user = {
        "full_name": user.full_name.strip(),
        "email": user.email.lower(),
        "password_hash": hashed_password,
        "role": user.role or "user",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }

    response = supabase.table("users").insert(new_user).execute()

    if not response.data:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )

    created_user = response.data[0]
    return {**created_user, "password_hash": None}


from app.core.security import create_access_token

@router.post("/login")
async def login(user: UserLogin):
    response = supabase.table("users").select("*").eq("email", user.email.lower()).execute()
    
    if not response.data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )

    db_user = response.data[0]

    if not verify_password(user.password, db_user.get("password_hash", "")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )

    access_token = create_access_token(data={"sub": db_user["id"], "role": db_user.get("role", "user")})

    return {
        "message": "Login successful",
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": db_user["id"],
            "email": db_user["email"],
            "full_name": db_user["full_name"],
            "role": db_user["role"]
        }
    }


@router.get("/me")
async def get_current_user(email: str):
    response = (
        supabase.table("users")
        .select("id, email, full_name, role, created_at, updated_at")
        .eq("email", email.lower())
        .execute()
    )
    
    if not response.data:
        raise HTTPException(status_code=404, detail="User not found")
        
    user = response.data[0]
    return {**user, "password_hash": None}


@router.put("/me")
async def update_current_user(email: str, full_name: str):
    response = supabase.table("users").update({
        "full_name": full_name.strip(),
        "updated_at": datetime.utcnow().isoformat()
    }).eq("email", email.lower()).execute()
    
    if not response.data:
        raise HTTPException(status_code=404, detail="User not found")
        
    user = response.data[0]
    return {**user, "password_hash": None}


from app.api.routers.camera.state import GLOBAL_SETTINGS, update_detection_settings

@router.get("/settings")
async def get_settings():
    return GLOBAL_SETTINGS


@router.put("/settings")
async def update_settings(settings: dict):
    update_detection_settings(settings)
    return GLOBAL_SETTINGS