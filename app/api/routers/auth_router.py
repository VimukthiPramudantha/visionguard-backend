# app/api/routers/auth_router.py
from fastapi import APIRouter, HTTPException, Depends, status
from app.schemas.user import UserCreate, UserLogin, UserResponse
from app.core.supabase import supabase
from passlib.context import CryptContext
from datetime import datetime

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

@router.post("/register", response_model=UserResponse)
async def register(user: UserCreate):
    # Check if user exists
    existing = supabase.table("users").select("id").eq("email", user.email).execute()
    if existing.data:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = hash_password(user.password)

    new_user = {
        "full_name": user.full_name,
        "email": user.email.lower(),
        "password_hash": hashed_password,
        "role": user.role,
    }

    response = supabase.table("users").insert(new_user).execute()

    if not response.data:
        raise HTTPException(status_code=500, detail="Failed to create user")

    created = response.data[0]
    return {**created, "password_hash": None}


@router.post("/login")
async def login(user: UserLogin):
    response = supabase.table("users").select("*").eq("email", user.email).execute()
    
    if not response.data:
        raise HTTPException(status_code=400, detail="Invalid credentials")

    db_user = response.data[0]

    if not verify_password(user.password, db_user["password_hash"]):
        raise HTTPException(status_code=400, detail="Invalid credentials")

    # Return user data + token (we'll improve with JWT later)
    return {
        "message": "Login successful",
        "user": {
            "id": db_user["id"],
            "email": db_user["email"],
            "full_name": db_user["full_name"],
            "role": db_user["role"]
        }
    }


@router.get("/me")
async def get_current_user(email: str):  # Temporary - will use token later
    response = supabase.table("users").select("*").eq("email", email).execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="User not found")
    user = response.data[0]
    return {**user, "password_hash": None}