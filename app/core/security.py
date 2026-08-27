# app/core/security.py
import os
import json
from cryptography.fernet import Fernet
from dotenv import load_dotenv

load_dotenv()

ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")

if not ENCRYPTION_KEY:
    key = Fernet.generate_key().decode()
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env_path = os.path.join(base_dir, ".env")
    try:
        with open(env_path, "a") as f:
            f.write(f"\nENCRYPTION_KEY={key}\n")
        ENCRYPTION_KEY = key
    except Exception as e:
        print("Warning: Could not save ENCRYPTION_KEY to .env file:", e)
        ENCRYPTION_KEY = key

cipher_suite = Fernet(ENCRYPTION_KEY.encode())

import jwt
from datetime import datetime as dt_datetime, timedelta, timezone as dt_timezone
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

JWT_SECRET = os.getenv("JWT_SECRET")

if not JWT_SECRET:
    import secrets
    key = secrets.token_hex(32)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env_path = os.path.join(base_dir, ".env")
    try:
        with open(env_path, "a") as f:
            f.write(f"\nJWT_SECRET={key}\n")
        JWT_SECRET = key
        print("[VisionGuard] Generated new secure JWT_SECRET and saved to .env")
    except Exception as e:
        print("Warning: Could not save JWT_SECRET to .env file:", e)
        JWT_SECRET = "fallback-visionguard-secret-super-key-change-in-prod"

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)

def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = dt_datetime.now(dt_timezone.utc) + expires_delta
    else:
        expire = dt_datetime.now(dt_timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> dict | None:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        return payload
    except jwt.PyJWTError:
        return None

def get_current_user_id(token: str = Depends(oauth2_scheme)) -> str:
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token missing",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_access_token(token)
    if not payload or "sub" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload["sub"]

def encrypt_data(data: str) -> str:
    if not data:
        return data
    return cipher_suite.encrypt(data.encode('utf-8')).decode('utf-8')

def decrypt_data(token: str) -> str:
    if not token:
        return token
    try:
        return cipher_suite.decrypt(token.encode('utf-8')).decode('utf-8')
    except Exception:
        return token

def encrypt_embedding(embedding: list) -> dict:
    serialized = json.dumps(embedding)
    encrypted_str = encrypt_data(serialized)
    return {"encrypted_data": encrypted_str}

def decrypt_embedding(encrypted_obj) -> list:
    if isinstance(encrypted_obj, dict) and "encrypted_data" in encrypted_obj:
        decrypted_str = decrypt_data(encrypted_obj["encrypted_data"])
        try:
            return json.loads(decrypted_str)
        except Exception:
            pass
    return encrypted_obj
