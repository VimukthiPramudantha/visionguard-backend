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

def encrypt_data(data: str) -> str:
    """Encrypt a string to base64 encrypted format."""
    if not data:
        return data
    return cipher_suite.encrypt(data.encode('utf-8')).decode('utf-8')

def decrypt_data(token: str) -> str:
    """Decrypt a base64 encrypted token back to string, fallback to original if not encrypted."""
    if not token:
        return token
    try:
        return cipher_suite.decrypt(token.encode('utf-8')).decode('utf-8')
    except Exception:
        return token

def encrypt_embedding(embedding: list) -> dict:
    """Serialize and encrypt the embedding vector list, returning a JSON-safe dict."""
    serialized = json.dumps(embedding)
    encrypted_str = encrypt_data(serialized)
    return {"encrypted_data": encrypted_str}

def decrypt_embedding(encrypted_obj) -> list:
    """Decrypt and deserialize the embedding vector back to list of floats."""
    if isinstance(encrypted_obj, dict) and "encrypted_data" in encrypted_obj:
        decrypted_str = decrypt_data(encrypted_obj["encrypted_data"])
        try:
            return json.loads(decrypted_str)
        except Exception:
            pass
    return encrypted_obj
