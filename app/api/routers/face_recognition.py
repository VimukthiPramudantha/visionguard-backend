# app/api/routers/face_recognition.py
from fastapi import APIRouter, UploadFile, File, HTTPException, status, Form
from fastapi.responses import JSONResponse
from deepface import DeepFace
from app.core.supabase import supabase
from app.core.security import encrypt_data, decrypt_data, encrypt_embedding, decrypt_embedding
from app.core.redis import get_cache, set_cache, delete_cache
import tempfile
import os
import uuid
import numpy as np
from typing import Dict, List

router = APIRouter(prefix="/face", tags=["face-recognition"])

def calculate_cosine_distance(v1: List[float], v2: List[float]) -> float:
    """Calculate the cosine distance between two embedding vectors"""
    a = np.array(v1)
    b = np.array(v2)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    cosine_similarity = dot_product / (norm_a * norm_b)
    return float(1.0 - cosine_similarity)

@router.post("/compare")
async def compare_faces(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...)
) -> Dict:
    """
    Compare two faces using DeepFace (1-to-1 comparison)
    """
    if not image1.content_type.startswith("image/") or not image2.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Both files must be images"
        )

    img1_path = ""
    img2_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp1:
            tmp1.write(await image1.read())
            img1_path = tmp1.name

        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp2:
            tmp2.write(await image2.read())
            img2_path = tmp2.name

        result = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name="VGG-Face",      
            detector_backend="retinaface",
            enforce_detection=False,
            distance_metric="cosine"
        )

        similarity = round((1 - result["distance"]) * 100, 2)
        is_match = result["verified"]

        os.unlink(img1_path)
        os.unlink(img2_path)

        return {
            "success": True,
            "similarity_percentage": similarity,
            "match": is_match,
            "distance": round(result["distance"], 4),
            "message": "Faces match successfully!" if is_match else "Faces do not match.",
            "model_used": "VGG-Face"
        }

    except Exception as e:
        if img1_path and os.path.exists(img1_path):
            os.unlink(img1_path)
        if img2_path and os.path.exists(img2_path):
            os.unlink(img2_path)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Face comparison failed: {str(e)}"
        )


@router.get("/registered")
async def get_registered_faces():
    """
    Get the list of registered faces from Supabase (excluding the raw embeddings for speed)
    """
    try:
        cache_key = "registered_faces:without_embeddings"
        cached_data = await get_cache(cache_key)
        if cached_data is not None:
            return cached_data

        response = supabase.table("registered_faces").select("id, name, created_at").execute()
        data = response.data or []
        for face in data:
            if "name" in face:
                face["name"] = decrypt_data(face["name"])
        
        await set_cache(cache_key, data, expire_seconds=3600)
        return data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch registered faces: {str(e)}"
        )


@router.post("/register")
async def register_face(
    name: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Register a face by extracting and storing its facial embedding vector,
    without saving the raw image file.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )

    if not name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Name cannot be empty"
        )

    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(await image.read())
            temp_path = tmp.name

        representations = DeepFace.represent(
            img_path=temp_path,
            model_name="VGG-Face",
            detector_backend="retinaface",
            enforce_detection=False
        )
        
        if not representations or len(representations) == 0:
            raise ValueError("No face detected in the image.")

        embedding = representations[0]["embedding"]

        os.unlink(temp_path)
        temp_path = ""

        face_id = str(uuid.uuid4())
        new_face = {
            "id": face_id,
            "name": encrypt_data(name.strip()),
            "embedding": encrypt_embedding(embedding) 
        }
        
        response = supabase.table("registered_faces").insert(new_face).execute()
        if not response.data:
            raise Exception("Failed to insert record into Supabase")

        # Invalidate cache
        await delete_cache("registered_faces:without_embeddings")
        await delete_cache("registered_faces:with_embeddings")

        return {
            "success": True, 
            "message": "Face embedding registered successfully! No image files were saved.", 
            "face": {"id": face_id, "name": name.strip()}
        }
    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register face vector: {str(e)}"
        )


@router.post("/identify")
async def identify_face(
    image: UploadFile = File(...)
):
    """
    Identify a face by comparing its embedding vector mathematically
    against the stored database vectors.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )

    try:
        cache_key = "registered_faces:with_embeddings"
        db = await get_cache(cache_key)
        if db is None:
            response = supabase.table("registered_faces").select("id, name, embedding").execute()
            db = response.data or []
            await set_cache(cache_key, db, expire_seconds=3600)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database query failed: {str(e)}"
        )

    if not db:
        return {
            "match": False,
            "message": "No registered faces found in the database. Please add some faces first."
        }

    temp_target_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(await image.read())
            temp_target_path = tmp.name

        representations = DeepFace.represent(
            img_path=temp_target_path,
            model_name="VGG-Face",
            detector_backend="retinaface",
            enforce_detection=False
        )
        
        if not representations or len(representations) == 0:
            raise ValueError("No face detected in the image.")

        target_embedding = representations[0]["embedding"]

        os.unlink(temp_target_path)
        temp_target_path = ""

        best_match = None
        min_distance = 1.0  
        MATCH_THRESHOLD = 0.40

        for person in db:
            stored_embedding = person.get("embedding")
            if not stored_embedding:
                continue

            try:
                decrypted_embedding = decrypt_embedding(stored_embedding)
                distance = calculate_cosine_distance(target_embedding, decrypted_embedding)
                if distance < min_distance:
                    min_distance = distance
                    best_match = person
            except Exception:
                continue

        if best_match and min_distance <= MATCH_THRESHOLD:
            decrypted_name = decrypt_data(best_match["name"])
            similarity = round((1.0 - min_distance) * 100, 2)
            return {
                "match": True,
                "name": decrypted_name,
                "similarity_percentage": similarity,
                "message": f"Match found: {decrypted_name} with {similarity}% confidence"
            }
        else:
            return {
                "match": False,
                "message": "No match found. Face does not match any registered profiles."
            }

    except Exception as e:
        if temp_target_path and os.path.exists(temp_target_path):
            os.unlink(temp_target_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Face identification failed: {str(e)}"
        )


@router.delete("/registered/{face_id}")
async def delete_registered_face(face_id: str):
    """
    Delete a registered face vector profile by ID from the database
    """
    try:
        response = supabase.table("registered_faces").delete().eq("id", face_id).execute()
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Registered face not found"
            )
        
        # Invalidate cache
        await delete_cache("registered_faces:without_embeddings")
        await delete_cache("registered_faces:with_embeddings")

        return {"success": True, "message": "Face profile deleted successfully!"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete face: {str(e)}"
        )