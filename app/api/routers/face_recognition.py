# app/api/routers/face_recognition.py
from fastapi import APIRouter, UploadFile, File, HTTPException, status, Form
from fastapi.responses import JSONResponse
from deepface import DeepFace
from app.core.supabase import supabase
import tempfile
import os
import uuid
import shutil
from typing import Dict, List

router = APIRouter(prefix="/face", tags=["face-recognition"])

STORAGE_DIR = "app/static/registered_faces"

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
            detector_backend="opencv",
            enforce_detection=True,
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
    Get the list of registered faces from Supabase
    """
    try:
        response = supabase.table("registered_faces").select("*").execute()
        return response.data or []
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
    Register a new face to the Supabase database
    """
    # Validation
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

    # Make sure storage dir exists
    os.makedirs(STORAGE_DIR, exist_ok=True)

    # Generate a unique name to prevent directory traversal or collisions
    file_ext = image.filename.split(".")[-1] if "." in image.filename else "jpg"
    if file_ext.lower() not in ["jpg", "jpeg", "png"]:
        file_ext = "jpg"
        
    unique_filename = f"{uuid.uuid4()}.{file_ext}"
    dest_path = os.path.join(STORAGE_DIR, unique_filename)

    try:
        # Save image locally on backend for DeepFace verify
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Update Supabase DB
        face_id = str(uuid.uuid4())
        
        # We store relative URL path for the frontend (which is mounted on /static)
        # and absolute local path for the backend (DeepFace verify)
        new_face = {
            "id": face_id,
            "name": name.strip(),
            "image_url": f"/static/registered_faces/{unique_filename}",
            "file_path": dest_path
        }
        
        response = supabase.table("registered_faces").insert(new_face).execute()
        if not response.data:
            raise Exception("Failed to insert record into Supabase")

        return {"success": True, "message": "Face registered successfully!", "face": new_face}
    except Exception as e:
        if os.path.exists(dest_path):
            os.unlink(dest_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register face: {str(e)}"
        )


@router.post("/identify")
async def identify_face(
    image: UploadFile = File(...)
):
    """
    Identify a face against all registered faces in the Supabase database
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )

    try:
        response = supabase.table("registered_faces").select("*").execute()
        db = response.data or []
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

    # Save incoming image to temporary file
    temp_target_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(await image.read())
            temp_target_path = tmp.name

        best_match = None
        highest_similarity = -1.0

        # Iterate over all registered faces
        for person in db:
            registered_img_path = person.get("file_path")
            if not registered_img_path or not os.path.exists(registered_img_path):
                continue

            try:
                result = DeepFace.verify(
                    img1_path=temp_target_path,
                    img2_path=registered_img_path,
                    model_name="VGG-Face",
                    detector_backend="opencv",
                    enforce_detection=True,
                    distance_metric="cosine"
                )
                
                similarity = round((1 - result["distance"]) * 100, 2)
                
                # Check match based on verification threshold
                if result["verified"] and similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = person
            except Exception:
                # If face is not detected in one of the images, skip that check
                continue

        os.unlink(temp_target_path)

        if best_match:
            return {
                "match": True,
                "name": best_match["name"],
                "similarity_percentage": highest_similarity,
                "message": f"Match found: {best_match['name']} with {highest_similarity}% similarity"
            }
        else:
            return {
                "match": False,
                "message": "No match found in the database. Face does not match any registered user."
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
    Delete a registered face by ID from database and disk
    """
    try:
        # Get face details first to delete the file
        response = supabase.table("registered_faces").select("*").eq("id", face_id).execute()
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Registered face not found"
            )
        
        person = response.data[0]
        file_path = person.get("file_path")
        if file_path and os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except Exception:
                pass
                
        # Delete from Supabase
        supabase.table("registered_faces").delete().eq("id", face_id).execute()
        return {"success": True, "message": "Face deleted successfully!"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete face: {str(e)}"
        )