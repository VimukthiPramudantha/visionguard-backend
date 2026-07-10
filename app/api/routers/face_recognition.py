# app/api/routers/face_recognition.py
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from deepface import DeepFace
import tempfile
import os
from typing import Dict

router = APIRouter(prefix="/face", tags=["face-recognition"])

@router.post("/compare")
async def compare_faces(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...)
) -> Dict:
    """
    Compare two faces using DeepFace
    """
    if not image1.content_type.startswith("image/") or not image2.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Both files must be images"
        )

    try:
        # Save uploaded files temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp1:
            tmp1.write(await image1.read())
            img1_path = tmp1.name

        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp2:
            tmp2.write(await image2.read())
            img2_path = tmp2.name

        # Perform face comparison using DeepFace
        result = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name="VGG-Face",      # Good balance of accuracy and speed
            detector_backend="opencv",
            enforce_detection=True,
            distance_metric="cosine"
        )

        similarity = round((1 - result["distance"]) * 100, 2)
        is_match = result["verified"]

        # Clean up temp files
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
        # Clean up in case of error
        for path in [img1_path, img2_path]:
            if os.path.exists(path):
                os.unlink(path)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Face comparison failed: {str(e)}"
        )