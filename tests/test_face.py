# tests/test_face.py
import os
import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app.main import app
import json

client = TestClient(app)

def test_face_endpoints():
    print("Starting face recognition API tests...")

    response = client.get("/face/registered")
    assert response.status_code == 200
    initial_data = response.json()
    print(f"GET /face/registered returned {len(initial_data)} items")

    response = client.post("/face/register", data={"name": "Test User"})
    assert response.status_code == 422
    print("Validation with missing file passed (returned 422)")

    dummy_img_path = "test_temp_face.jpg"
    with open(dummy_img_path, "wb") as f:
        f.write(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9')

    try:
        with open(dummy_img_path, "rb") as img:
            response = client.post(
                "/face/register",
                data={"name": "Alice Cooper"},
                files={"image": ("test_temp_face.jpg", img, "image/jpeg")}
            )
        assert response.status_code == 200
        res_data = response.json()
        assert res_data["success"] is True
        assert res_data["face"]["name"] == "Alice Cooper"
        face_id = res_data["face"]["id"]
        print("POST /face/register Alice Cooper: Successful")

        response = client.get("/face/registered")
        db_data = response.json()
        assert any(item["id"] == face_id for item in db_data)
        print("GET /face/registered includes registered face: Successful")

        response = client.delete(f"/face/registered/{face_id}")
        assert response.status_code == 200
        print("DELETE /face/registered: Successful")

        response = client.get("/face/registered")
        final_data = response.json()
        assert not any(item["id"] == face_id for item in final_data)
        print("Verified clean state after deletion: Successful")

    finally:
        if os.path.exists(dummy_img_path):
            os.unlink(dummy_img_path)

    print("All Face Recognition API unit tests passed successfully!")

if __name__ == "__main__":
    test_face_endpoints()
