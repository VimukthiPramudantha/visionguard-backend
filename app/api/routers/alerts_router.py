# app/api/routers/alerts_router.py
from fastapi import APIRouter, HTTPException, Depends
from typing import List
from app.core.db_service import get_alerts_for_user, mark_alert_read, get_unread_alert_count, delete_alert
from app.core.security import get_current_user_id
from app.schemas.alert import AlertResponse

router = APIRouter()


@router.get("", response_model=List[AlertResponse])
async def get_alerts(user_id: str = Depends(get_current_user_id)):
    alerts = get_alerts_for_user(user_id)
    return alerts


@router.get("/unread-count")
async def get_unread_count(user_id: str = Depends(get_current_user_id)):
    count = get_unread_alert_count(user_id)
    return {"unread_count": count}


@router.patch("/{alert_id}/read")
async def mark_read(alert_id: str, user_id: str = Depends(get_current_user_id)):
    result = mark_alert_read(alert_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"status": "ok", "alert": result}


@router.delete("/{alert_id}")
async def delete_alert_record(alert_id: str, user_id: str = Depends(get_current_user_id)):
    success = delete_alert(alert_id)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found or failed to delete")
    return {"status": "ok", "message": "Alert deleted successfully"}
