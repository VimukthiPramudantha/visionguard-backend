# app/api/routers/alerts_router.py
from fastapi import APIRouter, HTTPException, Query
from typing import List
from app.core.db_service import get_alerts_for_user, mark_alert_read, get_unread_alert_count
from app.schemas.alert import AlertResponse

router = APIRouter()


@router.get("", response_model=List[AlertResponse])
async def get_alerts(user_id: str = Query(..., description="The user ID to fetch alerts for")):
    """Get all alerts for a user, sorted newest first."""
    alerts = get_alerts_for_user(user_id)
    return alerts


@router.get("/unread-count")
async def get_unread_count(user_id: str = Query(..., description="The user ID to count unread alerts for")):
    """Get the number of unread alerts for a user."""
    count = get_unread_alert_count(user_id)
    return {"unread_count": count}


@router.patch("/{alert_id}/read")
async def mark_read(alert_id: str):
    """Mark a single alert as read."""
    result = mark_alert_read(alert_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"status": "ok", "alert": result}
