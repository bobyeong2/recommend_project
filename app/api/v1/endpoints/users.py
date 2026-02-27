from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.user import User
from app.schemas.user import UserResponse, UserUpdate
from app.api.dependencies import get_current_user

router = APIRouter()

@router.get("/me", response_model=UserResponse)
async def get_my_info(
    current_user: User = Depends(get_current_user)
):
    """내 정보 조회"""
    return current_user

@router.put("/me", response_model=UserResponse)
async def update_my_info(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """내 정보 수정"""
    if user_update.username:
        current_user.username = user_update.username
    
    await db.commit()
    await db.refresh(current_user)
    return current_user