import pytest
import pytest_asyncio
from datetime import date
from sqlalchemy import event
from app.core.database import engine
from app.models.user import User
from app.models.user_rating import UserRating

# created_at, updated_at 자동 설정
@event.listens_for(User, 'before_insert')
def set_user_timestamps(mapper, connection, target):
    if target.created_at is None:
        target.created_at = date.today()
    if target.updated_at is None:
        target.updated_at = date.today()

@event.listens_for(UserRating, 'before_insert')
def set_rating_timestamps(mapper, connection, target):
    if target.created_at is None:
        target.created_at = date.today()
    if target.updated_at is None:
        target.updated_at = date.today()

@pytest_asyncio.fixture(scope="function", autouse=True)
async def cleanup_engine():
    """각 테스트 후 DB engine dispose"""
    yield
    await engine.dispose()