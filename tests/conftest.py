# tests/conftest.py
import pytest
import pytest_asyncio
import os
from datetime import date
from sqlalchemy import event
from app.core.database import engine
from app.models.user import User
from app.models.user_rating import UserRating

@pytest_asyncio.fixture(scope="session", autouse=True)
def setup_test_env():
    """테스트 환경변수 설정"""
    os.environ["TESTING"] = "true"
    os.environ["DATABASE_URL"] = "mysql+aiomysql://root:rootpass@localhost:13306/bob_movie_db_test"

@pytest_asyncio.fixture(scope="function", autouse=True)
async def cleanup_engine():
    """각 테스트 후 DB engine dispose"""
    from app.ml.inference.predictor import MovieRecommender
    MovieRecommender._instance = None
    MovieRecommender._initialized = False
    
    def set_user_dates(mapper, connection, target):
        if target.created_at is None:
            target.created_at = date.today()
        if target.updated_at is None:
            target.updated_at = date.today()
    
    def set_rating_dates(mapper, connection, target):
        if target.created_at is None:
            target.created_at = date.today()
        if target.updated_at is None:
            target.updated_at = date.today()
    
    event.listen(User, "before_insert", set_user_dates)
    event.listen(UserRating, "before_insert", set_rating_dates)
    
    yield
    
    event.remove(User, "before_insert", set_user_dates)
    event.remove(UserRating, "before_insert", set_rating_dates)
    await engine.dispose()