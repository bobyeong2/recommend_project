import pytest
import pytest_asyncio
from app.core.database import engine

@pytest_asyncio.fixture(scope="function", autouse=True)
async def cleanup_engine():
    """각 테스트 후 DB engine dispose"""
    yield
    await engine.dispose()
