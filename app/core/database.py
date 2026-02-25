# app/core/database.py

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from .config import settings

# 비동기 엔진 생성

engine = create_async_engine(
    settings.DATABASE_URL, #.env database url
    echo=settings.DEBUG, # sql query log
    pool_pre_ping=True, # connect check
)


AsyncSessionLocal = sessionmaker(
    engine,
    class_ = AsyncSession,
    expire_on_commit=False,
)

Base = declarative_base()

async def get_db():
    async with AsyncSessionLocal() as session :
        yield session
        
        