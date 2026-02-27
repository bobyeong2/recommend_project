# create_tables.py

import asyncio 
from app.core.database import engine, Base

# 모든 모델 import
from app.models.movie import Movie
from app.models.user import User
from app.models.training import TrainingUser, TrainingRating
from app.models.rating import UserRating

async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print(" All tables created!")
    print("   - movies")
    print("   - users")
    print("   - training_users")
    print("   - training_ratings")
    print("   - user_ratings")

async def main():
    await create_tables()
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())