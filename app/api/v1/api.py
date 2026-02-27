from fastapi import APIRouter

from app.api.v1.endpoints import recommendations
from app.api.v1.endpoints import movies
from app.api.v1.endpoints import auth
from app.api.v1.endpoints import ratings
from app.api.v1.endpoints import users
api_router = APIRouter()

api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["auth"]
)

api_router.include_router(
    users.router,
    prefix="/users",
    tags=["users"]
)

api_router.include_router(
    ratings.router,
    prefix="/ratings",
    tags=["ratings"]
)

api_router.include_router(
    recommendations.router,
    prefix="/recommendations",
    tags=["recommendations"]
)


api_router.include_router(
    movies.router,
    prefix="/movies",
    tags=["movies"]
)

