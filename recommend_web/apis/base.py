'''
해당 파일에서 API router를 정의 -> main.py에서 사용

<예시>

from fastapi import APIRouter

api_router = APIRouter()
api_router.include_router(route_users.router, prefix="/users", tags=["users"])

'''

from apis.version1 import (
    route_users
    )
from fastapi import APIRouter

from apis.version1 import route_recommend

api_router = APIRouter()

api_router.include_router(route_users.router, prefix="/users", tags=["users"])
api_router.include_router(route_recommend.router, prefix="/recommend",tags=["items"])
# api_router.include_router(route_recommend.router, prefix="/recommend", tags=["movies"])