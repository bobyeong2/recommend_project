'''
해당 파일에서 API router를 정의 -> main.py에서 사용

<예시>

from fastapi import APIRouter

api_router = APIRouter()
api_router.include_router(route_users.router, prefix="/users", tags=["users"])

'''
