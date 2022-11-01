'''
API router를 include하는 공간
API router들을 정의하고 main.py에서 import 함

<예시>
from fastapi import APIRouter

api_router = APIRouter()
api_router.include_router(route_jobs.router, prefix="", tags=[""])
api_router. ~~~~~~ etc..

'''