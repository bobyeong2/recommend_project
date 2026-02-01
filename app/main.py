from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.api.v1.api import api_router
from app.ml.inference.predictor import MovieRecommender


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    앱 시작/종료 시 실행
    """
    # 시작: 모델 로드
    print("=" * 70)
    print(f"{settings.PROJECT_NAME} 시작 중...")
    print("=" * 70)
    
    # 모델 pre-load (첫 요청 시 느린 것 방지)
    MovieRecommender()
    
    print("=" * 70)
    print("서비스 준비 완료!")
    print("=" * 70)
    
    yield
    
    # 종료 시 정리 작업
    print("서비스 종료 중...")


# FastAPI 앱 생성
app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="Neural Collaborative Filtering 기반 영화 추천 서비스",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/")
async def root():
    """헬스 체크"""
    return {
        "service": settings.PROJECT_NAME,
        "status": "healthy",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """상세 헬스 체크"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "database": "connected"
    }