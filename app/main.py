import logging #260324 추가
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.logging_config import setup_logging  #260324 추가
from app.api.v1.api import api_router
from app.ml.inference.predictor import MovieRecommender
from app.core.redis_client import redis_client
from app.middleware.request_logging import RequestLoggingMiddleware  #260324 추가

from prometheus_client import generate_latest, CONTENT_TYPE_LATEST #260325 추가
from starlette.responses import Response
setup_logging(log_level="DEBUG" if settings.DEBUG else "INFO")
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    앱 시작/종료 시 실행
    """
    # 시작: 모델 로드
    logger.info("=" * 70)
    logger.info(f"{settings.PROJECT_NAME} 시작 중...")
    logger.info("=" * 70)
    
    # 모델 pre-load (첫 요청 시 느린 것 방지)
    MovieRecommender()
    
    logger.info("=" * 70)
    logger.info("서비스 준비 완료!")
    logger.info("=" * 70)
    
    # Redis 연결
    logger.info("Redis 연결 중...")
    try:
        await redis_client.connect(settings.REDIS_URL)
        logger.info("Redis connected")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e} (service will run without cache)")
        
    logger.info("=" * 70)
    logger.info(" 서비스 준비 완료!")
    logger.info("=" * 70)
    
    yield
    
    # ==================== 종료 ====================
    logger.info("=" * 70)
    logger.info("서비스 종료 중...")
    logger.info("=" * 70)
    
    # Redis 연결 종료
    logger.info("🔌 Redis 연결 종료 중...")
    await redis_client.disconnect()
    
    logger.info("✅ 서비스 종료 완료")


# FastAPI 앱 생성
app = FastAPI(
    title=settings.PROJECT_NAME,
    version="2.2.0",
    description="Neural Collaborative Filtering 기반 영화 추천 서비스",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

"""
요청 → [CORS (바깥)] → [로깅 (안쪽)] → 핸들러
응답 ← [CORS (바깥)] ← [로깅 (안쪽)] ← 핸들러
"""
# 미들웨어 등록 (순서 중요 : 먼저 등록된 것이 안쪽)
app.add_middleware(RequestLoggingMiddleware)
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
        "version": "2.2.0"
    }


@app.get("/health")
async def health_check():
    """상세 헬스 체크"""
    # Redis 연결 상태 확인
    redis_connected = redis_client.redis is not None
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "database": "connected",
        "redis": "connected" if redis_connected else "disconnected"
    }
    
@app.get("/metrics")
async def prometheus_metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )