from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List

from app.core.database import get_db
from app.models.movie import Movie
from app.models.user_rating import UserRating # 260320추가
from app.ml.inference.predictor import MovieRecommender
from app.schemas.recommendation import (
    PredictionRequest,
    PredictionResponse,
    PredictionItem,
    RecommendationResponse,
    RecommendationItem
)

from app.api.dependencies import get_current_user # 260320추가
from app.models.user import User # 260320추가
from app.core.redis_client import redis_client # 260320추가
import logging # 260320추가
 
logger = logging.getLogger(__name__)

router = APIRouter()

# 전역 recommender (싱글톤)
recommender = None


def get_recommender() -> MovieRecommender:
    """Recommender 인스턴스 가져오기"""
    global recommender
    if recommender is None:
        recommender = MovieRecommender()
    return recommender


@router.get("",response_model=RecommendationResponse)
async def get_my_recommendations(
    top_k: int = Query(10, ge=1, le=100, description="추천 영화 개수"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    
    """
    로그인한 사용자 맞춤 추천 (Redis 캐싱)
    
    - top_k : 추천할 영화 개수 (기본 10개)
    """
    
    user_id = current_user.id
    
    # 캐싱 확인
    cached_data = await redis_client.get_recommendation_cache(user_id)
    if cached_data:
        logger.info(f"캐시에서 추천 결과 반환: user_id={user_id}")
        return RecommendationResponse(**cached_data)
    
    logger.info(f" 새로운 추천 생성 중 : user_id={user_id}")
    
    # 사용자 평점 개수 확인
    result = await db.execute(
        select(UserRating).where(UserRating.user_id == user_id)
    )
    user_ratings = result.scalars().all()
    rating_count = len(user_ratings)
    
    # 모든 영화 ID를 가져오기
    result = await db.execute(select(Movie.id))
    all_movie_ids = [row[0] for row in result.fetchall()]
    
    if not all_movie_ids:
        return HTTPException(status_code=404, detail="영화 데이터가 없습니다.")
    
    # 추천 (임시로 training_user_id 1 사용)
    # 추후 하이브리드 추천 로직 적용
    rec = get_recommender()
    recommendations = rec.recommend(1,all_movie_ids,top_k)
    
    # 영화 정보 조회
    movie_ids = [r['movie_id'] for r in recommendations]
    result = await db.execute(
        select(Movie).where(Movie.id.in_(movie_ids))
    )
    
    movies = {m.id: m for m in result.scalars().all()}
    
    # 결과 조합
    response_items = []
    for rec_item in recommendations:
        movie_id = rec_item['movie_id']
        if movie_id in movies:
            movie = movies[movie_id]
            response_items.append(
                RecommendationItem(
                    movie_id=movie_id,
                    title=movie.title,
                    predicted_rating=round(rec_item["predicted_rating"],2)
                )
            )
    
    # Res 데이터 생성
    response_data = {
        "user_id": user_id,
        "recommendations": [item.model_dump() for item in response_items]
    }
    
    await redis_client.set_recommendation_cache(user_id, response_data, ttl=3600)
    
    logger.info(f"추천 완료 및 캐싱 : user_id={user_id}, count={len(response_items)}")
    
    return RecommendationResponse(**response_data)

@router.get("/{user_id}", response_model=RecommendationResponse)
async def get_recommendations_by_user_id(
    user_id: int,
    top_k: int = Query(10, ge=1, le=100, description="추천 영화 개수"),
    db: AsyncSession = Depends(get_db)
):
    """
    사용자 맞춤 영화 추천
    
    - **user_id**: 사용자 ID (training_user_id)
    - **top_k**: 추천할 영화 개수 (기본 10개)
    """
    
    # 모든 영화 ID 가져오기
    result = await db.execute(select(Movie.id))
    all_movie_ids = [row[0] for row in result.fetchall()]
    
    if not all_movie_ids:
        raise HTTPException(status_code=404, detail="영화 데이터가 없습니다")
    
    # 추천
    rec = get_recommender()
    recommendations = rec.recommend(user_id, all_movie_ids, top_k)
    
    # 영화 정보 조회
    movie_ids = [r['movie_id'] for r in recommendations]
    result = await db.execute(
        select(Movie).where(Movie.id.in_(movie_ids))
    )
    movies = {m.id: m for m in result.scalars().all()}
    
    # 결과 조합
    response_items = []
    for rec_item in recommendations:
        movie_id = rec_item['movie_id']
        if movie_id in movies:
            movie = movies[movie_id]
            response_items.append(
                RecommendationItem(
                    movie_id=movie.id,
                    title=movie.title,
                    predicted_rating=round(rec_item['predicted_rating'], 2)
                )
            )
    
    return RecommendationResponse(
        user_id=user_id,
        recommendations=response_items
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict_ratings(request: PredictionRequest):
    """
    특정 영화들에 대한 평점 예측
    
    - **user_id**: 사용자 ID (training_user_id)
    - **movie_ids**: 예측할 영화 ID 목록
    """
    
    rec = get_recommender()
    predictions = rec.predict(request.user_id, request.movie_ids)
    
    return PredictionResponse(
        user_id=request.user_id,
        predictions=[
            PredictionItem(
                movie_id=mid,
                predicted_rating=round(rating, 2)
            )
            for mid, rating in predictions.items()
        ]
    )