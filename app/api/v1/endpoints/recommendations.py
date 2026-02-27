from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List

from app.core.database import get_db
from app.models.movie import Movie
from app.ml.inference.predictor import MovieRecommender
from app.schemas.recommendation import (
    PredictionRequest,
    PredictionResponse,
    PredictionItem,
    RecommendationResponse,
    RecommendationItem
)

router = APIRouter()

# 전역 recommender (싱글톤)
recommender = None


def get_recommender() -> MovieRecommender:
    """Recommender 인스턴스 가져오기"""
    global recommender
    if recommender is None:
        recommender = MovieRecommender()
    return recommender


@router.get("/{user_id}", response_model=RecommendationResponse)
async def get_recommendations(
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