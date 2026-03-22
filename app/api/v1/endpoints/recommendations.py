from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func #260322추가
from typing import List, Dict #260322추가

from app.core.database import get_db
from app.models.movie import Movie
from app.models.user_rating import UserRating # 260320추가
from app.models.training import TrainingRating #260322추가
from app.models.user import User # 260320추가
from app.ml.inference.predictor import MovieRecommender
from app.schemas.recommendation import (
    PredictionRequest,
    PredictionResponse,
    PredictionItem,
    RecommendationResponse,
    RecommendationItem
)

from app.api.dependencies import get_current_user # 260320추가
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

async def get_movie_stats(db : AsyncSession) -> List[Dict]:
    """
    모든 영화의 통계 정보 조회 (인기 추천용)
    """
    result = await db.execute(
        select(
            TrainingRating.movie_id,
            func.avg(TrainingRating.rating).label("avg_rating"),
            func.count(TrainingRating.id).label("rating_count")
        )
        .group_by(TrainingRating.movie_id)
        .having(func.count(TrainingRating.id) >= 10) # 최소 10개 평점
    )
    
    return [
        {
            "movie_id": row.movie_id,
            "avg_rating": float(row.avg_rating),
            "rating_count": int(row.rating_count)
        }
        for row in result.fetchall()
    ]
    
async def get_user_rated_movies_with_genres(
    db: AsyncSession,
    user_id: int
) -> List[Dict]:
    """
    사용자가 평가한 영화와 장르정보 (CBF용)
    """
    result = await db.execute(
        select(UserRating, Movie.genres)
        .join(Movie, UserRating.movie_id == Movie.id)
        .where(UserRating.user_id == user_id)
    )
    
    return [
        {
            "movie_id": rating.movie_id,
            "rating": float(rating.rating),
            "genres": movie_genres or ""
        }
        for rating, movie_genres in result.fetchall()
    ]

async def get_candidate_movies_with_genres(
    db: AsyncSession,
    exclude_movie_ids: List[int]
) -> List[Dict]:
    """
    추천 후보 영화 + 장르 정보
    """
    result = await db.execute(
        select(Movie.id, Movie.genres)
        .where(Movie.id.notin_(exclude_movie_ids) if exclude_movie_ids else True)
    )
    
    return [
        {
            "movie_id": movie_id,
            "genres": genres or ""
        }
        for movie_id, genres in result.fetchall()
    ]


async def calculate_collaborative_scores(
    db: AsyncSession,
    user_id: int,
    candidate_movie_ids: List[int],
    top_similarity_users: int = 50
) -> Dict[int, float]:
    """
    협업 필터링 점수 계산 (하이브리드용)
    
    간단한 User-based CF:
    1. 시용자와 비슷한 평점 패턴을 가진 사용자를 찾은 뒤
    2. 그들이 높게 평가한 영화를 점수화함
    """
    
    user_ratings_result = await db.execute(
        select(UserRating.movie_id, UserRating.rating)
        .where(UserRating.user_id == user_id)
    )
    user_ratings = { r.movie_id: r.rating for r in user_ratings_result.fetchall()}

    if not user_ratings:
        return {}

    # 간단한 CF (추후 업데이트 예정)
    result = await db.execute(
        select(
            TrainingRating.movie_id,
            func.avg(TrainingRating.rating).label("avg_rating")
        )
        .where(TrainingRating.movie_id.in_(candidate_movie_ids))
        .group_by(TrainingRating.movie_id)
    )
    
    return {
        row.movie_id: float(row.avg_rating)
        for row in result.fetchall()
    }
    
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
    
    # 전략 선택
    rec = get_recommender()
    if rating_count == 0 :
        # 신규 사용자용
        logger.info(f"인기 영화 추천: user_id={user_id}")
        movie_stats = await get_movie_stats(db)
        recommendations = rec.recommend_popular(movie_stats, top_k)
        
    elif rating_count <= 4:
        # 초기 사용자: CBF 기반
        logger.info(f"CBF 기반 추천: user_id = {user_id}, rating_count={rating_count}") 
        user_rated_movies = await get_user_rated_movies_with_genres(db, user_id)
        rated_movie_ids = [r["movie_id"] for r in user_rated_movies]
        candiate_movies = await get_candidate_movies_with_genres(db, rated_movie_ids)
        recommendations = rec.recommend_content_based(user_rated_movies, candiate_movies, top_k)
    
    else :
        # 그외 사용자
        logger.info(f"하이브리드 추천: user_id = {user_id}, rating_count = {rating_count}")
        rated_movie_ids = [r.movie_id for r in user_ratings]
        candidate_movie_ids = [mid for mid in all_movie_ids if mid not in rated_movie_ids]
        
        #cf 점수 계산
        cf_scores = await calculate_collaborative_scores(db, user_id, candidate_movie_ids)

        # 하이브리드 추천
        recommendations = rec.recommend_hybrid(
            user_id,
            candidate_movie_ids,
            cf_scores,
            top_k,
            ncf_weight=0.7
            
        )
        
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