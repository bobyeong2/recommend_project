from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func #260322추가
from typing import List, Dict #260322추가

from app.core.database import get_db
from app.models.movie import Movie
from app.models.user_rating import UserRating # 260320추가
# from app.models.training import TrainingRating #260322추가
from app.models.movie_stats import MovieStats #260323추가
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
    
    기존: training_ratings 79M row Group by (7Min)
    개선: movie_stats 약 2만행 select (밀리초)
    """
    result = await db.execute(
        select(
            MovieStats.movie_id,
            MovieStats.avg_rating,
            MovieStats.rating_count
        )
        .where(MovieStats.rating_count >= 10)
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
    
    기존: training_ratings 79M row 에서 candiate 2만개 groupby (약7 - 9분)
    개선: movie_stats에서 candidate IN 조회 (밀리초)
    """
    if not candidate_movie_ids:
        return {}
    result = await db.execute(
        select(MovieStats.movie_id, MovieStats.avg_rating)
        .where(MovieStats.movie_id.in_(candidate_movie_ids))
    )
    
    return {
        row.movie_id: float(row.avg_rating)
        for row in result.fetchall()
    }
    
@router.get("",response_model=RecommendationResponse)
async def get_my_recommendations(
    top_k: int = Query(10, ge=1, le=100, description="추천 영화 개수"),
    apply_mmr: bool = Query(True, description="MMR 다양성 보장 적용 여부"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    
    """
    로그인한 사용자 맞춤 추천 (Redis 캐싱)
    
    - top_k : 추천할 영화 개수 (기본 10개)
    - apply_mmr : MMR 다양성 보정 적용 여부 (default True)
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
    strategy = ""
    reason_template = ""
    if rating_count == 0 :
        # 신규 사용자용
        logger.info(f"인기 영화 추천: user_id={user_id}")
        strategy = "popular"
        reason_template = "많은 사양자가 높게 평가한 인기영화"
        
        movie_stats = await get_movie_stats(db)
        recommendations = rec.recommend_popular(movie_stats, top_k * 2) # MMR을 위해 top_k의 2배를 지정
        
    elif rating_count <= 4:
        # 초기 사용자: CBF 기반
        logger.info(f"CBF 기반 추천: user_id = {user_id}, rating_count={rating_count}")
        strategy = "content_based"
        reason_template = "종하사는 영화와 유사한 장르"
        
        user_rated_movies = await get_user_rated_movies_with_genres(db, user_id)
        rated_movie_ids = [r["movie_id"] for r in user_rated_movies]
        candiate_movies = await get_candidate_movies_with_genres(db, rated_movie_ids)
        recommendations = rec.recommend_content_based(user_rated_movies, candiate_movies, top_k * 2)
    
    else :
        # 그외 사용자 (충분한 평점을 갖고 있는 경우)
        logger.info(f"하이브리드 추천: user_id = {user_id}, rating_count = {rating_count}")
        strategy = "hybrid"
        reason_template = "당신의 평점 패턴 기반 개인화 추천"
        rated_movie_ids = [r.movie_id for r in user_ratings]
        candidate_movie_ids = [mid for mid in all_movie_ids if mid not in rated_movie_ids]
        
        #cf 점수 계산
        cf_scores = await calculate_collaborative_scores(db, user_id, candidate_movie_ids)

        # 하이브리드 추천
        recommendations = rec.recommend_hybrid(
            user_id,
            candidate_movie_ids,
            cf_scores,
            top_k * 2,
            ncf_weight=0.7
            
        )
        
    if apply_mmr and len(recommendations) > top_k:
        # 장르 정보를 추가
        movie_ids_for_genres = [r["movie_id"] for r in recommendations]
        result = await db.execute(
            select(Movie.id, Movie.genres)
            .where(Movie.id.in_(movie_ids_for_genres))
        )
        genres_map = {movie_id: genres for movie_id, genres in result.fetchall()}
        
        # recommendations에 장르 추가
        
        for rec_item in recommendations:
            rec_item["genres"] = genres_map.get(rec_item["movie_id"],"")
            
        # MMR 적용
        recommendations = rec.apply_mmr_diversity(recommendations, top_k, lambda_param=0.7)
    else :
        recommendations = recommendations[:top_k]
        
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
                    predicted_rating=round(rec_item["predicted_rating"],2),
                    reason=reason_template,
                    genres=movie.genres
                )
            )
    
    # Res 데이터 생성
    response_data = {
        "user_id": user_id,
        "strategy": strategy,
        "recommendations": [item.model_dump() for item in response_items]
    }
    
    await redis_client.set_recommendation_cache(user_id, response_data, ttl=3600)
    
    logger.info(f"추천 완료 및 캐싱 : user_id={user_id}, count={len(response_items)}, strategy={strategy}")
    
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
                    predicted_rating=round(rec_item['predicted_rating'], 2),
                    genres=movie.genres
                )
            )
    
    return RecommendationResponse(
        user_id=user_id,
        strategy="ncf",
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