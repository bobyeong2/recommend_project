from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from typing import List

from app.core.database import get_db
from app.models.user import User
from app.models.user_rating import UserRating
from app.models.movie import Movie
from app.schemas.rating import RatingCreate, RatingUpdate, RatingResponse, RatingWithMovie
from app.api.dependencies import get_current_user

router = APIRouter()


@router.post("", response_model=RatingResponse, status_code=status.HTTP_201_CREATED)
async def create_rating(
    rating_data: RatingCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    평점 등록
    
    - 1.0 ~ 10.0 범위, 0.5 단위
    - 한 영화당 하나의 평점만 가능
    """
    
    # 영화 존재 확인
    result = await db.execute(
        select(Movie).where(Movie.id == rating_data.movie_id)
    )
    movie = result.scalar_one_or_none()
    
    if not movie:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="영화를 찾을 수 없습니다"
        )
    
    # 이미 평점을 등록했는지 확인
    result = await db.execute(
        select(UserRating).where(
            and_(
                UserRating.user_id == current_user.id,
                UserRating.movie_id == rating_data.movie_id
            )
        )
    )
    existing_rating = result.scalar_one_or_none()
    
    if existing_rating:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="이미 평점을 등록한 영화입니다. 수정하려면 PUT 요청을 사용하세요"
        )
    
    # 평점 생성
    new_rating = UserRating(
        user_id=current_user.id,
        movie_id=rating_data.movie_id,
        rating=rating_data.rating
    )
    
    db.add(new_rating)
    await db.commit()
    await db.refresh(new_rating)
    
    return new_rating


@router.get("", response_model=List[RatingWithMovie])
async def get_my_ratings(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    내 평점 목록 조회
    
    - 최신순 정렬
    - 페이지네이션 지원
    """
    
    result = await db.execute(
        select(UserRating, Movie.title)
        .join(Movie, UserRating.movie_id == Movie.id)
        .where(UserRating.user_id == current_user.id)
        .order_by(UserRating.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    
    ratings = []
    for rating, movie_title in result.all():
        ratings.append(
            RatingWithMovie(
                id=rating.id,
                movie_id=rating.movie_id,
                movie_title=movie_title,
                rating=rating.rating,
                created_at=rating.created_at,
                updated_at=rating.updated_at
            )
        )
    
    return ratings


@router.get("/movie/{movie_id}", response_model=RatingResponse)
async def get_my_rating_for_movie(
    movie_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """특정 영화에 대한 내 평점 조회"""
    
    result = await db.execute(
        select(UserRating).where(
            and_(
                UserRating.user_id == current_user.id,
                UserRating.movie_id == movie_id
            )
        )
    )
    rating = result.scalar_one_or_none()
    
    if not rating:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="평점을 찾을 수 없습니다"
        )
    
    return rating


@router.put("/{rating_id}", response_model=RatingResponse)
async def update_rating(
    rating_id: int,
    rating_data: RatingUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """평점 수정"""
    
    # 평점 조회 (본인 것만)
    result = await db.execute(
        select(UserRating).where(
            and_(
                UserRating.id == rating_id,
                UserRating.user_id == current_user.id
            )
        )
    )
    rating = result.scalar_one_or_none()
    
    if not rating:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="평점을 찾을 수 없습니다"
        )
    
    # 평점 수정
    rating.rating = rating_data.rating
    await db.commit()
    await db.refresh(rating)
    
    return rating


@router.delete("/{rating_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_rating(
    rating_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """평점 삭제"""
    
    # 평점 조회 (본인 것만)
    result = await db.execute(
        select(UserRating).where(
            and_(
                UserRating.id == rating_id,
                UserRating.user_id == current_user.id
            )
        )
    )
    rating = result.scalar_one_or_none()
    
    if not rating:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="평점을 찾을 수 없습니다"
        )
    
    # 평점 삭제
    await db.delete(rating)
    await db.commit()


@router.get("/stats/summary")
async def get_my_rating_stats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """내 평점 통계"""
    
    result = await db.execute(
        select(
            func.count(UserRating.id).label('total_count'),
            func.avg(UserRating.rating).label('avg_rating'),
            func.max(UserRating.rating).label('max_rating'),
            func.min(UserRating.rating).label('min_rating')
        )
        .where(UserRating.user_id == current_user.id)
    )
    stats = result.one()
    
    return {
        "total_ratings": stats.total_count or 0,
        "average_rating": round(float(stats.avg_rating), 2) if stats.avg_rating else 0,
        "highest_rating": float(stats.max_rating) if stats.max_rating else 0,
        "lowest_rating": float(stats.min_rating) if stats.min_rating else 0
    }
    