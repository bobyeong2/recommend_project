from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_
from typing import List
import ast

from app.core.database import get_db
from app.models.movie import Movie
from app.schemas.movie import MovieResponse

router = APIRouter()


def parse_movie(movie: Movie) -> dict:
    """
    SQLAlchemy Movie 객체를 dict로 변환하면서 genres 파싱
    genres 컬럼이 문자열로 저장되어 있어서 직접 파싱 필요
    """
    genres = movie.genres
    if genres is None:
        parsed_genres = None
    elif isinstance(genres, list):
        parsed_genres = genres
    elif isinstance(genres, str):
        if '|' in genres:
            parsed_genres = [g.strip() for g in genres.split('|') if g.strip()]
        else:
            try:
                parsed_genres = ast.literal_eval(genres)
            except Exception:
                parsed_genres = [genres] if genres else None
    else:
        parsed_genres = None

    return {
        "id": movie.id,
        "title": movie.title,
        "original_title": movie.original_title,
        "overview": movie.overview,
        "genres": parsed_genres,
        "runtime": movie.runtime,
        "release_date": movie.release_date,
        "poster_path": movie.poster_path,
        "mean_rating": movie.mean_rating,
        "popularity": movie.popularity,
    }


@router.get("", response_model=List[MovieResponse])
async def get_movies(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """영화 목록 조회"""
    result = await db.execute(
        select(Movie)
        .offset(skip)
        .limit(limit)
    )
    movies = result.scalars().all()
    return [parse_movie(m) for m in movies]


@router.get("/search", response_model=List[MovieResponse])
async def search_movies(
    q: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """영화 검색"""
    result = await db.execute(
        select(Movie)
        .where(
            or_(
                Movie.title.like(f"%{q}%"),
                Movie.original_title.like(f"%{q}%")
            )
        )
        .limit(limit)
    )
    movies = result.scalars().all()
    return [parse_movie(m) for m in movies]


@router.get("/{movie_id}", response_model=MovieResponse)
async def get_movie(
    movie_id: int,
    db: AsyncSession = Depends(get_db)
):
    """영화 상세 조회"""
    result = await db.execute(
        select(Movie).where(Movie.id == movie_id)
    )
    movie = result.scalars().first()

    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")

    return parse_movie(movie)