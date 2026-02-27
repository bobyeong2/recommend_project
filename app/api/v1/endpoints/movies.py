from fastapi import APIRouter, Depends, HTTPException,Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_
from typing import List, Optional
from app.core.database import get_db
from app.models.movie import Movie
from app.schemas.movie import MovieResponse

router = APIRouter()

@router.get("",response_model=List[MovieResponse])
async def get_movies(
    skip: int = Query(0, ge=0),
    limit: int = Query(20,ge=1,le=100),
    db: AsyncSession = Depends(get_db)
):
    
    """영화 목록 조회"""
    result = await db.execute(
        select(Movie)
        .offset(skip)
        .limit(limit)
    )
    return result.scalars().all()

@router.get("/search",response_model=List[MovieResponse])
async def search_movies(
    q: str = Query(...,min_length=1),
    limit: int = Query(20,ge=1,le=100),
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
    return result.scalars().all()
