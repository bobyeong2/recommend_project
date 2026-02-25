from pydantic import BaseModel, Field, validator
from datetime import date
from typing import Optional


class RatingCreate(BaseModel):
    movie_id: int
    rating: float = Field(..., ge=1.0, le=10.0)
    
    @validator('rating')
    def validate_rating(cls, v):
        """0.5 단위 검증"""
        if v % 0.5 != 0:
            raise ValueError('평점은 0.5 단위로 입력해주세요 (예: 7.5, 8.0)')
        return v


class RatingUpdate(BaseModel):
    rating: float = Field(..., ge=1.0, le=10.0)
    
    @validator('rating')
    def validate_rating(cls, v):
        """0.5 단위 검증"""
        if v % 0.5 != 0:
            raise ValueError('평점은 0.5 단위로 입력해주세요 (예: 7.5, 8.0)')
        return v


class RatingResponse(BaseModel):
    id: int
    user_id: int
    movie_id: int
    rating: float
    created_at: date
    updated_at: date
    
    class Config:
        from_attributes = True


class RatingWithMovie(BaseModel):
    """영화 정보 포함"""
    id: int
    movie_id: int
    movie_title: str
    rating: float
    created_at: date
    updated_at: date
    
    class Config:
        from_attributes = True