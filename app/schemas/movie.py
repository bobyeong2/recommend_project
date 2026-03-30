from pydantic import BaseModel, field_validator
from typing import Optional, List
from datetime import date
import ast

class MovieBase(BaseModel):
    title: str
    original_title: Optional[str] = None
    overview: Optional[str] = None
    genres: Optional[List[str]] = None  # API 응답은 List[str] 유지
    runtime: Optional[int] = None
    release_date: Optional[date] = None
    poster_path: Optional[str] = None
    mean_rating: Optional[float] = None
    popularity: Optional[float] = None
    
    @field_validator('genres', mode='before')
    @classmethod
    def parse_genres(cls, v):
        if v is None:
            return None
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            # 파이프 구분 문자열 처리 (신규 형식)
            if '|' in v:
                return [g.strip() for g in v.split('|') if g.strip()]
            # 기존 형식 호환 (마이그레이션 중)
            try:
                return ast.literal_eval(v)
            except:
                return [v] if v else None
        return v
    
    class Config:
        from_attributes = True

class MovieResponse(MovieBase):
    id: int
    
    class Config:
        from_attributes = True
        
class MovieDetail(MovieResponse):
    """상세 정보 포함"""
    description: Optional[str] = None
    poster_url: Optional[str] = None
    
    class Config:
        from_attributes = True