from pydantic import BaseModel,field_validator
from typing import Optional, List
from datetime import date
import ast

class MovieBase(BaseModel):
    title: str
    original_title: Optional[str] = None
    overview: Optional[str] = None
    genres: Optional[List[str]] = None
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
            try:
                # "['액션', '모험']" 형태를 파싱
                return ast.literal_eval(v)
            except:
                return None
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
        