from pydantic import BaseModel, Field
from typing import List, Optional

class PredictionRequest(BaseModel):
    user_id: int = Field(..., description="사용자 ID")
    movie_ids: List[int] = Field(..., description="예측할 영화 ID 목록")
    
class PredictionItem(BaseModel):
    movie_id: int
    predicted_rating: float
    
class PredictionResponse(BaseModel):
    user_id: int
    predictions: List[PredictionItem]
    
class RecommendationItem(BaseModel):
    movie_id: int
    title: str
    predicted_rating: float
    reason: Optional[str] = Field(None, description="추천 이유")
    genres: Optional[str] = Field(None, description="장르")
class RecommendationResponse(BaseModel):
    user_id: int
    strategy: str = Field(..., description="추천 전략 (popular/content_based/hybrid/ncf)")
    recommendations: List[RecommendationItem]
    
    