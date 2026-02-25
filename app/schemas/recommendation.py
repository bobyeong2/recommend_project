from pydantic import BaseModel, Field
from typing import List

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
    
class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[RecommendationItem]
    
    