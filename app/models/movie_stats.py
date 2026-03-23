from sqlalchemy import Column, Integer, Float, DateTime
from sqlalchemy.sql import func
from app.core.database import Base

class MovieStats(Base):
    """
    영화별 사전 집계 통계
    
    trainging_ratings(79M) + user ratings를 매 요청마다 집계하면 7-9분을 소요함
    매 요청마다 집계하는 대신 사전 계산된 통계를 저장해 API의 응답속도를 개선하기 위함
    
    갱신 시점:
    - 초기: generate_movie_stats.py 스크립트로 training_ratings 기준 1회 생성
    - 이후: 평점 등록/수정/삭제 시 BackgroundTask로 해당 movie_id만 갱신
    """
    
    __tablename__ = "movie_stats"
    
    movie_id = Column(Integer, primary_key=True, autoincrement=False)
    avg_rating = Column(Float, nullable=False, default=0.0)
    rating_count = Column(Integer, nullable=True, default=0)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    