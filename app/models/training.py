# app/models/training.py

from sqlalchemy import Column, Integer, String, Date, ForeignKey, UniqueConstraint
from sqlalchemy.sql import func
from app.core.database import Base

class TrainingUser(Base):
    """
    학습용 익명 사용자
    크롤링 데이터의 user_id 저장
    NCF 모델 학습 전용
    """
    __tablename__ = "training_users"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # 크롤링 데이터의 원본 user_id (예: du566958)
    original_user_id = Column(String(100), unique=True, nullable=False, index=True)
    
    # 타임스탬프
    created_at = Column(Date, server_default=func.now())


class TrainingRating(Base):
    """
    학습용 평점 데이터
    크롤링 데이터의 평점 저장
    NCF 모델 학습 전용
    """
    __tablename__ = "training_ratings"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # 외래 키
    training_user_id = Column(Integer, ForeignKey('training_users.id', ondelete='CASCADE'), nullable=False, index=True)
    movie_id = Column(Integer, ForeignKey('movies.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # 평점 (1-10)
    rating = Column(Integer, nullable=False)
    
    # 출처 (naver, daum, watcha)
    source = Column(String(20), default='crawled')
    
    # 타임스탬프
    created_at = Column(Date, server_default=func.now())
    
    # Unique 제약조건 (한 사용자가 같은 영화에 1개 평점만)
    __table_args__ = (
        UniqueConstraint('training_user_id', 'movie_id', name='unique_training_user_movie'),
    )