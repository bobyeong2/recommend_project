# app/models/rating.py

from sqlalchemy import Column, Integer, ForeignKey, Date, UniqueConstraint
from sqlalchemy.sql import func
from app.core.database import Base

class UserRating(Base):
    """
    실제 서비스 사용자 평점
    회원가입한 사용자의 평점
    """
    __tablename__ = "user_ratings"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # 외래 키
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    movie_id = Column(Integer, ForeignKey('movies.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # 평점 (1-10)
    rating = Column(Integer, nullable=False)
    
    # 타임스탬프
    created_at = Column(Date, server_default=func.now())
    updated_at = Column(Date, server_default=func.now(), onupdate=func.now())
    
    # Unique 제약조건
    __table_args__ = (
        UniqueConstraint('user_id', 'movie_id', name='unique_user_movie'),
    )