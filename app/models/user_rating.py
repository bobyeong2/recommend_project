from sqlalchemy import Column, Integer, Float, ForeignKey, Date, UniqueConstraint
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.core.database import Base


class UserRating(Base):
    """
    실제 서비스 사용자 평점
    """
    __tablename__ = "user_ratings"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # 외래 키
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    movie_id = Column(Integer, ForeignKey("movies.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # 평점 (1.0 ~ 10.0, 0.5 단위)
    rating = Column(Float, nullable=False)
    
    # 타임스탬프
    created_at = Column(Date, server_default=func.now())
    updated_at = Column(Date, server_default=func.now(), onupdate=func.now())
    
    # 관계
    user = relationship("User", backref="ratings")
    movie = relationship("Movie", backref="user_ratings")
    
    # 제약 조건: 한 사용자는 한 영화에 하나의 평점만
    __table_args__ = (
        UniqueConstraint('user_id', 'movie_id', name='unique_user_movie_rating'),
    )