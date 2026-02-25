# app/models/movie.py

from sqlalchemy import Column, Integer, String, Text, Date, Float, JSON
from sqlalchemy.sql import func
from app.core.database import Base

class Movie(Base):
    __tablename__ = "movies"
    
    # 기본 키
    id = Column(Integer, primary_key=True, autoincrement=True)  # 자동 증가 PK
    movie_code = Column(Integer, unique=True, index=True)  # 기존 CSV의 영화 코드 (매핑용)
    
    # 제목
    title = Column(String(500), nullable=False, index=True)  # 한글 제목 (검색 인덱스)
    original_title = Column(String(500))  # 영어 제목
    
    # 내용
    overview = Column(Text)  # 줄거리 (긴 텍스트)
    
    # 메타데이터
    genres = Column(JSON)  # ["액션", "SF"] - JSON 배열로 저장
    runtime = Column(Integer)  # 상영시간 (분)
    release_date = Column(Date)  # 개봉일
    
    # 이미지
    poster_path = Column(String(500))  # 포스터 이미지 URL
    
    # 평점 (참고용, 실제는 user_ratings에서 집계)
    mean_rating = Column(Float)  # CSV의 기존 평균 평점
    
    # 인기도
    popularity = Column(Float, default=0)  # 평점 개수 등으로 계산
    
    # 타임스탬프
    created_at = Column(Date, server_default=func.now())  # 생성일 (자동)
    updated_at = Column(Date, server_default=func.now(), onupdate=func.now())  # 수정일 (자동)