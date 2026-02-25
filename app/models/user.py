# app/models/user.py

from sqlalchemy import Column, Integer, String, Boolean, Date
from sqlalchemy.sql import func
from app.core.database import Base

class User(Base):
    """
    실제 서비스 사용자
    회원가입을 통해 생성된 사용자
    """
    __tablename__ = "users"
    
    # 기본 키
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # 인증 정보
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    
    # 프로필
    full_name = Column(String(100))
    
    # 상태
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    
    # 역할
    role = Column(String(20), default='user')  # 'user', 'admin'
    
    # 타임스탬프
    created_at = Column(Date, server_default=func.now())
    updated_at = Column(Date, server_default=func.now(), onupdate=func.now())
    last_login_at = Column(Date)