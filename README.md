# 🎬 Bob Movie Recommendation

Neural Collaborative Filtering 기반 영화 추천 서비스

> 7,900만+ 평점 데이터로 학습된 딥러닝 모델을 활용한 개인화 추천 API

---

## 💡 프로젝트 소개

2019년 팀 프로젝트로 개발했던 영화 추천 시스템을 최신 기술 스택으로 완전히 재구성한 프로젝트입니다.

### 왜 만들었나?
- 과거 프로젝트를 최신 기술로 리뉴얼하는 경험
- 딥러닝 기반 추천 시스템 구현 학습
- FastAPI, Docker, Redis 등 현대적인 백엔드 기술 스택 적용

### 주요 개선사항
- **추천 알고리즘**: CBF + CF → **NCF (딥러닝)**
- **백엔드**: Flask → **FastAPI** (비동기 처리)
- **인프라**: Docker 컨테이너 기반 환경
- **데이터베이스**: MySQL 8.0 + Redis 캐싱

---

## 🛠 기술 스택

**Backend**
- FastAPI (비동기 웹 프레임워크)
- MySQL 8.0 + Redis 7
- SQLAlchemy (Async ORM)
- JWT 인증

**Machine Learning**
- PyTorch (NCF 모델)
- 하이브리드 추천 (NCF + 협업 필터링 + 콘텐츠 기반)

**DevOps**
- Docker, Docker Compose
- Alembic (DB 마이그레이션)

---

## 🚀 빠른 시작

### 1. 사전 준비
- Python 3.11+
- Docker & Docker Compose

### 2. 프로젝트 클론
```bash
git clone https://github.com/yourusername/bob-movie-api.git
cd bob-movie-api
```

### 3. 환경 설정
```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 4. 환경 변수 설정
`.env` 파일 생성:
```env
DATABASE_URL=mysql+aiomysql://bob_user:bobpass@localhost:13306/bob_movie_db
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=your-secret-key-change-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7
```

### 5. Docker 실행
```bash
# MySQL + Redis 컨테이너 시작
docker-compose up -d

# DB 마이그레이션
alembic upgrade head
```

### 6. 서버 실행
```bash
# 개발 모드
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 7. API 문서 확인
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 📚 주요 기능

### API 엔드포인트
```
인증:     POST /api/v1/auth/register, /login, /refresh
사용자:   GET/PUT /api/v1/users/me
영화:     GET /api/v1/movies, /movies/{id}, /movies/search
평점:     POST/GET/PUT/DELETE /api/v1/ratings
추천:     GET /api/v1/recommendations
```

### 추천 전략
- **평점 0개**: 인기 영화 추천
- **평점 1~4개**: 콘텐츠 기반 (장르 유사도)
- **평점 5개 이상**: 하이브리드 (NCF + 협업 필터링)

---

## 📁 프로젝트 구조
```
recommend_project/
├── app/
│   ├── api/v1/endpoints/      # API 엔드포인트
│   ├── core/                  # 설정, DB, 인증
│   ├── ml/                    # NCF 모델, 추천 로직
│   ├── models/                # SQLAlchemy 모델
│   ├── schemas/               # Pydantic 스키마
│   └── main.py
├── data/                      # 학습 데이터
├── models/                    # 학습된 모델 (.pth)
├── alembic/                   # DB 마이그레이션
├── docker-compose.yml
└── requirements.txt
```

---

## 🧠 추천 알고리즘

**Neural Collaborative Filtering (NCF)**
- Embedding Dimension: 32
- Hidden Layers: [64, 32, 16]
- Dropout: 0.1
- Optimizer: Adam (LR: 0.001)

**하이브리드 전략**
```
NCF 예측 (70%) + 협업 필터링 (30%)
```

---

## 🔧 개발 명령어
```bash
# 컨테이너 관리
docker-compose up -d         # 시작
docker-compose down          # 종료
docker-compose logs -f app   # 로그 확인

# 마이그레이션
alembic revision --autogenerate -m "description"
alembic upgrade head
```

---

## 📝 라이선스

MIT License

---

## 👨‍💻 개발자

**Bang Bobyeong**
- GitHub: [@bobyeong2](https://github.com/bobyeong2)
- Email: a01025494880@gmail.com

---

## 🙏 감사의 말

- 2019년 팀 프로젝트 팀원들
- 네이버, 다음, 왓챠 영화 데이터

# 업데이트 내역
v1.5.0 (2025-02-25)
🎯 주요 기능 추가
1. 평점 시스템 완성

평점 등록 (POST /api/v1/ratings)
평점 목록 조회 (GET /api/v1/ratings)
특정 영화 평점 조회 (GET /api/v1/ratings/movie/{id})
평점 수정 (PUT /api/v1/ratings/{id})
평점 삭제 (DELETE /api/v1/ratings/{id})
평점 통계 (GET /api/v1/ratings/stats/summary)

2. 인증 시스템 구현

JWT 기반 인증 (Access Token + Refresh Token)
회원가입 (POST /api/v1/auth/register)
로그인 (POST /api/v1/auth/login)
토큰 갱신 (POST /api/v1/auth/refresh)
로그아웃 (POST /api/v1/auth/logout)

3. 데이터베이스 설계

users 테이블 (서비스 사용자)
user_ratings 테이블 (사용자 평점)
movies 테이블 (영화 정보)
training_users, training_ratings 테이블 (학습 데이터)

🛠 인프라 구축

Docker Compose 환경 구성

MySQL 8.0 (포트 13306)
Redis 7 (포트 6379)


Alembic 마이그레이션 설정


v1.0.0 (2025-01-23)
🎯 ML 모델 개발
1. NCF 모델 학습

Neural Collaborative Filtering 구현 (PyTorch)
하이퍼파라미터 최적화

Embedding Dimension: 32
Hidden Layers: [64, 32, 16]
Dropout: 0.1
Learning Rate: 0.001


GPU 가속 학습 (Mixed Precision)
모델 저장 (best_ncf_model.pth)

2. 데이터 임포트 파이프라인

영화 데이터 임포트 (20,000+ 편)
학습용 사용자 데이터 임포트 (140,000+ 명)
평점 데이터 임포트 (79,000,000+ 건)
데이터 검증 스크립트

3. 추천 로직 구현

app/ml/inference/predictor.py: MovieRecommender 클래스
기본 NCF 예측 로직
학습 데이터 기반 추천


v0.1.0 (2025-01-22)
🎯 프로젝트 초기 설정
1. 프로젝트 구조 설계

FastAPI 프로젝트 구조 설계
SQLAlchemy 비동기 ORM 설정
Pydantic 스키마 정의

2. 개발 환경 구축

Python 3.11 가상환경 설정
requirements.txt 작성
Git 저장소 초기화

3. 학습 데이터 수집

네이버, 다음, 왓챠 크롤링 데이터 확보
CSV 파일 정제 및 저장