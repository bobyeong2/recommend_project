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
- **추천 알고리즘**: CBF + CF → **NCF (딥러닝) + 하이브리드**
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

**Testing**
- pytest + pytest-asyncio
- httpx (async HTTP client)

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

### 지능형 추천 전략 🎯
사용자의 평점 개수에 따라 최적의 추천 전략을 자동 선택:

- **평점 0개**: 인기 영화 추천 (평균 평점 기반)
- **평점 1~4개**: 콘텐츠 기반 필터링 (장르 유사도)
- **평점 5개 이상**: 하이브리드 추천
  - NCF 딥러닝 예측 (70%)
  - 협업 필터링 (30%)
  - 최종 가중 평균으로 추천

---

## 📁 프로젝트 구조
```
recommend_project/
├── app/
│   ├── api/v1/endpoints/      # API 엔드포인트
│   ├── core/                  # 설정, DB, 인증, Redis
│   ├── ml/                    # NCF 모델, 추천 로직
│   ├── models/                # SQLAlchemy 모델
│   ├── schemas/               # Pydantic 스키마
│   └── main.py
├── tests/                     # pytest 테스트
│   ├── conftest.py           # pytest 설정 (async engine 격리)
│   └── test_hybrid_recommendations.py
├── data/                      # 학습 데이터
├── models/                    # 학습된 모델 (.pth)
├── alembic/                   # DB 마이그레이션
├── docker-compose.yml
├── pytest.ini                 # pytest 설정
└── requirements.txt
```

---

## 🧠 추천 알고리즘

**Neural Collaborative Filtering (NCF)**
- Embedding Dimension: 32
- Hidden Layers: [64, 32, 16]
- Dropout: 0.1
- Optimizer: Adam (LR: 0.001)
- RMSE: 1.3858

**하이브리드 전략**
```python
# 평점 5개 이상 사용자
final_score = (NCF_prediction × 0.7) + (CF_score × 0.3)

# 협업 필터링 (CF)
- 코사인 유사도 기반 사용자-사용자 유사도
- Top-K 유사 사용자의 평점 가중 평균
```

---

## 🧪 테스트
```bash
# 전체 테스트 실행
pytest

# 특정 테스트 실행
pytest tests/test_hybrid_recommendations.py -v

# 커버리지 확인
pytest --cov=app tests/
```

**테스트 커버리지**
- 추천 전략별 통합 테스트 (인기/CBF/Hybrid)
- 비동기 DB 세션 격리 (conftest.py)
- Event loop 충돌 해결

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

# 테스트
pytest tests/ -v
```

---

## 📋 업데이트 내역

최신 업데이트 내역은 [UPDATE.md](./UPDATE.md)를 참고하세요.

**Current Version**: v2.2.0

## 📊 데이터셋

| 구분 | 개수 |
|------|------|
| 영화 | 20,228편 |
| 학습용 사용자 | 146,340명 |
| 평점 데이터 | 79,201,585건 |

**출처**: 네이버, 다음, 왓챠

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

---