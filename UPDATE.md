# 📋 Updates & Changelog

Bob Movie Recommendation 프로젝트의 업데이트 내역입니다.
---

## v2.2.0 (2026-03-24)

### ✨ 새로운 기능

#### 1. 추천 다양성 및 설명성 개선 (#27)
- **MMR(Maximal Marginal Relevance) 다양성 보정**
  - Jaccard 유사도 기반 장르 다양성 알고리즘
  - lambda_param=0.7 (관련성 70%, 다양성 30%)
  - `apply_mmr` 파라미터로 On/Off 제어 가능
  
- **추천 이유 표시**
  - 전략별 추천 근거 메시지 제공
    - popular: "많은 사용자가 높게 평가한 인기 영화"
    - content_based: "좋아하신 영화와 유사한 장르"
    - hybrid: "당신의 평점 패턴 기반 개인화 추천"
  
- **API 스키마 확장**
  - `RecommendationResponse`: `strategy` 필드 추가
  - `RecommendationItem`: `reason`, `genres` 필드 추가

**변경된 파일:**
- `app/schemas/recommendation.py`
- `app/ml/inference/predictor.py` (apply_mmr_diversity 메서드)
- `app/api/v1/endpoints/recommendations.py`
- `tests/test_hybrid_recommendations.py`

#### 2. NCF Incremental Retraining + 추천 API 성능 최적화 (#24)

**2-1. Incremental Retraining 파이프라인**
- 서비스 유저를 NCF 모델에 포함하는 재학습 파이프라인 구축
- Warm start 방식: 기존 가중치 복사, 신규 유저만 초기화
- user_embedding 확장: [146,340, 64] → [146,345, 64]
- `service_user_mapping` 지원으로 서비스 유저 NCF 추론 가능
- **성과**: RMSE 1.3858 → 1.2771 개선

**신규 파일:**
- `scripts/retrain_with_service_users.py`
- `app/ml/inference/predictor.py` (service_user_mapping 지원)

**2-2. 추천 API 성능 최적화**
- `movie_stats` 사전 집계 테이블 도입 (Alembic 마이그레이션)
- training_ratings 79M행 실시간 GROUP BY 제거
- 평점 변경 시 BackgroundTask로 movie_stats 실시간 갱신
- **성과**: 추천 API 응답 시간 **~7분 → 밀리초** (약 42,000배 개선)

**신규 파일:**
- `app/models/movie_stats.py`
- `app/services/movie_stats_updater.py`
- `scripts/generate_movie_stats.py`
- Alembic migration: `add_movie_stats_table`

**변경된 파일:**
- `app/api/v1/endpoints/recommendations.py` (movie_stats 조회)
- `app/api/v1/endpoints/ratings.py` (BackgroundTask 추가)

### v2.1.0 (2026-03-22)

#### 🎯 하이브리드 추천 시스템 구현
- **지능형 전략 선택**: 사용자 평점 개수에 따라 최적 추천 방식 자동 선택
  - 평점 0개: 인기 영화 추천 (평균 평점 기반)
  - 평점 1~4개: 콘텐츠 기반 필터링 (장르 유사도)
  - 평점 5개+: 하이브리드 (NCF 70% + CF 30%)
- **협업 필터링 통합**: 코사인 유사도 기반 사용자-사용자 협업 필터링
- **성능 최적화**: Redis 캐싱으로 추천 결과 캐시 (TTL: 1시간)

#### 🧪 테스트 인프라 개선
- **pytest 환경 구축**: 추천 전략별 통합 테스트 3개 작성
- **비동기 격리 설정**: conftest.py로 SQLAlchemy async engine 격리
  - Event loop 충돌 문제 해결
  - `@pytest_asyncio.fixture(autouse=True)` 패턴 적용
  - `asyncio_mode=auto` 설정
- **테스트 커버리지**: 하이브리드 추천 시스템 전 전략 검증


### v2.0.0 (2025-03-20)

#### ⚡ 성능 개선
- **Redis 캐싱 시스템 도입**: 추천 API 응답 속도 **29배 향상**
  - 첫 요청: 321ms (모델 계산)
  - 캐시 히트: 11ms
- 평점 변경 시 자동 캐시 무효화
- JWT Refresh Token 블랙리스트 구현

#### 🎯 주요 기능 추가
- 영화 상세 조회 API (`GET /api/v1/movies/{id}`)
- Redis 기반 캐싱 시스템 (TTL: 1시간)

#### 📚 개발 환경 개선
- GitHub 템플릿 추가 (PR/Issue)
- CONTRIBUTING.md 가이드 작성
- docker-compose.yml 정리

### v1.5.0 (2025-02-25)

#### 🎯 주요 기능 추가
**1. 평점 시스템 완성**
- 평점 등록 (POST /api/v1/ratings)
- 평점 목록 조회 (GET /api/v1/ratings)
- 특정 영화 평점 조회 (GET /api/v1/ratings/movie/{id})
- 평점 수정 (PUT /api/v1/ratings/{id})
- 평점 삭제 (DELETE /api/v1/ratings/{id})
- 평점 통계 (GET /api/v1/ratings/stats/summary)

**2. 인증 시스템 구현**
- JWT 기반 인증 (Access Token + Refresh Token)
- 회원가입 (POST /api/v1/auth/register)
- 로그인 (POST /api/v1/auth/login)
- 토큰 갱신 (POST /api/v1/auth/refresh)
- 로그아웃 (POST /api/v1/auth/logout)

**3. 데이터베이스 설계**
- users 테이블 (서비스 사용자)
- user_ratings 테이블 (사용자 평점)
- movies 테이블 (영화 정보)
- training_users, training_ratings 테이블 (학습 데이터)

#### 🛠 인프라 구축
- Docker Compose 환경 구성
  - MySQL 8.0 (포트 13306)
  - Redis 7 (포트 6379)
- Alembic 마이그레이션 설정

### v1.0.0 (2025-01-23)

#### 🎯 ML 모델 개발
**1. NCF 모델 학습**
- Neural Collaborative Filtering 구현 (PyTorch)
- 하이퍼파라미터 최적화
  - Embedding Dimension: 32
  - Hidden Layers: [64, 32, 16]
  - Dropout: 0.1
  - Learning Rate: 0.001
- GPU 가속 학습 (Mixed Precision)
- 모델 저장 (best_ncf_model.pth)

**2. 데이터 임포트 파이프라인**
- 영화 데이터 임포트 (20,000+ 편)
- 학습용 사용자 데이터 임포트 (140,000+ 명)
- 평점 데이터 임포트 (79,000,000+ 건)
- 데이터 검증 스크립트

**3. 추천 로직 구현**
- app/ml/inference/predictor.py: MovieRecommender 클래스
- 기본 NCF 예측 로직
- 학습 데이터 기반 추천

### v0.1.0 (2025-01-22)

#### 🎯 프로젝트 초기 설정
**1. 프로젝트 구조 설계**
- FastAPI 프로젝트 구조 설계
- SQLAlchemy 비동기 ORM 설정
- Pydantic 스키마 정의

**2. 개발 환경 구축**
- Python 3.11 가상환경 설정
- requirements.txt 작성
- Git 저장소 초기화

**3. 학습 데이터 수집**
- 기존에 사용헀던 네이버, 다음, 왓챠 크롤링 데이터 확보
- CSV 파일 정제 및 저장
