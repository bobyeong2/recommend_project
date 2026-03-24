import pytest
from httpx import AsyncClient, ASGITransport  
from app.main import app 
import time
from app.core.database import engine

# 테스트 실행 시각을 이메일에 추가
timestamp = str(int(time.time()))

@pytest.mark.asyncio
async def test_recommedation_strategy_popular():
    
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        
        register_response = await client.post(
            "api/v1/auth/register",
            json={
                "email": f"newuser{timestamp}@test.com",
                "username": f"newuser{timestamp}",
                "password": "Test1234!"
            }
        )
        assert register_response.status_code == 201
        
        login_response = await client.post(
            "api/v1/auth/login",
            json={
                "email": f"newuser{timestamp}@test.com",
                "password": "Test1234!"
            }
        )
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]
        
        rec_response = await client.get(
            "api/v1/recommendations",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert rec_response.status_code == 200
        data = rec_response.json()
        
        assert "recommendations" in data
        assert len(data["recommendations"]) > 0 
        
        avg_rating = sum(r["predicted_rating"] for r in data["recommendations"]) / len((data["recommendations"]))
        assert avg_rating >= 7.0

        # 새로운 필드 검증
        assert "strategy" in data
        assert data["strategy"] == "popular"
        
        # 첫 번째 추천 아이템 검증
        first_rec = data["recommendations"][0]
        assert "movie_id" in first_rec
        assert "title" in first_rec
        assert "predicted_rating" in first_rec
        # reason, genres는 Optional이므로 있으면 확인
        if "reason" in first_rec:
            assert isinstance(first_rec["reason"], str)
        if "genres" in first_rec:
            assert isinstance(first_rec["genres"], str)
            
@pytest.mark.asyncio
async def test_recommendation_strategy_content_based():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        
        register_response = await client.post(
            "api/v1/auth/register",
            json={
                "email": f"cbfuser{timestamp}@test.com",
                "username": f"cbfuser{timestamp}",
                "password": "Test1234!"
            }
        )
        assert register_response.status_code == 201
        
        login_response = await client.post(
            "api/v1/auth/login",
            json={
                "email": f"cbfuser{timestamp}@test.com",
                "password": "Test1234!"
            }
        )
        token = login_response.json()["access_token"]
        
        # 평점 3개 등록
        for movie_id in [1, 2, 3]:
            await client.post(
                "api/v1/ratings",
                headers={"Authorization": f"Bearer {token}"},
                json={"movie_id": movie_id, "rating": 9.0}
            )
            
        # 추천 요청
        rec_response = await client.get(
            "api/v1/recommendations",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert rec_response.status_code == 200
        data = rec_response.json()
        
        assert "recommendations" in data
        assert len(data["recommendations"]) > 0
        
        # 새로운 필드 검증
        assert "strategy" in data
        assert data["strategy"] == "content_based"
        

@pytest.mark.asyncio
async def test_recommendation_strategy_hybrid():
    """평점 5개 이상 사용자 → 하이브리드 추천"""
    
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        
        # 회원가입
        register_response = await client.post(
            "api/v1/auth/register",
            json={
                "email": f"hybriduser{timestamp}@test.com",
                "username": f"hybriduser{timestamp}",
                "password": "Test1234!"
            }
        )
        assert register_response.status_code == 201
        
        # 로그인
        login_response = await client.post(
            "api/v1/auth/login",
            json={
                "email": f"hybriduser{timestamp}@test.com",
                "password": "Test1234!"
            }
        )
        token = login_response.json()["access_token"]
        
        # 평점 10개 등록
        for movie_id in range(1, 11):
            await client.post(
                "api/v1/ratings",
                headers={"Authorization": f"Bearer {token}"},
                json={"movie_id": movie_id, "rating": 8.0}
            )
        
        # 추천 요청
        rec_response = await client.get(
            "api/v1/recommendations",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert rec_response.status_code == 200
        data = rec_response.json()
        
        assert "recommendations" in data
        assert len(data["recommendations"]) > 0

        # 새로운 필드 검증
        assert "strategy" in data
        assert data["strategy"] == "hybrid"
        