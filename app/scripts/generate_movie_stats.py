"""
movie_stats 테이블 초기 데이터 생성

training_ratings(79M)에서 영화별 avg_rating, rating_count를 집계
movie_stats table에 저장함.

1회성 실행, 이후 갱신은 BackGroundTask에서 처리됨

사용법:
    python scripts/generate_movie_stats.py
    
"""
import asyncio
from datetime import datetime
from sqlalchemy import text

from app.core.database import AsyncSessionLocal, engine

async def main():
    print(f"시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print("movie_stats 테이블 데이터 생성 중...")
    print("training_ratings 79M행 집계 (수 분 소요)")
    print("=" * 70)
    
    async with AsyncSessionLocal() as session:
        # 기존 데이터 삭제
        await session.execute(text("Delete From movie_stats"))
        await session.commit()
        print("기존 movie_stats Data 삭제 완료")
        
        # training_ratings에서 집계 후 INSERT
        print("training_ratings 집계 중...")
        result = await session.execute(text(
            """
            Insert into movie_stats (movie_id, avg_rating, rating_count, updated_at)
            select
                movie_id,
                AVG(rating) as avg_rating,
                count(*) as rating_count,
                NOW() as updated_at
            from training_ratings
            group by movie_id
            having count(*) >= 1
            """
        ))
        
        await session.commit()
        
        # 결과 확인
        count_result = await session.execute(text(
            """select count(*)
                from movie_stats
            """
        ))
        total = count_result.scalar()
        
        sample_result = await session.execute(text(
            """
            select movie_id, avg_rating, rating_count
                from movie_stats
                order by rating_count DESC
                limit 5
            """
        ))
        top_movies = sample_result.fetchall()
        
        print(f"\nmovie_stats 생성 완료: {total:,}건")
        print("\n인기 영화 TOP 5:")
        for movie_id, avg_rating, rating_count in top_movies:
            print(f"  movie_id={movie_id}, avg={avg_rating:.2f}, count={rating_count:,}")
 
    print(f"\n종료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    
if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(engine.dispose())
        loop.close()
 