"""
movie_stats 갱신 유틸 파일

평점 등록/수정/삭제 시 BackgroundTask로 호출되어 해당 movie_id의 통계만 갱신 

training_ratings + user_ratings를 합산해 계산함

"""
import asyncio
import logging
from sqlalchemy import text

from app.core.database import AsyncSessionLocal

logger = logging.getLogger(__name__)

async def update_movie_stats(movie_id: int):
    """
    특정 영화의 movie_stats를 갱신하는 함수
    
    training_ratings + user_ratings 합산 평균/개수를 계산해
    movie_stats Table에 Update
    
    BackgroundTask에서 호출되어 별도의 세션을 사용
    """
    
    try :
        async with AsyncSessionLocal() as session:
            await session.execute(text(
                """
                    insert into movie_stats (movie_id, avg_rating, rating_count, updated_at)
                    select
                        combined.movie_id,
                        AVG(combined.rating) as avg_rating,
                        COUNT(*) as rating_count,
                        NOW() as updated_at
                    from(
                            select movie, rating
                                from training_ratings
                                where movie_id = :movie_id
                                union all
                                select movie_id, rating
                                    from user_ratings
                                    where movie_id = :movie_id
                    ) combined
                    
                    group by combined.movie_id
                    ON duplicate key update
                        avg_rating = values(avg_rating),
                        rating_count = values(rating_count),
                        update_at = NOW()
                """
            ), {"movie_id":movie_id})
            
            await session.commit()
            logger.info(f"movie_stats updated: movie={movie_id}")
    except Exception as e :
        logger.error(f"movie_stats update failed: movie_id={movie_id}, error={e}")
        
def run_update_movie_stats(movie_id: int):
    """
    BackgroundTask에서 호출하는 동기 wapper
    
    fastapi backgroundTasks는 동기 함수도 지원함.
    새 event loop에서 async 함수를 실행
    """
    
    loop = asyncio.new_event_loop()
    
    try:
        loop.run_until_complete(update_movie_stats(movie_id))
    finally:
        loop.close()
        