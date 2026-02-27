import asyncio
import pandas as pd
from sqlalchemy import select
from app.core.database import AsyncSessionLocal, engine
from app.models.movie import Movie
from app.models.training import TrainingUser, TrainingRating

async def import_training_ratings_batch(session, batch_df, movie_map, user_map, batch_num):
    """단일 배치 처리"""
    ratings_data = []
    skipped = 0
    
    for _, row in batch_df.iterrows():
        movie_id = movie_map.get(int(row["movie_code"]))
        if not movie_id:
            skipped += 1
            continue
        
        training_user_id = user_map.get(str(row['user_id']))
        if not training_user_id:
            skipped += 1
            continue
        
        ratings_data.append({
            "training_user_id": training_user_id,
            "movie_id": movie_id,
            "rating": int(row["score"]),
            "source": "crawled"
        })
    
    if ratings_data:
        await session.run_sync(
            lambda sync_session: sync_session.bulk_insert_mappings(
                TrainingRating, ratings_data
            )
        )
        await session.commit()
    
    print(f"  배치 {batch_num} 완료: {len(ratings_data):,}건 (스킵: {skipped:,})")
    return len(ratings_data), skipped


async def import_training_ratings():
    """training_ratings만 임포트"""
    try:
        df = pd.read_csv("data/users_super_final.csv", encoding="utf8")
        print(f"평점 데이터: {len(df):,}건")
        
        # 매핑 데이터 로드
        async with AsyncSessionLocal() as map_session:
            movie_result = await map_session.execute(select(Movie.movie_code, Movie.id))
            movie_map = {code: id for code, id in movie_result}
            
            user_result = await map_session.execute(
                select(TrainingUser.original_user_id, TrainingUser.id)
            )
            user_map = {uid: id for uid, id in user_result}
        
        print(f"영화 매핑: {len(movie_map):,}개")
        print(f"사용자 매핑: {len(user_map):,}명")
        
        batch_size = 20000
        total_batches = (len(df) + batch_size - 1) // batch_size
        print(f"총 {total_batches}개 배치 처리\n")
        
        total_imported = 0
        total_skipped = 0
        
        for i in range(0, len(df), batch_size):
            batch_df = df[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            
            async with AsyncSessionLocal() as batch_session:
                try:
                    imported, skipped = await import_training_ratings_batch(
                        batch_session,
                        batch_df,
                        movie_map,
                        user_map,
                        batch_num
                    )
                    total_imported += imported
                    total_skipped += skipped
                except Exception as e:
                    print(f"  배치 {batch_num} 에러: {e}")
                    await batch_session.rollback()
                    raise
        
        print(f"\n임포트 완료!")
        print(f"   임포트: {total_imported:,}건")
        print(f"   스킵: {total_skipped:,}건")
    
    finally:
        await engine.dispose()


if __name__ == "__main__":
    import time
    start = time.time()
    
    asyncio.run(import_training_ratings())
    
    elapsed = time.time() - start
    print(f"\n소요 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")