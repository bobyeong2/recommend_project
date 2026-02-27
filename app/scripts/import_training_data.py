# import_training_data.py

import asyncio
import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.mysql import insert
from app.core.database import AsyncSessionLocal, engine
from app.models.movie import Movie
from app.models.training import TrainingUser, TrainingRating

async def import_training_users():
    """
    평점 데이터에서 unique user_id를 추출하여 training_users를 생성
    bulk_insert_mappings 사용으로 대폭 속도 향상
    """
    try:
        df = pd.read_csv("data/users_super_final.csv", encoding="utf8")
        print(f"평점 데이터: {len(df):,}건")
        
        unique_users = df["user_id"].unique()
        print(f"unique user: {len(unique_users):,}명")
        
        async with AsyncSessionLocal() as session:
            # bulk insert용 데이터 준비
            user_data = [
                {"original_user_id": str(user_id)}
                for user_id in unique_users
            ]
            
            # 10,000개씩 배치 삽입
            batch_size = 10000
            total = len(user_data)
            
            for i in range(0, total, batch_size):
                batch = user_data[i:i+batch_size]
                
                # bulk insert 
                await session.run_sync(
                    lambda sync_session: sync_session.bulk_insert_mappings(
                        TrainingUser, batch
                    )
                )
                await session.commit()
                
                print(f"  진행: {min(i+batch_size, total):,}/{total:,}")
            
            print(f" training_users 생성 완료: {len(unique_users):,}명")
    
    finally:
        await engine.dispose()


# import_training_data.py (batch 함수 수정)

async def import_training_ratings_batch(
    session,
    batch_df,
    movie_map,
    user_map,
    batch_num
):
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
        await session.flush()  # 추가! 강제 반영
    
    print(f"  배치 {batch_num} 완료: {len(ratings_data):,}건 (스킵: {skipped:,})")
    return len(ratings_data), skipped


async def import_training_ratings():
    """평점 데이터 임포트 - 세션 관리 개선"""
    df = pd.read_csv("data/users_super_final.csv", encoding="utf8")
    print(f"\n평점 데이터 임포트 시작: {len(df):,}건")
    
    # 매핑 데이터 먼저 로드
    async with AsyncSessionLocal() as map_session:
        movie_result = await map_session.execute(select(Movie.movie_code, Movie.id))
        movie_map = {code: id for code, id in movie_result}
        
        user_result = await map_session.execute(
            select(TrainingUser.original_user_id, TrainingUser.id)
        )
        user_map = {uid: id for uid, id in user_result}
        # map_session은 여기서 자동 종료
    
    print(f"영화 매핑: {len(movie_map):,}개")
    print(f"사용자 매핑: {len(user_map):,}명")
    
    batch_size = 20000
    total_batches = (len(df) + batch_size - 1) // batch_size
    print(f"총 {total_batches}개 배치로 처리")
    
    total_imported = 0
    total_skipped = 0
    
    for i in range(0, len(df), batch_size):
        batch_df = df[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        
        # 각 배치마다 완전히 새로운 세션
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
            # context manager 종료 시 자동으로 세션 닫힘
    
    print(f"\n training_ratings 임포트 완료!")
    print(f"   임포트: {total_imported:,}건")
    print(f"   스킵: {total_skipped:,}건")


async def main():
    """전체 실행"""
    import time
    
    print("=" * 60)
    print("학습 데이터 임포트 시작")
    print("=" * 60)
    
    start_time = time.time()
    
    # Step 1: 사용자 생성
    print("\n[Step 1/2] Training Users 생성")
    await import_training_users()
    
    # Step 2: 평점 임포트
    print("\n[Step 2/2] Training Ratings 임포트")
    await import_training_ratings()
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f" 모든 데이터 임포트 완료! (소요 시간: {elapsed/60:.1f}분)")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())