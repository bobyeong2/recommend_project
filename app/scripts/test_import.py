# import_movies.py

import asyncio
import pandas as pd
from sqlalchemy import select
from app.core.database import AsyncSessionLocal, engine
from app.models.movie import Movie

async def import_all_movies():
    """
    전체 영화 데이터 임포트 (bulk insert 최적화)
    """
    try:
        # 전체 데이터 읽기
        df = pd.read_csv("data/movies_info_super_final.csv", encoding="utf8")
        print(f"총 영화 데이터: {len(df):,}건")
        
        async with AsyncSessionLocal() as session:
            # bulk insert용 데이터 준비
            movies_data = []
            
            for idx, row in df.iterrows():
                # 장르 파싱 (콤마로 구분된 경우)
                genres = None
                if pd.notna(row.get('genre')):
                    genres = [g.strip() for g in str(row['genre']).split(',')]
                
                movie_dict = {
                    "movie_code": int(row["movie_code"]),
                    "title": row["title"],
                    "original_title": row.get("title_en") if pd.notna(row.get("title_en")) else None,
                    "overview": row.get("story") if pd.notna(row.get("story")) else None,
                    "genres": str(genres) if genres else None,  # JSON은 나중에, 일단 문자열
                    "runtime": int(row["runtime"]) if pd.notna(row.get("runtime")) else None,
                    "release_date": f"{int(row['prd_year'])}-01-01" if pd.notna(row.get("prd_year")) else None,
                    "poster_path": row.get("img_src") if pd.notna(row.get("img_src")) else None,
                    "mean_rating": float(row["mean_rating"]) if pd.notna(row.get("mean_rating")) else None,
                    "popularity": float(row.get("total_count", 0)) if pd.notna(row.get("total_count")) else 0.0
                }
                
                movies_data.append(movie_dict)
                
                # 진행 상황
                if (idx + 1) % 1000 == 0:
                    print(f"  데이터 준비 중: {idx+1:,}/{len(df):,}")
            
            print(f"\n데이터 준비 완료! DB 삽입 시작...")
            
            # bulk insert (10,000개씩)
            batch_size = 10000
            total = len(movies_data)
            
            for i in range(0, total, batch_size):
                batch = movies_data[i:i+batch_size]
                
                await session.run_sync(
                    lambda sync_session: sync_session.bulk_insert_mappings(
                        Movie, batch
                    )
                )
                await session.commit()
                
                print(f"  DB 삽입: {min(i+batch_size, total):,}/{total:,}")
            
            # 확인
            result = await session.execute(select(Movie))
            movies = result.scalars().all()
            print(f"\n DB에 저장된 영화: {len(movies):,}개")
    
    finally:
        await engine.dispose()

if __name__ == "__main__":
    import time
    start = time.time()
    
    asyncio.run(import_all_movies())
    
    elapsed = time.time() - start
    print(f"\n소요 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")