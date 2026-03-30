import asyncio
import json
from sqlalchemy import text
from app.core.database import engine

async def migrate_genres():
    """
    운영 DB의 genres 컬럼을 JSON → VARCHAR로 마이그레이션
    """
    async with engine.begin() as conn:
        # 1. 기존 데이터 조회
        result = await conn.execute(text("SELECT id, genres FROM movies"))
        movies = result.fetchall()
        
        print(f"총 {len(movies)}개 영화 데이터 변환 시작...")
        
        # 2. genres 컬럼 타입 변경
        await conn.execute(text("ALTER TABLE movies MODIFY COLUMN genres VARCHAR(255)"))
        print("✓ genres 컬럼 타입 변경: JSON → VARCHAR(255)")
        
        # 3. 데이터 변환: JSON 배열 → 파이프 구분 문자열
        for movie_id, genres_json in movies:
            if genres_json:
                try:
                    # JSON 배열을 파이프 구분 문자열로 변환
                    if isinstance(genres_json, str):
                        genres_list = json.loads(genres_json)
                    else:
                        genres_list = genres_json
                    
                    genres_str = "|".join(genres_list)
                    
                    await conn.execute(
                        text("UPDATE movies SET genres = :genres WHERE id = :id"),
                        {"genres": genres_str, "id": movie_id}
                    )
                except Exception as e:
                    print(f"⚠ 영화 ID {movie_id} 변환 실패: {e}")
        
        print(f"✓ {len(movies)}개 영화 genres 데이터 변환 완료")
        print("\n마이그레이션 성공!")

if __name__ == "__main__":
    asyncio.run(migrate_genres())