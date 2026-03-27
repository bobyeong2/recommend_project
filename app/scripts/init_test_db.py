"""
CI/CD용 테스트 DB 초기화 스크립트

운영 DB(bob_movie_db)에서 테스트 DB(bob_movie_db_test)로 필요한 데이터만 복사:
- movies 테이블 (20,228건)
- movie_stats 테이블 (20,228건)

training_ratings는 79M건이므로 CI에서 제외
"""
import asyncio
import os
from sqlalchemy import text
from app.core.database import engine

async def init_test_database():
    """테스트 DB 초기화"""
    print("=" * 70)
    print("테스트 DB 초기화 시작...")
    print("=" * 70)
    
    async with engine.begin() as conn:
        # 1. 테스트 DB 생성 (이미 있으면 무시)
        print("\n[1/5] 테스트 DB 확인...")
        try:
            await conn.execute(text("CREATE DATABASE IF NOT EXISTS bob_movie_db_test"))
            print(" bob_movie_db_test 준비 완료")
        except Exception as e:
            print(f"⚠️ DB 생성 건너뜀: {e}")
        
        # 2. 테스트 DB 선택
        await conn.execute(text("USE bob_movie_db_test"))
        
        # 3. movies 테이블 구조 복사
        print("\n[2/5] movies 테이블 생성...")
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS movies (
                id INT PRIMARY KEY,
                title VARCHAR(255) NOT NULL,
                genres TEXT,
                year INT,
                director VARCHAR(255),
                actors TEXT,
                plot TEXT,
                poster_url VARCHAR(512),
                naver_code VARCHAR(50),
                daum_code VARCHAR(50),
                INDEX idx_title (title),
                INDEX idx_year (year)
            )
        """))
        print(" movies 테이블 생성 완료")
        
        # 4. movie_stats 테이블 구조 복사
        print("\n[3/5] movie_stats 테이블 생성...")
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS movie_stats (
                id INT AUTO_INCREMENT PRIMARY KEY,
                movie_id INT NOT NULL UNIQUE,
                avg_rating DECIMAL(3,2),
                rating_count INT DEFAULT 0,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                FOREIGN KEY (movie_id) REFERENCES movies(id) ON DELETE CASCADE,
                INDEX idx_movie_id (movie_id),
                INDEX idx_avg_rating (avg_rating)
            )
        """))
        print(" movie_stats 테이블 생성 완료")
        
        # 5. 운영 DB에서 데이터 복사
        print("\n[4/5] movies 데이터 복사 중...")
        result = await conn.execute(text("""
            INSERT IGNORE INTO bob_movie_db_test.movies
            SELECT * FROM bob_movie_db.movies
        """))
        print(f" movies 복사 완료: {result.rowcount}건")
        
        print("\n[5/5] movie_stats 데이터 복사 중...")
        result = await conn.execute(text("""
            INSERT IGNORE INTO bob_movie_db_test.movie_stats
            SELECT * FROM bob_movie_db.movie_stats
        """))
        print(f" movie_stats 복사 완료: {result.rowcount}건")
        
        # 6. 통계 확인
        print("\n" + "=" * 70)
        print("테스트 DB 통계:")
        print("=" * 70)
        
        result = await conn.execute(text("SELECT COUNT(*) as cnt FROM movies"))
        movies_count = result.scalar()
        print(f"Movies: {movies_count:,}건")
        
        result = await conn.execute(text("SELECT COUNT(*) as cnt FROM movie_stats"))
        stats_count = result.scalar()
        print(f"Movie Stats: {stats_count:,}건")
        
        print("\n 테스트 DB 초기화 완료!")
        print("=" * 70)

async def main():
    """메인 실행 함수"""
    try:
        await init_test_database()
    except Exception as e:
        print(f"\n 오류 발생: {e}")
        raise
    finally:
        await engine.dispose()

if __name__ == "__main__":
    # 환경변수 설정
    os.environ["TESTING"] = "true"
    os.environ["DATABASE_URL"] = "mysql+aiomysql://root:rootpass@localhost:13306/bob_movie_db"
    
    asyncio.run(main())
