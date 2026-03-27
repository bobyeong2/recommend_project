"""
CI/CD용 테스트 DB 초기화 - 더미 데이터 직접 생성
"""
import asyncio
import os
from sqlalchemy import text
from app.core.database import engine

async def init_test_database():
    """테스트 DB 초기화 및 더미 데이터 생성"""
    print("=" * 70)
    print("테스트 DB 초기화 시작...")
    print("=" * 70)
    
    async with engine.begin() as conn:
        await conn.execute(text("CREATE DATABASE IF NOT EXISTS bob_movie_db_test"))
        await conn.execute(text("USE bob_movie_db_test"))
        
        # movies 테이블
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
        
        # users 테이블
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                hashed_password VARCHAR(255) NOT NULL,
                full_name VARCHAR(100),
                is_active BOOLEAN DEFAULT TRUE,
                is_verified BOOLEAN DEFAULT FALSE,
                role VARCHAR(20) DEFAULT 'user',
                created_at DATE NOT NULL,
                updated_at DATE NOT NULL,
                last_login_at DATETIME,
                INDEX idx_email (email),
                INDEX idx_username (username)
            )
        """))
        
        # user_ratings 테이블
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_ratings (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                movie_id INT NOT NULL,
                rating DECIMAL(3,1) NOT NULL,
                source VARCHAR(50) DEFAULT 'user',
                created_at DATE NOT NULL,
                updated_at DATE NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (movie_id) REFERENCES movies(id) ON DELETE CASCADE,
                UNIQUE KEY unique_user_movie (user_id, movie_id),
                INDEX idx_user_id (user_id),
                INDEX idx_movie_id (movie_id)
            )
        """))
        
        # movie_stats 테이블
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
        
        # 더미 영화 데이터
        dummy_movies = [
            (1, "영화1", "드라마|로맨스", 2020, "감독1", "배우1", "줄거리1", None, None, None),
            (2, "영화2", "액션|스릴러", 2021, "감독2", "배우2", "줄거리2", None, None, None),
            (3, "영화3", "코미디", 2019, "감독3", "배우3", "줄거리3", None, None, None),
            (4, "영화4", "SF|판타지", 2022, "감독4", "배우4", "줄거리4", None, None, None),
            (5, "영화5", "드라마", 2020, "감독5", "배우5", "줄거리5", None, None, None),
            (6, "영화6", "액션", 2021, "감독6", "배우6", "줄거리6", None, None, None),
            (7, "영화7", "로맨스", 2019, "감독7", "배우7", "줄거리7", None, None, None),
            (8, "영화8", "스릴러", 2022, "감독8", "배우8", "줄거리8", None, None, None),
            (9, "영화9", "코미디|로맨스", 2020, "감독9", "배우9", "줄거리9", None, None, None),
            (10, "영화10", "드라마|액션", 2021, "감독10", "배우10", "줄거리10", None, None, None),
            (11, "영화11", "SF", 2019, "감독11", "배우11", "줄거리11", None, None, None),
            (12, "영화12", "판타지", 2022, "감독12", "배우12", "줄거리12", None, None, None),
            (13, "영화13", "액션|SF", 2020, "감독13", "배우13", "줄거리13", None, None, None),
            (14, "영화14", "드라마", 2021, "감독14", "배우14", "줄거리14", None, None, None),
            (15, "영화15", "코미디", 2019, "감독15", "배우15", "줄거리15", None, None, None),
            (16, "영화16", "로맨스|드라마", 2022, "감독16", "배우16", "줄거리16", None, None, None),
            (17, "영화17", "액션", 2020, "감독17", "배우17", "줄거리17", None, None, None),
            (18, "영화18", "스릴러|액션", 2021, "감독18", "배우18", "줄거리18", None, None, None),
            (19, "영화19", "SF|액션", 2019, "감독19", "배우19", "줄거리19", None, None, None),
            (20, "영화20", "드라마|로맨스", 2022, "감독20", "배우20", "줄거리20", None, None, None),
        ]
        
        await conn.execute(text("""
            INSERT IGNORE INTO movies 
            (id, title, genres, year, director, actors, plot, poster_url, naver_code, daum_code)
            VALUES 
            (:id, :title, :genres, :year, :director, :actors, :plot, :poster_url, :naver_code, :daum_code)
        """), [
            {
                "id": m[0], "title": m[1], "genres": m[2], "year": m[3],
                "director": m[4], "actors": m[5], "plot": m[6], 
                "poster_url": m[7], "naver_code": m[8], "daum_code": m[9]
            } for m in dummy_movies
        ])
        
        # movie_stats 더미 데이터
        await conn.execute(text("""
            INSERT IGNORE INTO movie_stats (movie_id, avg_rating, rating_count)
            VALUES 
            (1, 8.5, 100), (2, 7.8, 80), (3, 8.2, 90),
            (4, 7.5, 70), (5, 8.8, 120), (6, 7.9, 85),
            (7, 8.1, 95), (8, 7.6, 75), (9, 8.3, 88),
            (10, 7.7, 82), (11, 8.0, 90), (12, 7.4, 68),
            (13, 8.6, 110), (14, 7.8, 80), (15, 8.2, 92),
            (16, 7.5, 72), (17, 8.4, 98), (18, 7.9, 86),
            (19, 8.1, 94), (20, 7.7, 78)
        """))
        
        print("테스트 DB 초기화 완료 (20개 영화, users/user_ratings 테이블)")

async def main():
    try:
        await init_test_database()
    finally:
        await engine.dispose()

if __name__ == "__main__":
    os.environ["TESTING"] = "true"
    asyncio.run(main())
