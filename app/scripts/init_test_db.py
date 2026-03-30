import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import asyncio
from sqlalchemy import text
from app.core.database import engine

async def init_db():
    async with engine.begin() as conn:
        # movies 테이블
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS movies (
                id INT PRIMARY KEY,
                movie_code VARCHAR(50),
                title VARCHAR(200),
                original_title VARCHAR(200),
                overview TEXT,
                genres VARCHAR(200),
                runtime INT,
                release_date DATE,
                poster_path VARCHAR(200),
                mean_rating FLOAT,
                popularity FLOAT,
                created_at DATE,
                updated_at DATE
            )
        """))
        
        # 테스트 데이터 (genres를 string으로)
        await conn.execute(text("""
            INSERT INTO movies (id, title, genres, mean_rating, popularity, created_at, updated_at) VALUES
            (1, '다크 나이트', '액션|범죄|드라마', 8.5, 100.0, CURDATE(), CURDATE()),
            (2, '인셉션', '액션|SF|스릴러', 8.8, 95.0, CURDATE(), CURDATE()),
            (3, '인터스텔라', 'SF|드라마|모험', 8.6, 90.0, CURDATE(), CURDATE()),
            (4, '펄프 픽션', '범죄|드라마', 8.9, 85.0, CURDATE(), CURDATE()),
            (5, '포레스트 검프', '드라마|로맨스', 8.8, 80.0, CURDATE(), CURDATE()),
            (6, '매트릭스', '액션|SF', 8.7, 88.0, CURDATE(), CURDATE()),
            (7, '쇼생크 탈출', '드라마', 9.3, 92.0, CURDATE(), CURDATE()),
            (8, '반지의 제왕', '판타지|모험|드라마', 8.9, 87.0, CURDATE(), CURDATE()),
            (9, '타이타닉', '드라마|로맨스', 7.8, 83.0, CURDATE(), CURDATE()),
            (10, '아바타', 'SF|액션|모험', 7.8, 89.0, CURDATE(), CURDATE()),
            (11, '어벤져스', '액션|SF|모험', 8.0, 91.0, CURDATE(), CURDATE()),
            (12, '글래디에이터', '액션|드라마|역사', 8.5, 84.0, CURDATE(), CURDATE()),
            (13, '라이언 일병 구하기', '드라마|전쟁', 8.6, 82.0, CURDATE(), CURDATE()),
            (14, '센과 치히로의 행방불명', '애니메이션|판타지|모험', 8.6, 81.0, CURDATE(), CURDATE()),
            (15, '기생충', '드라마|스릴러|코미디', 8.6, 93.0, CURDATE(), CURDATE()),
            (16, '조커', '범죄|드라마|스릴러', 8.4, 94.0, CURDATE(), CURDATE()),
            (17, '겨울왕국', '애니메이션|모험|코미디', 7.4, 86.0, CURDATE(), CURDATE()),
            (18, '스파이더맨', '액션|SF|모험', 7.3, 85.0, CURDATE(), CURDATE()),
            (19, '헝거게임', 'SF|액션|스릴러', 7.2, 79.0, CURDATE(), CURDATE()),
            (20, '트와일라잇', '판타지|드라마|로맨스', 5.2, 78.0, CURDATE(), CURDATE())
        """))
        
        # movie_stats
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS movie_stats (
                movie_id INT PRIMARY KEY,
                avg_rating FLOAT,
                rating_count INT,
                updated_at DATETIME
            )
        """))
        
        await conn.execute(text("""
            INSERT INTO movie_stats (movie_id, avg_rating, rating_count, updated_at) VALUES
            (1, 8.5, 1500, NOW()), (2, 8.8, 1300, NOW()), (3, 8.6, 1200, NOW()),
            (4, 8.9, 1100, NOW()), (5, 8.8, 1000, NOW()), (6, 8.7, 950, NOW()),
            (7, 9.3, 1800, NOW()), (8, 8.9, 1400, NOW()), (9, 7.8, 900, NOW()),
            (10, 7.8, 850, NOW()), (11, 8.0, 1600, NOW()), (12, 8.5, 800, NOW()),
            (13, 8.6, 750, NOW()), (14, 8.6, 700, NOW()), (15, 8.6, 1700, NOW()),
            (16, 8.4, 1550, NOW()), (17, 7.4, 650, NOW()), (18, 7.3, 600, NOW()),
            (19, 7.2, 550, NOW()), (20, 5.2, 500, NOW())
        """))
        
        # users
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50),
                email VARCHAR(100),
                hashed_password VARCHAR(200),
                full_name VARCHAR(100),
                is_active TINYINT(1),
                is_verified TINYINT(1),
                role VARCHAR(20),
                created_at DATE,
                updated_at DATE,
                last_login_at DATE
            )
        """))
        
        # user_ratings
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_ratings (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                movie_id INT,
                rating FLOAT,
                created_at DATE,
                updated_at DATE
            )
        """))
    
    print("✓ Test DB initialized")

if __name__ == "__main__":
    asyncio.run(init_db())
