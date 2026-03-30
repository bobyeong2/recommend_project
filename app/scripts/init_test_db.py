import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import asyncio
from sqlalchemy import text
from app.core.database import engine
import json

async def init_test_db():
    async with engine.begin() as conn:
        await conn.execute(text("DROP TABLE IF EXISTS user_ratings"))
        await conn.execute(text("DROP TABLE IF EXISTS movie_stats"))
        await conn.execute(text("DROP TABLE IF EXISTS users"))
        await conn.execute(text("DROP TABLE IF EXISTS movies"))
        
        await conn.execute(text("""
            CREATE TABLE movies (
                id INT PRIMARY KEY,
                movie_code VARCHAR(50),
                title VARCHAR(255),
                original_title VARCHAR(255),
                overview TEXT,
                genres JSON,
                runtime INT,
                release_date DATE,
                poster_path VARCHAR(255),
                mean_rating FLOAT,
                popularity FLOAT,
                created_at DATE,
                updated_at DATE
            )
        """))
        
        await conn.execute(text("""
            CREATE TABLE movie_stats (
                movie_id INT PRIMARY KEY,
                avg_rating FLOAT,
                rating_count INT,
                updated_at DATETIME,
                FOREIGN KEY (movie_id) REFERENCES movies(id)
            )
        """))
        
        await conn.execute(text("""
            CREATE TABLE users (
                id INT PRIMARY KEY AUTO_INCREMENT,
                username VARCHAR(50) UNIQUE,
                email VARCHAR(100) UNIQUE,
                hashed_password VARCHAR(255),
                full_name VARCHAR(100),
                is_active BOOLEAN DEFAULT TRUE,
                is_verified BOOLEAN DEFAULT FALSE,
                role VARCHAR(20) DEFAULT 'user',
                created_at DATE,
                updated_at DATE,
                last_login_at DATE
            )
        """))
        
        await conn.execute(text("""
            CREATE TABLE user_ratings (
                id INT PRIMARY KEY AUTO_INCREMENT,
                user_id INT,
                movie_id INT,
                rating FLOAT,
                created_at DATE,
                updated_at DATE,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (movie_id) REFERENCES movies(id)
            )
        """))
        
        movies_data = [
            (1, 'M001', '테스트 영화 1', 'Test Movie 1', '첫 번째 테스트 영화', json.dumps(['액션', '드라마']), 120, '2020-01-01', '/poster1.jpg', 7.5, 100.0),
            (2, 'M002', '테스트 영화 2', 'Test Movie 2', '두 번째 테스트 영화', json.dumps(['코미디']), 90, '2020-02-01', '/poster2.jpg', 6.8, 80.0),
            (3, 'M003', '테스트 영화 3', 'Test Movie 3', '세 번째 테스트 영화', json.dumps(['스릴러', '미스터리']), 110, '2020-03-01', '/poster3.jpg', 8.0, 120.0),
            (4, 'M004', '테스트 영화 4', 'Test Movie 4', '네 번째 테스트 영화', json.dumps(['로맨스']), 95, '2020-04-01', '/poster4.jpg', 7.2, 90.0),
            (5, 'M005', '테스트 영화 5', 'Test Movie 5', '다섯 번째 테스트 영화', json.dumps(['SF', '액션']), 130, '2020-05-01', '/poster5.jpg', 8.5, 150.0),
            (6, 'M006', '테스트 영화 6', 'Test Movie 6', '여섯 번째 테스트 영화', json.dumps(['드라마']), 105, '2020-06-01', '/poster6.jpg', 7.8, 110.0),
            (7, 'M007', '테스트 영화 7', 'Test Movie 7', '일곱 번째 테스트 영화', json.dumps(['액션', '코미디']), 100, '2020-07-01', '/poster7.jpg', 7.0, 95.0),
            (8, 'M008', '테스트 영화 8', 'Test Movie 8', '여덟 번째 테스트 영화', json.dumps(['공포']), 85, '2020-08-01', '/poster8.jpg', 6.5, 75.0),
            (9, 'M009', '테스트 영화 9', 'Test Movie 9', '아홉 번째 테스트 영화', json.dumps(['다큐멘터리']), 120, '2020-09-01', '/poster9.jpg', 8.2, 130.0),
            (10, 'M010', '테스트 영화 10', 'Test Movie 10', '열 번째 테스트 영화', json.dumps(['애니메이션']), 95, '2020-10-01', '/poster10.jpg', 7.9, 115.0),
            (11, 'M011', '테스트 영화 11', 'Test Movie 11', '11번 영화', json.dumps(['액션']), 100, '2020-11-01', '/poster11.jpg', 7.3, 100.0),
            (12, 'M012', '테스트 영화 12', 'Test Movie 12', '12번 영화', json.dumps(['드라마']), 110, '2020-12-01', '/poster12.jpg', 7.6, 105.0),
            (13, 'M013', '테스트 영화 13', 'Test Movie 13', '13번 영화', json.dumps(['코미디']), 90, '2021-01-01', '/poster13.jpg', 7.1, 85.0),
            (14, 'M014', '테스트 영화 14', 'Test Movie 14', '14번 영화', json.dumps(['스릴러']), 115, '2021-02-01', '/poster14.jpg', 7.7, 110.0),
            (15, 'M015', '테스트 영화 15', 'Test Movie 15', '15번 영화', json.dumps(['로맨스']), 95, '2021-03-01', '/poster15.jpg', 7.4, 90.0),
            (16, 'M016', '테스트 영화 16', 'Test Movie 16', '16번 영화', json.dumps(['SF']), 120, '2021-04-01', '/poster16.jpg', 8.0, 125.0),
            (17, 'M017', '테스트 영화 17', 'Test Movie 17', '17번 영화', json.dumps(['액션', '드라마']), 105, '2021-05-01', '/poster17.jpg', 7.5, 95.0),
            (18, 'M018', '테스트 영화 18', 'Test Movie 18', '18번 영화', json.dumps(['코미디']), 88, '2021-06-01', '/poster18.jpg', 6.9, 80.0),
            (19, 'M019', '테스트 영화 19', 'Test Movie 19', '19번 영화', json.dumps(['공포']), 92, '2021-07-01', '/poster19.jpg', 7.2, 88.0),
            (20, 'M020', '테스트 영화 20', 'Test Movie 20', '20번 영화', json.dumps(['드라마']), 108, '2021-08-01', '/poster20.jpg', 7.8, 112.0),
        ]
        
        for movie in movies_data:
            await conn.execute(text("""
                INSERT INTO movies (id, movie_code, title, original_title, overview, genres, runtime, release_date, poster_path, mean_rating, popularity, created_at, updated_at)
                VALUES (:id, :code, :title, :orig, :overview, :genres, :runtime, :release, :poster, :rating, :pop, CURDATE(), CURDATE())
            """), {
                'id': movie[0], 'code': movie[1], 'title': movie[2], 'orig': movie[3],
                'overview': movie[4], 'genres': movie[5], 'runtime': movie[6],
                'release': movie[7], 'poster': movie[8], 'rating': movie[9], 'pop': movie[10]
            })
        
        stats_data = [
            (1, 7.5, 100), (2, 6.8, 80), (3, 8.0, 120), (4, 7.2, 90), (5, 8.5, 150),
            (6, 7.8, 110), (7, 7.0, 95), (8, 6.5, 75), (9, 8.2, 130), (10, 7.9, 115),
            (11, 7.3, 100), (12, 7.6, 105), (13, 7.1, 85), (14, 7.7, 110), (15, 7.4, 90),
            (16, 8.0, 125), (17, 7.5, 95), (18, 6.9, 80), (19, 7.2, 88), (20, 7.8, 112),
        ]
        
        for stat in stats_data:
            await conn.execute(text("""
                INSERT INTO movie_stats (movie_id, avg_rating, rating_count, updated_at)
                VALUES (:movie_id, :avg, :count, NOW())
            """), {'movie_id': stat[0], 'avg': stat[1], 'count': stat[2]})
    
    print("Test DB initialized successfully")

if __name__ == "__main__":
    asyncio.run(init_test_db())