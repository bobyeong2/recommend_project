"""
CI/CD용 테스트 DB 초기화 - 더미 데이터 직접 생성
"""
import asyncio
import os
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

async def init_test_database():
    """테스트 DB 초기화 및 더미 데이터 생성"""
    print("=" * 70)
    print("테스트 DB 초기화 시작...")
    print("=" * 70)
    
    root_url = "mysql+aiomysql://root:rootpass@localhost:3306"
    engine = create_async_engine(root_url, echo=False)
    
    async with engine.begin() as conn:
        await conn.execute(text("CREATE DATABASE IF NOT EXISTS bob_movie_db_test"))
        await conn.execute(text("USE bob_movie_db_test"))
        
        # movies 테이블 (실제 Movie 모델과 일치)
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS movies (
                id INT AUTO_INCREMENT PRIMARY KEY,
                movie_code INT UNIQUE,
                title VARCHAR(500) NOT NULL,
                original_title VARCHAR(500),
                overview TEXT,
                genres JSON,
                runtime INT,
                release_date DATE,
                poster_path VARCHAR(500),
                mean_rating FLOAT,
                popularity FLOAT DEFAULT 0,
                created_at DATE NOT NULL,
                updated_at DATE NOT NULL,
                INDEX idx_movie_code (movie_code),
                INDEX idx_title (title)
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
        
        # 더미 영화 데이터 (실제 스키마에 맞춤)
        dummy_movies = [
            (1001, "기생충", "Parasite", "전원 백수인 기택 가족...", '["드라마", "스릴러"]', 132, "2019-05-30", None, 8.6, 95.5, "2024-01-01", "2024-01-01"),
            (1002, "범죄도시", "The Outlaws", "불법 체류자들을 상대로...", '["액션", "범죄"]', 121, "2017-10-03", None, 7.8, 88.2, "2024-01-01", "2024-01-01"),
            (1003, "극한직업", "Extreme Job", "마약반 형사들이...", '["코미디", "범죄"]', 111, "2019-01-23", None, 8.2, 92.0, "2024-01-01", "2024-01-01"),
            (1004, "명량", "The Admiral", "1597년 정유재란...", '["드라마", "역사"]', 128, "2014-07-30", None, 7.5, 85.0, "2024-01-01", "2024-01-01"),
            (1005, "신과함께", "Along with the Gods", "49일간의 재판...", '["판타지", "드라마"]', 139, "2017-12-20", None, 8.8, 98.0, "2024-01-01", "2024-01-01"),
            (1006, "베테랑", "Veteran", "재벌 3세의 범죄...", '["액션", "범죄"]', 123, "2015-08-05", None, 7.9, 87.5, "2024-01-01", "2024-01-01"),
            (1007, "광해", "Masquerade", "광해군을 대신한...", '["드라마", "역사"]', 131, "2012-09-13", None, 8.1, 89.0, "2024-01-01", "2024-01-01"),
            (1008, "부산행", "Train to Busan", "정체불명의 바이러스...", '["액션", "스릴러"]', 118, "2016-07-20", None, 7.6, 86.0, "2024-01-01", "2024-01-01"),
            (1009, "국제시장", "Ode to My Father", "한국 현대사의...", '["드라마"]', 126, "2014-12-17", None, 8.3, 90.5, "2024-01-01", "2024-01-01"),
            (1010, "7번방의 선물", "Miracle in Cell No.7", "억울한 누명을...", '["코미디", "드라마"]', 127, "2013-01-23", None, 7.7, 88.0, "2024-01-01", "2024-01-01"),
            (1011, "어벤져스", "The Avengers", "지구를 지키기 위해...", '["액션", "SF"]', 143, "2012-04-26", None, 8.0, 94.0, "2024-01-01", "2024-01-01"),
            (1012, "타짜", "Tazza", "치밀한 판돈 싸움...", '["범죄", "드라마"]', 139, "2006-09-28", None, 7.4, 82.0, "2024-01-01", "2024-01-01"),
            (1013, "설국열차", "Snowpiercer", "빙하기가 닥친 지구...", '["액션", "SF"]', 126, "2013-08-01", None, 8.6, 96.0, "2024-01-01", "2024-01-01"),
            (1014, "해운대", "Haeundae", "대한민국 최대 해변...", '["액션", "드라마"]', 120, "2009-07-22", None, 7.8, 85.5, "2024-01-01", "2024-01-01"),
            (1015, "도둑들", "The Thieves", "한국과 마카오...", '["범죄", "액션"]', 135, "2012-07-25", None, 8.2, 91.0, "2024-01-01", "2024-01-01"),
            (1016, "왕의남자", "King and the Clown", "조선시대 광대들...", '["드라마", "역사"]', 119, "2005-12-29", None, 7.5, 83.0, "2024-01-01", "2024-01-01"),
            (1017, "아저씨", "The Man from Nowhere", "전당포를 운영하는...", '["액션", "스릴러"]', 119, "2010-08-04", None, 8.4, 93.5, "2024-01-01", "2024-01-01"),
            (1018, "실미도", "Silmido", "1971년 북파공작...", '["액션", "드라마"]', 135, "2003-12-24", None, 7.9, 87.0, "2024-01-01", "2024-01-01"),
            (1019, "변호인", "The Attorney", "1981년 부산...", '["드라마"]', 127, "2013-12-18", None, 8.1, 90.0, "2024-01-01", "2024-01-01"),
            (1020, "괴물", "The Host", "한강에 나타난...", '["액션", "SF"]', 120, "2006-07-27", None, 7.7, 86.5, "2024-01-01", "2024-01-01"),
        ]
        
        for m in dummy_movies:
            await conn.execute(text("""
                INSERT IGNORE INTO movies 
                (movie_code, title, original_title, overview, genres, runtime, release_date, 
                 poster_path, mean_rating, popularity, created_at, updated_at)
                VALUES 
                (:movie_code, :title, :original_title, :overview, :genres, :runtime, :release_date,
                 :poster_path, :mean_rating, :popularity, :created_at, :updated_at)
            """), {
                "movie_code": m[0], "title": m[1], "original_title": m[2], "overview": m[3],
                "genres": m[4], "runtime": m[5], "release_date": m[6], "poster_path": m[7],
                "mean_rating": m[8], "popularity": m[9], "created_at": m[10], "updated_at": m[11]
            })
        
        # movie_stats 더미 데이터
        await conn.execute(text("""
            INSERT IGNORE INTO movie_stats (movie_id, avg_rating, rating_count)
            SELECT id, mean_rating, FLOOR(popularity) 
            FROM movies 
            LIMIT 20
        """))
        
        print("테스트 DB 초기화 완료 (20개 영화, users/user_ratings 테이블)")
    
    await engine.dispose()

if __name__ == "__main__":
    os.environ["TESTING"] = "true"
    asyncio.run(init_test_database())
