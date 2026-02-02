import asyncio
import ast
import numpy as np
from sqlalchemy import text
from datetime import datetime
import pickle
import os

from app.core.database import AsyncSessionLocal


async def extract_movie_features():
    """
    영화 메타데이터에서 Feature 추출
    """
    
    print("=" * 70)
    print("영화 메타데이터 Feature 추출")
    print("=" * 70)
    
    async with AsyncSessionLocal() as session:
        query = text("""
            SELECT 
                id,
                genres,
                runtime,
                release_date,
                mean_rating,
                popularity
            FROM movies
            ORDER BY id
        """)
        
        result = await session.execute(query)
        movies = result.fetchall()
    
    print(f"총 영화: {len(movies):,}개")
    
    # 1. 전체 장르 수집
    all_genres = set()
    parse_errors = 0
    
    for movie in movies:
        if movie.genres and str(movie.genres) != 'null':
            try:
                # JSON 문자열 → Python 리스트
                genre_list = ast.literal_eval(str(movie.genres))
                if isinstance(genre_list, list):
                    # None이나 빈 문자열 제외
                    clean_genres = [g for g in genre_list if g and str(g).strip()]
                    all_genres.update(clean_genres)
            except:
                parse_errors += 1
    
    if parse_errors > 0:
        print(f"⚠ 파싱 실패: {parse_errors}개")
    
    genre_to_idx = {g: i for i, g in enumerate(sorted(all_genres))}
    print(f"고유 장르: {len(genre_to_idx)}개")
    print(f"장르 목록: {sorted(all_genres)}")
    
    # 2. Feature 벡터 생성
    movie_features = {}
    
    runtimes = []
    years = []
    ratings = []
    popularities = []
    
    for movie in movies:
        # 장르 Multi-hot
        genre_vector = np.zeros(len(genre_to_idx))
        if movie.genres and str(movie.genres) != 'null':
            try:
                genre_list = ast.literal_eval(str(movie.genres))
                if isinstance(genre_list, list):
                    for g in genre_list:
                        if g and g in genre_to_idx:
                            genre_vector[genre_to_idx[g]] = 1
            except:
                pass
        
        # 수치형 데이터 수집
        runtime = movie.runtime if movie.runtime else 0
        runtimes.append(runtime)
        
        year = movie.release_date.year if movie.release_date else 2000
        years.append(year)
        
        rating = movie.mean_rating if movie.mean_rating else 5.0
        ratings.append(rating)
        
        popularity = movie.popularity if movie.popularity else 0.0
        popularities.append(popularity)
        
        movie_features[movie.id] = {
            'genre_vector': genre_vector,
            'runtime': runtime,
            'year': year,
            'rating': rating,
            'popularity': popularity
        }
    
    # 3. 정규화 파라미터
    runtime_mean = np.mean(runtimes)
    runtime_std = np.std(runtimes) + 1e-6
    
    year_mean = np.mean(years)
    year_std = np.std(years) + 1e-6
    
    rating_mean = np.mean(ratings)
    rating_std = np.std(ratings) + 1e-6
    
    popularity_mean = np.mean(popularities)
    popularity_std = np.std(popularities) + 1e-6
    
    print(f"\n정규화 파라미터:")
    print(f"  Runtime: mean={runtime_mean:.1f}, std={runtime_std:.1f}")
    print(f"  Year: mean={year_mean:.1f}, std={year_std:.1f}")
    print(f"  Rating: mean={rating_mean:.2f}, std={rating_std:.2f}")
    print(f"  Popularity: mean={popularity_mean:.2f}, std={popularity_std:.2f}")
    
    # 4. Feature 벡터 완성
    feature_dim = len(genre_to_idx) + 4
    
    final_features = {}
    
    for movie_id, feat in movie_features.items():
        runtime_norm = (feat['runtime'] - runtime_mean) / runtime_std
        year_norm = (feat['year'] - year_mean) / year_std
        rating_norm = (feat['rating'] - rating_mean) / rating_std
        popularity_norm = (feat['popularity'] - popularity_mean) / popularity_std
        
        feature_vector = np.concatenate([
            feat['genre_vector'],
            [runtime_norm, year_norm, rating_norm, popularity_norm]
        ])
        
        final_features[movie_id] = feature_vector.astype(np.float32)
    
    print(f"\nFeature 차원: {feature_dim}")
    
    # 5. 저장
    os.makedirs('models', exist_ok=True)
    
    metadata = {
        'features': final_features,
        'genre_to_idx': genre_to_idx,
        'feature_dim': feature_dim,
        'normalization': {
            'runtime_mean': runtime_mean,
            'runtime_std': runtime_std,
            'year_mean': year_mean,
            'year_std': year_std,
            'rating_mean': rating_mean,
            'rating_std': rating_std,
            'popularity_mean': popularity_mean,
            'popularity_std': popularity_std
        }
    }
    
    with open('models/movie_features.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print("\n✓ Feature 저장 완료: models/movie_features.pkl")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(extract_movie_features())