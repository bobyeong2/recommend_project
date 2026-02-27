import asyncio
from sqlalchemy import select, func, and_, text
from app.core.database import AsyncSessionLocal
from app.models.movie import Movie
from app.models.training import TrainingUser, TrainingRating

async def validate_ratings():
    """
    평점 데이터 검증
    """
    
    print("=" * 70)
    print("평점 데이터 검증 시작")
    print("=" * 70)

    async with AsyncSessionLocal() as session :
        
        # 기본 통계
        print("\n[1] 기본 통계")
        print("-" * 70)
        
        total_ratings = await session.scalar(
            select(
                func.count(TrainingRating.id)
            )
        )
        
        print(f"  총 평점 수: {total_ratings:,}건")
        
        total_users = await session.scalar(
            select(
                func.count(TrainingUser.id)
            )
        )
        
        print(f"  총 사용자: {total_users:,}명")
        
        total_movies = await session.scalar(
            select(func.count(Movie.id))
        )
        
        print(f"  총 영화: {total_movies:,}개")
        
        avg_ratings_per_user = total_ratings / total_users if total_users > 0 else 0
        print(f"  평균 평점/사용자: {avg_ratings_per_user:.1f}개")
        
        # 평점 범위 검증
        print("\n[2] 평점 범위 검증 (1-10)")
        print("-" * 70)

        invalid_ratings = await session.scalar(
            select(
                func.count(TrainingRating.id)
            ).where(
                (TrainingRating.rating < 1) | (TrainingRating.rating > 10)
            )
        )
        
        if invalid_ratings == 0:
            print(f"  모든 평점이 유효 범위 (1-10) 내에 있습니다")
        else:
            print(f"  유효하지 않은 평점: {invalid_ratings:,}건")
        
        # 평점 분포
        print("\n[3] 평점 분포")
        print("-" * 70)
        
        distribution = await session.execute(
            select(
                TrainingRating.rating,
                func.count(TrainingRating.id).label('count')
            ).group_by(TrainingRating.rating).order_by(TrainingRating.rating)
        )
        
        total_for_pct = total_ratings
        for rating, count in distribution:
            percentage = (count / total_for_pct * 100) if total_for_pct > 0 else 0
            bar = "█" * int(percentage / 2)
            print(f"  {rating:2d}점: {count:10,}건 ({percentage:5.2f}%) {bar}")
            
        # NULL 값 검증
        print("\n[4] NULL 값 검증")
        print("-" * 70)
        
        null_user = await session.scalar(
            select(
                func.count(TrainingRating.id)
            ).where(
                TrainingRating.training_user_id.is_(None)
            )
        )
        
        null_movie = await session.scalar(
            select(
                func.count(TrainingRating.id)
            ).where(
                TrainingRating.movie_id.is_(None)
            )
        )
        
        null_rating = await session.scalar(
            select(
                func.count(TrainingRating.id)
            ).where(
                TrainingRating.rating.is_(None)
            )
        )
        
        if null_user == 0 and null_movie == 0 and null_rating == 0:
            print(f"  NULL 값 없음")
        else:
            print(f"  NULL 사용자: {null_user:,}건")
            print(f"  NULL 영화: {null_movie:,}건")
            print(f"  NULL 평점: {null_rating:,}건")
        
        
        # 5. 외래키 무결성 검증 (레코드 확인)
        print("\n[5] 외래키 무결성 검증")
        print("-" * 70)
        
        # 존재하지 않는 사용자를 참조하는 평점
        orphan_users = await session.scalar(
            select(func.count(TrainingRating.id))
            .select_from(TrainingRating)
            .outerjoin(TrainingUser, TrainingRating.training_user_id == TrainingUser.id)
            .where(TrainingUser.id.is_(None))
        )
        
        # 존재하지 않는 영화를 참조하는 평점
        orphan_movies = await session.scalar(
            select(func.count(TrainingRating.id))
            .select_from(TrainingRating)
            .outerjoin(Movie, TrainingRating.movie_id == Movie.id)
            .where(Movie.id.is_(None))
        )
        
        if orphan_users == 0 and orphan_movies == 0:
            print(f"  모든 외래키가 유효합니다")
        else:
            print(f"  고아 평점 (존재하지 않는 사용자): {orphan_users:,}건")
            print(f"  고아 평점 (존재하지 않는 영화): {orphan_movies:,}건")
        
        
        # 6. 중복 평점 검증 (샘플링으로 빠르게)
        print("\n[6] 중복 평점 검증 (샘플 10만 건)")
        print("-" * 70)
        
        # 전체 검증은 너무 오래 걸리므로 샘플링
        duplicates = await session.execute(
            select(
                TrainingRating.training_user_id,
                TrainingRating.movie_id,
                func.count(TrainingRating.id).label('count')
            )
            .where(TrainingRating.id <= 100000)
            .group_by(TrainingRating.training_user_id, TrainingRating.movie_id)
            .having(func.count(TrainingRating.id) > 1)
        )
        
        duplicate_list = list(duplicates)
        if len(duplicate_list) == 0:
            print(f"  중복 평점 없음 (샘플 검사)")
        else:
            print(f"  중복 평점 발견: {len(duplicate_list):,}개 조합 (샘플 검사)")
            print(f"  (같은 사용자가 같은 영화에 여러 평점)")
        
        
        # 7. 데이터 샘플 조회
        print("\n[7] 데이터 샘플 (최근 5개)")
        print("-" * 70)
        
        samples = await session.execute(
            select(
                TrainingRating.id,
                TrainingUser.original_user_id,
                Movie.title,
                TrainingRating.rating
            )
            .join(TrainingUser, TrainingRating.training_user_id == TrainingUser.id)
            .join(Movie, TrainingRating.movie_id == Movie.id)
            .order_by(TrainingRating.id.desc())
            .limit(5)
        )
        
        for rating_id, user_id, movie_title, rating in samples:
            print(f"  ID:{rating_id:8d} | {user_id:12s} | {rating:2d}점 | {movie_title}")
        
        
        # 8. 영화별 평점 통계 (상위 10개)
        print("\n[8] 평점이 많은 영화 TOP 10")
        print("-" * 70)
        
        top_movies = await session.execute(
            select(
                Movie.title,
                func.count(TrainingRating.id).label('rating_count'),
                func.avg(TrainingRating.rating).label('avg_rating')
            )
            .join(TrainingRating, Movie.id == TrainingRating.movie_id)
            .group_by(Movie.id, Movie.title)
            .order_by(func.count(TrainingRating.id).desc())
            .limit(10)
        )
        
        for idx, (title, count, avg) in enumerate(top_movies, 1):
            print(f"  {idx:2d}. {title:40s} | {count:6,}개 | 평균 {avg:.1f}점")
        
        
        # 9. 사용자별 평점 통계 (간소화 - 빠른 버전)
        print("\n[9] 사용자 평점 활동 통계")
        print("-" * 70)
        
        # 간단한 통계만 표시
        print(f"  총 평점: {total_ratings:,}건")
        print(f"  총 사용자: {total_users:,}명")
        print(f"  평균 평점/사용자: {avg_ratings_per_user:.1f}개")
        
        # 가장 활발한 사용자 TOP 5
        print("\n  가장 활발한 사용자 TOP 5:")
        top_users = await session.execute(
            text("""
                SELECT 
                    tu.original_user_id,
                    COUNT(*) as rating_count
                FROM training_ratings tr
                JOIN training_users tu ON tr.training_user_id = tu.id
                GROUP BY tu.id, tu.original_user_id
                ORDER BY rating_count DESC
                LIMIT 5
            """)
        )
        
        for user_id, count in top_users:
            print(f"    - {user_id}: {count:,}개 평점")
    
    
    print("\n" + "=" * 70)
    print("검증 완료!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(validate_ratings())