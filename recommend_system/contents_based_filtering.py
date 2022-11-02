import pandas as pd
import recom_func

movies =pd.read_csv('./data/tmdb_5000_movies.csv')
# 필요한 데이터 column만 slice
movies_df = movies[['id','title', 'genres', 'vote_average', 'vote_count',
                    'popularity', 'keywords', 'overview']]

'''
가중 평점을 계산하기위한 변수 정의
데이터를 보면 영화 한 편당 평점 개수의 불균형으로 영화 평점이 높은데,
평가 건수가 1개인 경우를 제외하기 위해 평가 건수를 상위 60%만 추출하였음.

'''
percentile = 0.6
m = movies_df['vote_count'].quantile(percentile)
C = movies_df['vote_average'].mean()


'''
추천서비스 실행 순서

1. 데이터 전처리 
2. 전처리된 DataFrame을 count vectorize 적용
3. consine similarity 적용 
4. 가중평점 적용
5. 가중평점을 기준으로 검색 키워드와 유사한 영화 추천

'''
movies_df = recom_func.get_preprocess_data(movies_df)
genre_mat = recom_func.get_count_vec(movies_df)
genre_sim_sorted_ind = recom_func.get_cosine_similarity(genre_mat)


# df.apply 함수를 적용하기 위해 함수 시그니쳐, 인자를 기재한 뒤 열(axis=1)로 적용
movies_df['weighted_vote'] = movies_df.apply(recom_func.weighted_vote_average,m=m,C=C, axis=1) 
# keyword와 비슷한 영화 중 평점이 높은(평가가 좋은) 영화 추천
similar_movies = recom_func.find_sim_movie(movies_df, genre_sim_sorted_ind, 'American Gangster',10)

# result 
print(similar_movies[['title', 'vote_average', 'weighted_vote']])