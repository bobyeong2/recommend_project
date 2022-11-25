from fastapi import APIRouter
import pandas as pd

router = APIRouter()

from recommend_lib import recom_func

movies =pd.read_csv('/root/08_recommend_system/recommend_web/recommend_lib/data/tmdb_5000_movies.csv')
movies['preprocessed'] = movies['title'].str.lower()
movies['preprocessed'] = movies['preprocessed'].str.replace(" ","")


@router.post("/create")
def create_recommend_items(keyword:str):
    movies_df = movies[['id','title', 'genres', 'vote_average', 'vote_count',
                    'popularity', 'keywords', 'overview']]
    percentile = 0.6
    m = movies_df['vote_count'].quantile(percentile)
    C = movies_df['vote_average'].mean()

    movies_df = recom_func.get_preprocess_data(movies_df)
    genre_mat = recom_func.get_count_vec(movies_df)
    genre_sim_sorted_ind = recom_func.get_cosine_similarity(genre_mat)

    movies_df['weighted_vote'] = movies_df.apply(recom_func.weighted_vote_average,m=m,C=C, axis=1) 

    # 전처리한 타이틀을 사용자 입력에 맞게 검색 // keyword의 띄어쓰기 제거
    keyword = keyword.replace(" ","").lower()
    
    keyword = movies[movies['preprocessed'].str.contains(keyword,na=True)]['title'].reset_index(drop=True)
    if len(keyword) :
        similar_movies = recom_func.find_sim_movie(movies_df, genre_sim_sorted_ind,keyword[0] ,10)
        return similar_movies[['title']],{"searched_keyword":keyword[0]}
    else :
        return {"errmsg":"Not Mathched keyword",
                "state_code":400
                }

    
