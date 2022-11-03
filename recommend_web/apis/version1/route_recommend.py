from fastapi import APIRouter
import pandas as pd
from pandas import DataFrame,Series

router = APIRouter()

from recommend_sys import recom_func

@router.post("/create")
def create_recommend_items(keyword):
    movies =pd.read_csv('/root/08_recommend_system/recommend_func/data/tmdb_5000_movies.csv')
    movies_df = movies[['id','title', 'genres', 'vote_average', 'vote_count',
                    'popularity', 'keywords', 'overview']]
    percentile = 0.6
    m = movies_df['vote_count'].quantile(percentile)
    C = movies_df['vote_average'].mean()

    movies_df = recom_func.get_preprocess_data(movies_df)
    genre_mat = recom_func.get_count_vec(movies_df)
    genre_sim_sorted_ind = recom_func.get_cosine_similarity(genre_mat)

    movies_df['weighted_vote'] = movies_df.apply(recom_func.weighted_vote_average,m=m,C=C, axis=1) 

    similar_movies = recom_func.find_sim_movie(movies_df, genre_sim_sorted_ind,keyword ,10)

    return similar_movies[['title']]
