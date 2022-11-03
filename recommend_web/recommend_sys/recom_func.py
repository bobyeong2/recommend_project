from pandas import DataFrame
import warnings; warnings.filterwarnings('ignore')
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_preprocess_data(data : DataFrame):

    data['genres'] = data['genres'].apply(literal_eval)
    data['keywords'] = data['keywords'].apply(literal_eval)

    data['genres'] = data['genres'].apply(lambda x : [ y['name'] for y in x])
    data['keywords'] = data['keywords'].apply(lambda x : [ y['name'] for y in x])

    # CountVectorizer를 적용하기 위해 공백문자로 word 단위가 구분되는 문자열로 변환. 
    data['genres_literal'] = data['genres'].apply(lambda x : (' ').join(x))

    return data

def get_count_vec(processed_data : DataFrame):

    count_vect = CountVectorizer(min_df=0, ngram_range=(1,2))
    count_vect_mat = count_vect.fit_transform(processed_data['genres_literal'])

    return count_vect_mat

def get_cosine_similarity(matrix):
    '''
    Cosine similarity를 적용하는 함수
    return 하기 전 argsort function을 사용해서 데이터를 내림차순으로 정렬한 이후 인덱스 값을 반환
    argsort : (default -> 오름차순, 내림차순 정렬 시 [::-1]사용
    '''
    genre_sim = cosine_similarity(matrix, matrix)
    genre_sim_sorted_ind = genre_sim.argsort()[:, ::-1]

    return genre_sim_sorted_ind

def weighted_vote_average(record,m,C):
    v = record['vote_count']
    R = record['vote_average']
    
    return ( (v/(v+m)) * R ) + ( (m/(m+v)) * C )   

def find_sim_movie(movies_df, sorted_ind, title_name, top_n=10):
    title_movie = movies_df[movies_df['title'] == title_name]
    title_index = title_movie.index.values
    
    # top_n의 2배에 해당하는 쟝르 유사성이 높은 index 추출 
    similar_indexes = sorted_ind[title_index, :(top_n*2)]
    similar_indexes = similar_indexes.reshape(-1)
    
    # 기준 영화 index는 제외
    similar_indexes = similar_indexes[similar_indexes != title_index]
    
    # top_n의 2배에 해당하는 후보군에서 weighted_vote 높은 순으로 top_n 만큼 추출 
    return movies_df.iloc[similar_indexes].sort_values('weighted_vote', ascending=False)[:top_n]
