import torch
import numpy as np
from typing import List, Dict
from pathlib import Path

from app.ml.models.ncf import NCF
import math # 250321추가
from collections import Counter # 250321추가

class MovieRecommender:
    """
    학습된 NCF 모델을 사용한 영화 추천기
    """
    
    _instance = None
    
    def __new__(cls, model_path: str = "models/best_ncf_model.pth"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, model_path: str = "models/best_ncf_model.pth"):
        if self._initialized:
            return
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델 로드
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.n_users = checkpoint["n_users"]
        self.n_items = checkpoint["n_items"]
        self.user_mapping = checkpoint["user_mapping"]
        self.item_mapping = checkpoint["item_mapping"]
        
        # 역매핑
        self.idx_to_item = {idx: iid for iid, idx in self.item_mapping.items()}
        self.idx_to_user = {idx: uid for uid, idx in self.user_mapping.items()}
        
        # 모델 초기화
        config = checkpoint["config"]
        self.model = NCF(
            n_users=self.n_users,
            n_items=self.n_items,
            embedding_dim=config["embedding_dim"],
            mlp_layers=config["mlp_layers"],
            dropout=config["dropout"]
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        # Cold Start용 기본값
        self.global_mean = checkpoint.get('rmse', 5.5)  # 전역 평균 (대략 중간값)
        
        self._initialized = True
        
        print(f"✓ 모델 로드 완료")
        print(f"  - Device: {self.device}")
        print(f"  - RMSE: {checkpoint['rmse']:.4f}")
        print(f"  - Users: {self.n_users:,}, Items: {self.n_items:,}")
    
    def predict(self, user_id: int, movie_ids: List[int]) -> Dict[int, float]:
        """
        특정 영화들에 대한 평점 예측
        
        Cold Start 전략:
        - User 없음 → 전역 평균 반환
        - Item 없음 → 전역 평균 반환
        - 둘 다 있음 → NCF 예측
        """
        
        predictions = {}
        
        # User Cold Start
        if user_id not in self.user_mapping:
            return {mid: self.global_mean for mid in movie_ids}
        
        user_idx = self.user_mapping[user_id]
        
        # Warm items와 Cold items 분리
        warm_items = []
        cold_items = []
        
        for movie_id in movie_ids:
            if movie_id in self.item_mapping:
                warm_items.append((movie_id, self.item_mapping[movie_id]))
            else:
                cold_items.append(movie_id)
        
        # Cold items → 전역 평균
        for movie_id in cold_items:
            predictions[movie_id] = self.global_mean
        
        # Warm items → 배치 예측
        if warm_items:
            with torch.no_grad():
                # 배치로 한 번에 예측
                movie_ids_batch = [mid for mid, _ in warm_items]
                item_indices = [idx for _, idx in warm_items]
                
                batch_size = len(item_indices)
                user_tensor = torch.LongTensor([user_idx] * batch_size).to(self.device)
                item_tensor = torch.LongTensor(item_indices).to(self.device)
                
                # 한 번에 예측
                preds = self.model(user_tensor, item_tensor).cpu().numpy()
                
                # Clipping
                preds = np.clip(preds, 1.0, 10.0)
                
                for movie_id, pred in zip(movie_ids_batch, preds):
                    predictions[movie_id] = float(pred)
        
        return predictions
        
    def recommend(
        self, 
        user_id: int, 
        candidate_movie_ids: List[int],
        top_k: int = 10
    ) -> List[Dict]:
        """
        상위 K개 영화 추천
        """
        
        predictions = self.predict(user_id, candidate_movie_ids)
        
        # 평점 기준 정렬
        sorted_movies = sorted(
            predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return [
            {"movie_id": movie_id, "predicted_rating": rating}
            for movie_id, rating in sorted_movies
        ]
        
        
    def recommend_popular(
        self,
        movie_stats: List[Dict], # [{"movie_id": int, "avg_rating": float, "rating_count": int}, ...]
        top_k: int = 10
    ) -> List[Dict]:
        """
        인기 영화 추천 (신규 사용자)
        평점 개수 * 평균 평점으로 정렬
        socre = avg_rating * log(1 +rating_count)
        """
        
        scored_movies = []
        for movie in movie_stats:
            movie_id = movie["movie_id"]
            avg_rating = movie["avg_rating"]
            rating_count = movie["rating_count"]
            
            # 인기 점수 = 평균 평점 * log(1 + 평점 개수)
            popularity_score = avg_rating * math.log(1 + rating_count)
            
            scored_movies.append({
                "movie_id":movie_id,
                "predicted_rating":avg_rating,
                "popularity_score":popularity_score
            })
            
        sorted_movies = sorted(
            scored_movies,
            key=lambda x: x["popularity_score"],
            reverse=True
        )[:top_k]
        
        return [
            {"movie_id": m["movie_id"], "predicted_rating":m["predicted_rating"]}
            for m in sorted_movies
        ]
        
    
    def recommend_content_based(
        self,
        user_rated_movies: List[Dict],  # [{"movie_id": int, "rating": float, "genres": str}, ...]
        candidate_movies: List[Dict],   # [{"movie_id": int, "genres": str}, ...]
        top_k: int = 10
    ) -> List[Dict]:
        """
        CBF 기반 추천 (평점 1 ~ 4개)
        사용자가 평가한 영화의 장르 기반
        """
        # 사용자가 선호하는 장르를 추출함 7점 이상
        genre_preferences = Counter()
        
        for movie in user_rated_movies:
            rating = movie["rating"]
            genres = movie["genres"].split("|") if movie["genres"] else []
            
            if rating >= 7.0:
                weight = rating / 10.0
                for genre in genres:
                    genre_preferences[genre] += weight
                    
        # 정규화를 하기 위한 최대 선호도
        max_preference = max(genre_preferences.values()) if genre_preferences else 1.0
        # 후보 영화에 대한 장르 유사도 계산
        scored_movies = []
        for movie in candidate_movies:
            movie_id = movie["movie_id"]
            genres = movie["genres"].split("|") if movie["genres"] else []

            if genres:
                # 평균 유사도 
                raw_similarity = sum(
                    genre_preferences.get(genre, 0) for genre in genres
                ) / len(genres)
                
                # 0 ~ 1 범위로 정규화
                similarity_score = min(raw_similarity / max_preference, 1.0) if max_preference > 0 else 0
                
                if similarity_score > 0:
                    predicted_rating = 7.0 + (similarity_score * 3.0)  # 7.0~10.0 범위
                    scored_movies.append({
                        "movie_id": movie_id,
                        "predicted_rating":predicted_rating,
                        "similarity_score": similarity_score,
                    })
                    
        sorted_movies = sorted(
            scored_movies,
            key=lambda x: x["similarity_score"],
            reverse=True
        )[: top_k]
        
        return [
            {"movie_id":m["movie_id"],"predicted_rating":m["predicted_rating"]}
            for m in sorted_movies
        ]
        
    def recommend_hybrid(
        self,
        user_id: int,
        candidate_movie_ids: List[int],
        collaborative_scores: Dict[int, float],
        top_k: int = 10,
        ncf_weight: float = 0.7
    ) -> List[Dict]:
        """
        하이브리드 추천 (평점 5개 이상)
        NCF(70%) + CF(30%)
        
        """
        # NCF를 사용한 예측
        ncf_predictions = self.predict(user_id, candidate_movie_ids)
        
        def normalize(scores: Dict[int,float]) -> Dict[int, float]:
            if not scores:
                return {}
            values = list(scores.values())
            min_val, max_val = min(values), max(values)
            if max_val == min_val:
                return {k:0.5 for k in scores}
            
            return {
                k: (v - min_val) / (max_val - min_val)
                for k,v in scores.items()
            }
            
        ncf_norm = normalize(ncf_predictions)
        cf_norm = normalize(collaborative_scores)
        
        # 가중 결합
        cf_weight = 1.0 - ncf_weight
        hybrid_scores = {}
        
        for movie_id in candidate_movie_ids:
            ncf_score = ncf_norm.get(movie_id, 0.5)
            cf_score = cf_norm.get(movie_id, 0.5)
            combined = (ncf_score * ncf_weight) + (cf_score * cf_weight)
            predicted_rating = 1.0 + (combined * 9.0)
            
            hybrid_scores[movie_id] = predicted_rating
            
        sorted_movies = sorted(
            hybrid_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return [
            {"movie_id":movie_id,"predicted_rating":rating}
            for movie_id, rating in sorted_movies
        ]
        