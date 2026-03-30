import numpy as np
from typing import List, Dict
from pathlib import Path

import math
from collections import Counter
import logging
import os

logger = logging.getLogger(__name__)

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
            
        if os.getenv("SKIP_MODEL_LOAD") == "true":
            self._initialized = True
            self.service_user_mapping = {}
            self.user_mapping = {}
            self.item_mapping = {}
            logger.info("SKIP_MODEL_LOAD=true, 모델 로드 생략")
            return
        
        # torch와 NCF lazy import
        import torch
        from app.ml.models.ncf import NCF
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.n_users = checkpoint["n_users"]
        self.n_items = checkpoint["n_items"]
        self.user_mapping = checkpoint["user_mapping"]
        self.item_mapping = checkpoint["item_mapping"]
        self.service_user_mapping = checkpoint.get("service_user_mapping",{})
        
        self.idx_to_item = {idx: iid for iid, idx in self.item_mapping.items()}
        self.idx_to_user = {idx: uid for uid, idx in self.user_mapping.items()}
        
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
        
        self.global_mean = checkpoint.get('rmse', 5.5)
        self._initialized = True
        
        logger.info(f"✓ 모델 로드 완료")
        logger.info(f"  - Device: {self.device}")
        logger.info(f"  - RMSE: {checkpoint['rmse']:.4f}")
        logger.info(f"  - Users: {self.n_users:,}, Items: {self.n_items:,}")

        if self.service_user_mapping:
            logger.info(f"  - Service users: {len(self.service_user_mapping)} (NCF enabled)")
        else:
            logger.info(f"  - Service users: cold start only (retrain not yet executed)")
    
    def _resolve_user_idx(self, user_id: int) -> int:
        if user_id in self.service_user_mapping:
            return self.service_user_mapping[user_id]
        if user_id in self.user_mapping:
            return self.user_mapping[user_id]
        return None
    
    def predict(self, user_id: int, movie_ids: List[int]) -> Dict[int, float]:
        import torch
        
        predictions = {}
        user_idx = self._resolve_user_idx(user_id)
        
        if user_idx is None:
            return {mid: self.global_mean for mid in movie_ids}
        
        warm_items = []
        cold_items = []
        
        for movie_id in movie_ids:
            if movie_id in self.item_mapping:
                warm_items.append((movie_id, self.item_mapping[movie_id]))
            else:
                cold_items.append(movie_id)
        
        for movie_id in cold_items:
            predictions[movie_id] = self.global_mean
        
        if warm_items:
            with torch.no_grad():
                movie_ids_batch = [mid for mid, _ in warm_items]
                item_indices = [idx for _, idx in warm_items]
                
                batch_size = len(item_indices)
                user_tensor = torch.LongTensor([user_idx] * batch_size).to(self.device)
                item_tensor = torch.LongTensor(item_indices).to(self.device)
                
                preds = self.model(user_tensor, item_tensor).cpu().numpy()
                preds = np.clip(preds, 1.0, 10.0)
                
                for movie_id, pred in zip(movie_ids_batch, preds):
                    predictions[movie_id] = float(pred)
        
        return predictions
        
    def recommend(self, user_id: int, candidate_movie_ids: List[int], top_k: int = 10) -> List[Dict]:
        predictions = self.predict(user_id, candidate_movie_ids)
        sorted_movies = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"movie_id": movie_id, "predicted_rating": rating} for movie_id, rating in sorted_movies]
        
    def apply_mmr_diversity(self, candidates: List[Dict], top_k: int = 10, lambda_param: float = 0.7) -> List[Dict]:
        if not candidates or len(candidates) <= top_k:
            return candidates[:top_k]
        
        def parse_genres(genres):
            if not genres:
                return set()
            if isinstance(genres, str):
                return set(genres.split(","))
            return set(genres)
        
        def jaccard_similarity(genres1, genres2):
            if not genres1 or not genres2:
                return 0.0
            intersection = len(genres1 & genres2)
            union = len(genres1 | genres2)
            return intersection / union if union > 0 else 0.0
        
        for cand in candidates:
            cand["_genres_set"] = parse_genres(cand.get("genres",""))
            
        scores = [c["predicted_rating"] for c in candidates]
        min_score, max_score = min(scores), max(scores)
        score_range = max_score - min_score if max_score > min_score else 1.0
        
        for cand in candidates:
            cand["_norm_score"] = (cand["predicted_rating"] - min_score) / score_range
    
        selected = []
        remaining_indices = list(range(len(candidates)))
        
        best_idx = max(remaining_indices, key=lambda i: candidates[i]['_norm_score'])
        selected.append(candidates[best_idx])
        remaining_indices.remove(best_idx)
        
        while len(selected) < top_k and remaining_indices:
            mmr_scores = []
            
            for idx in remaining_indices:
                cand = candidates[idx]
                relevance = cand['_norm_score']
                max_sim = max(jaccard_similarity(cand['_genres_set'], s['_genres_set']) for s in selected)
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                mmr_scores.append((idx, mmr))
            
            best_idx, _ = max(mmr_scores, key=lambda x: x[1])
            selected.append(candidates[best_idx])
            remaining_indices.remove(best_idx)
        
        for item in selected:
            item.pop('_genres_set', None)
            item.pop('_norm_score', None)
        
        return selected
    
    def recommend_popular(self, movie_stats: List[Dict], top_k: int = 10) -> List[Dict]:
        scored_movies = []
        for movie in movie_stats:
            movie_id = movie["movie_id"]
            avg_rating = movie["avg_rating"]
            rating_count = movie["rating_count"]
            popularity_score = avg_rating * math.log(1 + rating_count)
            scored_movies.append({"movie_id": movie_id, "predicted_rating": avg_rating, "popularity_score": popularity_score})
            
        sorted_movies = sorted(scored_movies, key=lambda x: x["popularity_score"], reverse=True)[:top_k]
        return [{"movie_id": m["movie_id"], "predicted_rating": m["predicted_rating"]} for m in sorted_movies]
        
    def recommend_content_based(self, user_rated_movies: List[Dict], candidate_movies: List[Dict], top_k: int = 10) -> List[Dict]:
        genre_preferences = Counter()
        
        for movie in user_rated_movies:
            rating = movie["rating"]
            if isinstance(genres, str):
                genres = genres.split("|")
            
            if rating >= 7.0:
                weight = rating / 10.0
                for genre in genres:
                    genre_preferences[genre] += weight
                    
        max_preference = max(genre_preferences.values()) if genre_preferences else 1.0
        scored_movies = []
        
        for movie in candidate_movies:
            movie_id = movie["movie_id"]
            if isinstance(genres, str):
                genres = genres.split("|")

            if genres:
                raw_similarity = sum(genre_preferences.get(genre, 0) for genre in genres) / len(genres)
                similarity_score = min(raw_similarity / max_preference, 1.0) if max_preference > 0 else 0
                
                if similarity_score > 0:
                    predicted_rating = 7.0 + (similarity_score * 3.0)
                    scored_movies.append({"movie_id": movie_id, "predicted_rating": predicted_rating, "similarity_score": similarity_score})
                    
        sorted_movies = sorted(scored_movies, key=lambda x: x["similarity_score"], reverse=True)[:top_k]
        return [{"movie_id": m["movie_id"], "predicted_rating": m["predicted_rating"]} for m in sorted_movies]
        
    def recommend_hybrid(self, user_id: int, candidate_movie_ids: List[int], collaborative_scores: Dict[int, float], top_k: int = 10, ncf_weight: float = 0.7) -> List[Dict]:
        ncf_predictions = self.predict(user_id, candidate_movie_ids)
        
        def normalize(scores: Dict[int, float]) -> Dict[int, float]:
            if not scores:
                return {}
            values = list(scores.values())
            min_val, max_val = min(values), max(values)
            if max_val == min_val:
                return {k: 0.5 for k in scores}
            return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}
            
        ncf_norm = normalize(ncf_predictions)
        cf_norm = normalize(collaborative_scores)
        
        cf_weight = 1.0 - ncf_weight
        hybrid_scores = {}
        
        for movie_id in candidate_movie_ids:
            ncf_score = ncf_norm.get(movie_id, 0.5)
            cf_score = cf_norm.get(movie_id, 0.5)
            combined = (ncf_score * ncf_weight) + (cf_score * cf_weight)
            predicted_rating = 1.0 + (combined * 9.0)
            hybrid_scores[movie_id] = predicted_rating
            
        sorted_movies = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"movie_id": movie_id, "predicted_rating": rating} for movie_id, rating in sorted_movies]
        
    @classmethod
    def reload(cls):
        cls._instance = None
