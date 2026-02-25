import torch
import numpy as np
from typing import List, Dict
from pathlib import Path

from app.ml.models.ncf import NCF


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