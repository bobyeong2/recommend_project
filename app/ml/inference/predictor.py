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

        # ---------------------------
        # 1. 실행 모드 결정 (핵심)
        # ---------------------------
        self.enabled = os.getenv("SKIP_MODEL_LOAD") != "true"

        # fallback 기본값 (항상 먼저 세팅)
        self.global_mean = 5.0

        if not self.enabled:
            logger.info("SKIP_MODEL_LOAD=true, 모델 로드 생략")

            self.service_user_mapping = {}
            self.user_mapping = {}
            self.item_mapping = {}

            self._initialized = True
            return

        # ---------------------------
        # 2. 모델 로드
        # ---------------------------
        import torch
        from app.ml.models.ncf import NCF

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=self.device)

        self.n_users = checkpoint["n_users"]
        self.n_items = checkpoint["n_items"]
        self.user_mapping = checkpoint["user_mapping"]
        self.item_mapping = checkpoint["item_mapping"]
        self.service_user_mapping = checkpoint.get("service_user_mapping", {})

        self.idx_to_item = {idx: iid for iid, idx in self.item_mapping.items()}
        self.idx_to_user = {idx: uid for uid, idx in self.user_mapping.items()}

        config = checkpoint["config"]

        self.model = NCF(
            n_users=self.n_users,
            n_items=self.n_items,
            embedding_dim=config["embedding_dim"],
            mlp_layers=config["mlp_layers"],
            dropout=config["dropout"],
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # 여기만 실제 값 사용
        self.global_mean = checkpoint.get("rmse", 5.5)

        self._initialized = True

        logger.info("✓ 모델 로드 완료")
        logger.info(f"  - Device: {self.device}")
        logger.info(f"  - RMSE: {checkpoint['rmse']:.4f}")
        logger.info(f"  - Users: {self.n_users:,}, Items: {self.n_items:,}")

        if self.service_user_mapping:
            logger.info(f"  - Service users: {len(self.service_user_mapping)} (NCF enabled)")
        else:
            logger.info("  - Service users: cold start only")

    # ---------------------------
    # 공통 fallback
    # ---------------------------
    def _fallback(self, movie_ids: List[int]) -> Dict[int, float]:
        return {mid: self.global_mean for mid in movie_ids}

    # ---------------------------
    # user resolve
    # ---------------------------
    def _resolve_user_idx(self, user_id: int):
        if user_id in self.service_user_mapping:
            return self.service_user_mapping[user_id]
        if user_id in self.user_mapping:
            return self.user_mapping[user_id]
        return None

    # ---------------------------
    # 핵심 predict
    # ---------------------------
    def predict(self, user_id: int, movie_ids: List[int]) -> Dict[int, float]:

        # 1. 모델 비활성 or 없음
        if not self.enabled or not hasattr(self, "model"):
            return self._fallback(movie_ids)

        import torch

        self.model.eval()
        predictions = {}

        user_idx = self._resolve_user_idx(user_id)
        if user_idx is None:
            return self._fallback(movie_ids)

        warm_items = []
        cold_items = []

        for movie_id in movie_ids:
            if movie_id in self.item_mapping:
                warm_items.append((movie_id, self.item_mapping[movie_id]))
            else:
                cold_items.append(movie_id)

        # cold → fallback
        for movie_id in cold_items:
            predictions[movie_id] = self.global_mean

        # warm → 모델
        if warm_items:
            with torch.no_grad():
                movie_ids_batch = [mid for mid, _ in warm_items]
                item_indices = [idx for _, idx in warm_items]

                user_tensor = torch.LongTensor([user_idx] * len(item_indices)).to(self.device)
                item_tensor = torch.LongTensor(item_indices).to(self.device)

                preds = self.model(user_tensor, item_tensor).cpu().numpy()
                preds = np.clip(preds, 1.0, 10.0)

                for movie_id, pred in zip(movie_ids_batch, preds):
                    predictions[movie_id] = float(pred)

        return predictions

    # ---------------------------
    # 기본 추천
    # ---------------------------
    def recommend(self, user_id: int, candidate_movie_ids: List[int], top_k: int = 10) -> List[Dict]:
        predictions = self.predict(user_id, candidate_movie_ids)
        sorted_movies = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"movie_id": movie_id, "predicted_rating": rating} for movie_id, rating in sorted_movies]

    # ---------------------------
    # MMR 다양성
    # ---------------------------
    def apply_mmr_diversity(self, candidates: List[Dict], top_k: int = 10, lambda_param: float = 0.7) -> List[Dict]:
        if not candidates or len(candidates) <= top_k:
            return candidates[:top_k]

        def parse_genres(genres):
            if not genres:
                return set()
            if isinstance(genres, str):
                return set(genres.split(","))
            return set(genres)

        def jaccard_similarity(g1, g2):
            if not g1 or not g2:
                return 0.0
            return len(g1 & g2) / len(g1 | g2)

        for c in candidates:
            c["_genres_set"] = parse_genres(c.get("genres", ""))

        scores = [c["predicted_rating"] for c in candidates]
        min_s, max_s = min(scores), max(scores)
        denom = max_s - min_s if max_s > min_s else 1.0

        for c in candidates:
            c["_norm_score"] = (c["predicted_rating"] - min_s) / denom

        selected = []
        remaining = list(range(len(candidates)))

        best_idx = max(remaining, key=lambda i: candidates[i]["_norm_score"])
        selected.append(candidates[best_idx])
        remaining.remove(best_idx)

        while len(selected) < top_k and remaining:
            mmr_scores = []

            for idx in remaining:
                cand = candidates[idx]
                relevance = cand["_norm_score"]
                max_sim = max(jaccard_similarity(cand["_genres_set"], s["_genres_set"]) for s in selected)
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                mmr_scores.append((idx, mmr))

            best_idx, _ = max(mmr_scores, key=lambda x: x[1])
            selected.append(candidates[best_idx])
            remaining.remove(best_idx)

        for item in selected:
            item.pop("_genres_set", None)
            item.pop("_norm_score", None)

        return selected

    # ---------------------------
    # Hybrid (핵심 개선)
    # ---------------------------
    def recommend_hybrid(
        self,
        user_id: int,
        candidate_movie_ids: List[int],
        collaborative_scores: Dict[int, float],
        top_k: int = 10,
        ncf_weight: float = 0.7,
    ) -> List[Dict]:

        ncf_predictions = self.predict(user_id, candidate_movie_ids)

        # NCF 비활성 → weight 자동 조정
        if not self.enabled:
            ncf_weight = 0.0

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