"""
INcremental Retraining: 서ㅗ비스 유저 데이터를 포함한 NCF 모델 재학습

기존 모델의 학습된 가중치를 warm start로 사용하고,
서비스 유저 임베딩만 새로 초기화해 fine-tuning함.

사용법 :
    python scripts/retrain_with_service_users.py
    
동작 :
    1. 기존 checkpoint Load (models/best_ncf_model.pth)
    2. DB에서 service user_ratings 조회
    3. training_ratings + user_ratings 통합 데이터 셋 구성
    4. NCF 모델 확장 (n_users += service_users) + warm start
    5. fine-tuning (낮은 lr, 적은 epoch)
    6. 새로운 check point 저장 (models/best_ncf_model.pth 덮어쓰기)
"""

import asyncio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import numpy as np
import os
import copy 
from datetime import datetime
from sqlalchemy import text

from app.core.database import AsyncSessionLocal, engine
from app.ml.models.ncf import NCF
from app.ml.data.dataset import RatingDataset

async def load_service_user_data(session):
    """
    service user_ratings를 DB에서 조회
    Returns: list of (user_id, movie_id, rating)
    
    """
    
    query = text("""
                 SELECT user_id, movie_id, rating
                    FROM user_ratings
                    ORDER BY user_id
                 """)
    result = await session.execute(query)
    rows = result.fetchall()
    print(f"서비스 유저 평점: {len(rows)}건")
    return rows

def build_service_user_mapping(service_ratings, existing_user_count):
    """
    서비스 유저 id -> NCF 내부 인덱스 매핑 생성
    기존 training_user Index (0 ~ existing_user_count - 1) 뒤에 이어 붙임
    
    Args:
        service_ratings: [ (user_id, movie_id, rating) ...]    
        existing_user_count: 기존 모델의 n_users (146340)
        
    Returns:
        service_user_mapping: { service_user_id: ncf_idx }
        
    """
    
    unique_service_user = sorted(set(r[0] for r in service_ratings))
    service_user_mapping = {}
    
    for i, uid in enumerate(unique_service_user):
        service_user_mapping[uid] = existing_user_count + i
        
    print(f"서비스 유저 {len(service_user_mapping)}명 -> idx {existing_user_count} ~ {existing_user_count + len(service_user_mapping) -1 } ")
    return service_user_mapping

def build_service_dataset(service_ratings, service_user_mapping, item_mapping):
    """
    서비스 유저 평점을 RatingDataset으로 변환
    
    item_mapping에 없는 영화는 스킵
    """
    
    user_indices = []
    item_indices = []
    ratings = []
    skipped = 0
    
    for user_id, movie_id, rating in service_ratings:
        if movie_id not in item_mapping:
            skipped +=1
            continue
        
        user_indices.append(service_user_mapping[user_id])
        item_indices.append(item_mapping[movie_id])
        ratings.append(float(rating))
        
        if skipped > 0 :
            print(f" item_mapping에 없는 영화 {skipped}건 스킵")
        
    print(f"서비스 유저 데이터셋: {len(ratings)}건")
    return RatingDataset(
        np.array(user_indices, dtype=np.int64),
        np.array(item_indices, dtype=np.int64),
        np.array(ratings, dtype=np.float32)
    )
    
def expand_model_with_warm_start(old_checkpoint, new_n_users):
    """
    기존 모델 가중치를 복사하고, user 임베딩만 확장
    
    - user_embedding_gmf, user_embedding_mlp: [old_n_users, 64] -> [new_n_users, 64]
        기존 행은 그대로 복사, 새로운 행은 Xavier uniform 초기화
    - item embedding, MLP, predict_layer : shape이 동일해 그대로 복사
    
    """
    
    config = old_checkpoint["config"]
    old_n_users = old_checkpoint["n_users"]
    n_items = old_checkpoint["n_items"]
    old_state = old_checkpoint["model_state_dict"]
    
    print(f"모델 확장 : n_users {old_n_users} -> {new_n_users}")
    
    # 새로운 모델 생성 (확장된 user)
    new_model = NCF(
        n_users=new_n_users,
        n_items=n_items,
        embedding_dim=config["embedding_dim"],
        mlp_layers=config["mlp_layers"],
        dropout=config["dropout"]
    )
    
    new_state = new_model.state_dict()
    
    # 가중치 복사
    for key in old_state:
        if key in ("user_embedding_gmf.weight","user_embedding_mlp.weight"):
            # user embedding: 기존 행 복사 , 나머지는 Xavier 초기화 상태를 유지
            new_state[key][:old_n_users] = old_state[key]
            # 새로운 유저는 NCF._init_weight()에서 이미 Xavier 초기화됨
            print(f" {key}: [{old_n_users}, 64] -> [{new_n_users}, 64] (기존 복사 + 신규 초기화)")
        else:
            # item embedding: MLP, predict_layer 그대로 복사
            new_state[key] = old_state[key]
            
    new_model.load_state_dict(new_state)
    return new_model, config 

async def load_training_sample(session, item_mapping, user_mapping, sample_size=500_000):
    """
    training_ratings에서 샘플을 로드 (fine-tuning용)
 
    전체 79M을 다시 학습하지 않고, 랜덤 샘플로 기존 지식을 유지합니다.
    서비스 유저 데이터와 함께 학습하면 catastrophic forgetting을 방지합니다.
    """
    print(f"training_ratings에서 {sample_size:,}건 샘플링 중...")
 
    query = text(f"""
        SELECT training_user_id, movie_id, rating
        FROM training_ratings
        ORDER BY RAND()
        LIMIT {sample_size}
    """)
    result = await session.execute(query)
    rows = result.fetchall()
 
    user_indices = []
    item_indices = []
    ratings = []
    skipped = 0
 
    for uid, mid, rating in rows:
        if uid not in user_mapping or mid not in item_mapping:
            skipped += 1
            continue
        user_indices.append(user_mapping[uid])
        item_indices.append(item_mapping[mid])
        ratings.append(float(rating))
 
    print(f"  training 샘플: {len(ratings):,}건 (스킵: {skipped})")
 
    return RatingDataset(
        np.array(user_indices, dtype=np.int64),
        np.array(item_indices, dtype=np.int64),
        np.array(ratings, dtype=np.float32)
    )
 
 
async def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
 
    for user_ids, item_ids, ratings in tqdm(dataloader, desc="Training"):
        user_ids = user_ids.to(device)
        item_ids = item_ids.to(device)
        ratings = ratings.to(device)
 
        optimizer.zero_grad()
        predictions = model(user_ids, item_ids)
        loss = criterion(predictions, ratings)
        loss.backward()
        optimizer.step()
 
        total_loss += loss.item()
 
    return total_loss / len(dataloader)
 
 
async def evaluate(model, dataloader, device):
    model.eval()
    total_mse = 0
    total_mae = 0
 
    with torch.no_grad():
        for user_ids, item_ids, ratings in tqdm(dataloader, desc="Evaluating"):
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            ratings = ratings.to(device)
 
            predictions = model(user_ids, item_ids)
 
            mse = ((predictions - ratings) ** 2).mean().item()
            mae = (predictions - ratings).abs().mean().item()
 
            total_mse += mse
            total_mae += mae
 
    rmse = (total_mse / len(dataloader)) ** 0.5
    mae = total_mae / len(dataloader)
    return rmse, mae
 
 
async def main():
    # --- 설정 ---
    MODEL_PATH = "models/best_ncf_model.pth"
    TRAINING_SAMPLE_SIZE = 500_000      # catastrophic forgetting 방지용 샘플
    FINETUNE_LR = 0.0001                # 기존 lr의 1/10 (fine-tuning)
    FINETUNE_EPOCHS = 5
    BATCH_SIZE = 2048
    SERVICE_DATA_OVERSAMPLE = 10        # 서비스 데이터 32건 * 10 = 320건으로 증폭
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"시작 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
 
    # --- Step 1: 기존 checkpoint 로드 ---
    print("\n[Step 1] 기존 모델 로드")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    old_n_users = checkpoint["n_users"]
    n_items = checkpoint["n_items"]
    user_mapping = checkpoint["user_mapping"]
    item_mapping = checkpoint["item_mapping"]
    config = checkpoint["config"]
 
    print(f"  기존 모델: {old_n_users:,} users, {n_items:,} items")
    print(f"  기존 RMSE: {checkpoint['rmse']:.4f}")
 
    # --- Step 2: 서비스 유저 데이터 로드 ---
    print("\n[Step 2] 서비스 유저 데이터 로드")
    async with AsyncSessionLocal() as session:
        service_ratings = await load_service_user_data(session)
 
        if not service_ratings:
            print("서비스 유저 평점이 없습니다. 재학습할 데이터가 없으므로 종료합니다.")
            return
 
        # --- Step 3: 매핑 확장 ---
        print("\n[Step 3] 서비스 유저 매핑 생성")
        service_user_mapping = build_service_user_mapping(service_ratings, old_n_users)
        new_n_users = old_n_users + len(service_user_mapping)
 
        # --- Step 4: 데이터셋 구성 ---
        print("\n[Step 4] 데이터셋 구성")
 
        # 서비스 유저 데이터셋 (oversampling 적용)
        service_dataset = build_service_dataset(service_ratings, service_user_mapping, item_mapping)
        # 서비스 데이터가 적으므로 반복해서 비중을 높임
        service_datasets = [service_dataset] * SERVICE_DATA_OVERSAMPLE
        oversampled_service = ConcatDataset(service_datasets)
        print(f"  서비스 데이터 oversample: {len(service_dataset)} * {SERVICE_DATA_OVERSAMPLE} = {len(oversampled_service)}건")
 
        # training_ratings 샘플
        training_dataset = await load_training_sample(
            session, item_mapping, user_mapping, TRAINING_SAMPLE_SIZE
        )
 
    # 통합 데이터셋
    combined_dataset = ConcatDataset([training_dataset, oversampled_service])
    print(f"  통합 데이터셋: {len(combined_dataset):,}건")
 
    # train/test 분할 (서비스 데이터는 적으므로 전부 train에 포함)
    train_size = int(len(combined_dataset) * 0.9)
    test_size = len(combined_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        combined_dataset, [train_size, test_size]
    )
 
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
 
    # --- Step 5: 모델 확장 + warm start ---
    print("\n[Step 5] 모델 확장 (warm start)")
    model, config = expand_model_with_warm_start(checkpoint, new_n_users)
    model = model.to(device)
 
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  총 파라미터: {total_params:,}개")
 
    # --- Step 6: Fine-tuning ---
    print(f"\n[Step 6] Fine-tuning (lr={FINETUNE_LR}, epochs={FINETUNE_EPOCHS})")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=FINETUNE_LR)
 
    best_rmse = float('inf')
    best_state = None
 
    for epoch in range(1, FINETUNE_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{FINETUNE_EPOCHS} ---")
 
        train_loss = await train_epoch(model, train_loader, criterion, optimizer, device)
        rmse, mae = await evaluate(model, test_loader, device)
 
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Test RMSE:  {rmse:.4f}")
        print(f"  Test MAE:   {mae:.4f}")
 
        if rmse < best_rmse:
            best_rmse = rmse
            best_state = copy.deepcopy(model.state_dict())
            print(f"  -> Best model updated (RMSE: {rmse:.4f})")
 
    # --- Step 7: 저장 ---
    print(f"\n[Step 7] 모델 저장")
 
    # 백업
    backup_path = MODEL_PATH.replace(".pth", f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
    if os.path.exists(MODEL_PATH):
        os.rename(MODEL_PATH, backup_path)
        print(f"  기존 모델 백업: {backup_path}")
 
    # 새 checkpoint 저장 (기존 키 구조 유지 + service_user_mapping 추가)
    new_checkpoint = {
        "epoch": FINETUNE_EPOCHS,
        "model_state_dict": best_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "rmse": best_rmse,
        "mae": mae,
        "config": config,
        "n_users": new_n_users,
        "n_items": n_items,
        "user_mapping": user_mapping,           # 기존 training user mapping 유지
        "item_mapping": item_mapping,
        "service_user_mapping": service_user_mapping,  # 서비스 유저 매핑 추가
    }
 
    torch.save(new_checkpoint, MODEL_PATH)
    print(f"  새 모델 저장: {MODEL_PATH}")
    print(f"  n_users: {old_n_users} -> {new_n_users}")
    print(f"  Best RMSE: {best_rmse:.4f}")
    print(f"  서비스 유저: {len(service_user_mapping)}명 포함")
 
    print(f"\n{'=' * 70}")
    print(f"재학습 완료!")
    print(f"종료 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}")
 
 
if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        # event loop 정리
        loop = asyncio.new_event_loop()
        loop.run_until_complete(engine.dispose())
        loop.close()