# scripts/train_final_model.py
import asyncio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler  # 변경
from tqdm import tqdm
from datetime import datetime
import os

from app.core.database import AsyncSessionLocal
from app.ml.models.ncf import NCF
from app.ml.data.dataset import prepare_data_for_training


def get_optimal_num_workers():
    cpu_count = os.cpu_count() or 4
    return min(int(cpu_count * 0.75), 8)


async def main():
    print("=" * 70)
    print("최종 모델 학습")
    print("=" * 70)
    
    # 최적 설정
    BEST_CONFIG = {
        'embedding_dim': 64,
        'mlp_layers': [256, 128, 64],
        'dropout': 0.1,
        'learning_rate': 0.001
    }
    
    BATCH_SIZE = 4096
    EPOCHS = 10
    NUM_WORKERS = get_optimal_num_workers()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"최적 설정: {BEST_CONFIG}\n")
    
    # 전체 데이터 로드
    print("전체 데이터 로딩 중...")
    async with AsyncSessionLocal() as session:
        (
            train_dataset,
            test_dataset,
            n_users,
            n_items,
            user_mapping,
            item_mapping
        ) = await prepare_data_for_training(
            session=session,
            sample_size=None
        )
    
    print(f"학습 데이터: {len(train_dataset):,}개")
    print(f"테스트 데이터: {len(test_dataset):,}개\n")
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    # 모델
    model = NCF(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=BEST_CONFIG['embedding_dim'],
        mlp_layers=BEST_CONFIG['mlp_layers'],
        dropout=BEST_CONFIG['dropout']
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=BEST_CONFIG['learning_rate'])
    scaler = GradScaler('cuda')  # device_type 명시
    
    best_rmse = float('inf')
    best_epoch = 0
    
    # 학습
    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
        
        # Train
        model.train()
        total_loss = 0
        
        for user_ids, item_ids, ratings in tqdm(train_loader, desc="학습", ncols=100):
            user_ids = user_ids.to(device, non_blocking=True)
            item_ids = item_ids.to(device, non_blocking=True)
            ratings = ratings.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # device_type='cuda' 명시
            with autocast(device_type='cuda'):
                predictions = model(user_ids, item_ids)
                loss = criterion(predictions, ratings)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Evaluate
        model.eval()
        total_mse = 0
        total_mae = 0
        count = 0
        
        with torch.no_grad():
            for user_ids, item_ids, ratings in tqdm(test_loader, desc="평가", ncols=100):
                user_ids = user_ids.to(device, non_blocking=True)
                item_ids = item_ids.to(device, non_blocking=True)
                ratings = ratings.to(device, non_blocking=True)
                
                # device_type='cuda' 명시
                with autocast(device_type='cuda'):
                    predictions = model(user_ids, item_ids)
                
                mse = ((predictions - ratings) ** 2).sum().item()
                mae = (predictions - ratings).abs().sum().item()
                
                total_mse += mse
                total_mae += mae
                count += ratings.size(0)
        
        rmse = (total_mse / count) ** 0.5
        mae = total_mae / count
        
        print(f"Loss: {avg_loss:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        # Best model 저장
        if rmse < best_rmse:
            best_rmse = rmse
            best_epoch = epoch
            
            os.makedirs("models", exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'rmse': rmse,
                'mae': mae,
                'config': BEST_CONFIG,
                'n_users': n_users,
                'n_items': n_items,
                'user_mapping': user_mapping,
                'item_mapping': item_mapping
            }, 'models/best_ncf_model.pth')
            
            print(f"✓ Best model 저장 (RMSE: {rmse:.4f})")
    
    print("\n" + "=" * 70)
    print("학습 완료!")
    print("=" * 70)
    print(f"Best Epoch: {best_epoch}")
    print(f"Best RMSE: {best_rmse:.4f}")
    print(f"모델 저장: models/best_ncf_model.pth")


if __name__ == "__main__":
    asyncio.run(main())