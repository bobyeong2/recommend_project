import asyncio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from datetime import datetime

from app.core.database import AsyncSessionLocal
from app.ml.models.ncf import NCF, SimpleMF
from app.ml.data.dataset import prepare_data_for_training

async def train_epoch(model, dataloader, criterion, optimizer, device):
    
    # 에폭 학습
    model.train()
    total_loss = 0
    
    for user_ids, item_ids, ratings in tqdm(dataloader, desc="Training"):
        user_ids = user_ids.to(device)
        item_ids = item_ids.to(device)
        ratings = ratings.to(device)
        
        optimizer.zero_grad()
        
        predictions = model(user_ids,item_ids)
        loss = criterion(predictions,ratings)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)


async def evaluate(model, dataloader, device):
    
    # 평가
    model.eval()
    total_mse = 0
    total_mae = 0
    
    with torch.no_grad():
        for user_ids, item_ids, ratings in tqdm(dataloader,desc="Evaluating"):
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            ratings = ratings.to(device)
            
            predictions = model(user_ids,item_ids)
            
            mse = ((predictions - ratings) ** 2 ).mean().item()
            mae = (predictions - ratings).abs().mean().item()
            
            total_mse += mse
            total_mae += mae
            
    rmse = (total_mse / len(dataloader)) ** 0.5
    mae = total_mae / len(dataloader)
    
    return rmse, mae

async def main():
    
    # 하이퍼 파라미터
    EMBEDDING_DIM = 64
    MLP_LAYERS = [128,64,32]
    DROPOUT = 0.2
    BATCH_SIZE = 4096
    EPOCHS = 10
    LEARNING_RATE = 0.001
    SAMPLE_SIZE = 1000000 # None 시 전체 데이터 학습
    
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device : {device}")
    
    # 데이터 준비
    async with AsyncSessionLocal() as session :
        (
            train_dataset,
            test_dataset,
            n_users,
            n_items,
            user_mapping,
            item_mapping
        ) = await prepare_data_for_training(
            session=session, sample_size=SAMPLE_SIZE
        )
        
    # dataloader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    model = NCF(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=EMBEDDING_DIM,
        mlp_layers=MLP_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    print(f"\n모델 파라미터: {sum(p.numel() for p in model.parameters()):,}개")
    
    # Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 학습
    best_rmse = float('inf')
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"{'='*70}")
        
        # Train
        train_loss = await train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        rmse, mae = await evaluate(model, test_loader, device)
        
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test MAE: {mae:.4f}")
        
        # 모델 저장
        if rmse < best_rmse:
            best_rmse = rmse
            
            os.makedirs("saved_models", exist_ok=True)  # 추가
            
            model_path = f"saved_models/ncf_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'rmse': rmse,
                'mae': mae,
                'n_users': n_users,
                'n_items': n_items,
                'user_mapping': user_mapping,
                'item_mapping': item_mapping
            }, model_path)
            
            print(f"모델 저장: {model_path} (RMSE: {rmse:.4f})")
    
    print(f"\n{'='*70}")
    print(f"학습 완료! Best RMSE: {best_rmse:.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
    