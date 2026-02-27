# validate_top_configs.py
import asyncio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import json
from datetime import datetime
import psutil
import os

from app.core.database import AsyncSessionLocal
from app.ml.models.ncf import NCF
from app.ml.data.dataset import prepare_data_for_training


def get_optimal_num_workers():
    """최적 worker 수 계산"""
    cpu_count = os.cpu_count() or 4
    # CPU 코어의 75% 사용 (시스템 안정성)
    return min(int(cpu_count * 0.75), 8)


def print_gpu_memory():
    """GPU 메모리 사용량 출력"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU 메모리: {allocated:.2f}GB 사용 / {reserved:.2f}GB 예약")


async def train_and_evaluate(
    train_loader,
    test_loader,
    n_users,
    n_items,
    config,
    epochs,
    device
):
    """단일 설정 학습 및 평가 (Mixed Precision)"""
    model = NCF(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=config['embedding_dim'],
        mlp_layers=config['mlp_layers'],
        dropout=config['dropout']
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Mixed Precision Scaler
    scaler = GradScaler()
    
    best_rmse = float('inf')
    best_mae = float('inf')
    
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        print_gpu_memory()
        
        # Train
        model.train()
        total_loss = 0
        batch_count = 0
        
        progress_bar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch} 학습",
            ncols=100
        )
        
        for user_ids, item_ids, ratings in progress_bar:
            user_ids = user_ids.to(device, non_blocking=True)
            item_ids = item_ids.to(device, non_blocking=True)
            ratings = ratings.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Mixed Precision Forward
            with autocast():
                predictions = model(user_ids, item_ids)
                loss = criterion(predictions, ratings)
            
            # Mixed Precision Backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            batch_count += 1
            
            # 진행 상황 업데이트
            if batch_count % 100 == 0:
                avg_loss = total_loss / batch_count
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        avg_train_loss = total_loss / len(train_loader)
        print(f"평균 학습 Loss: {avg_train_loss:.4f}")
        
        # Evaluate
        model.eval()
        total_mse = 0
        total_mae = 0
        eval_count = 0
        
        with torch.no_grad():
            for user_ids, item_ids, ratings in tqdm(
                test_loader, 
                desc=f"Epoch {epoch} 평가",
                ncols=100
            ):
                user_ids = user_ids.to(device, non_blocking=True)
                item_ids = item_ids.to(device, non_blocking=True)
                ratings = ratings.to(device, non_blocking=True)
                
                with autocast():
                    predictions = model(user_ids, item_ids)
                
                mse = ((predictions - ratings) ** 2).sum().item()
                mae = (predictions - ratings).abs().sum().item()
                
                batch_size = ratings.size(0)
                total_mse += mse
                total_mae += mae
                eval_count += batch_size
        
        rmse = (total_mse / eval_count) ** 0.5
        mae = total_mae / eval_count
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_mae = mae
        
        print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f} (Best RMSE: {best_rmse:.4f})")
    
    return best_rmse, best_mae


async def main():
    print("=" * 70)
    print("상위 5개 설정 전체 데이터 검증 (GPU 최적화)")
    print("=" * 70)
    
    # 시스템 정보
    print(f"CPU 코어: {os.cpu_count()}")
    print(f"RAM: {psutil.virtual_memory().total / 1024**3:.1f}GB")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # 샘플 Grid Search 상위 5개
    top_configs = [
        {
            'name': '1위 (샘플)',
            'embedding_dim': 32,
            'mlp_layers': [128, 64, 32],
            'dropout': 0.1,
            'learning_rate': 0.001,
            'sample_rmse': 1.6351
        },
        {
            'name': '2위 (샘플)',
            'embedding_dim': 64,
            'mlp_layers': [256, 128, 64],
            'dropout': 0.1,
            'learning_rate': 0.0005,
            'sample_rmse': 1.6356
        },
        {
            'name': '3위 (샘플)',
            'embedding_dim': 32,
            'mlp_layers': [64, 32, 16],
            'dropout': 0.1,
            'learning_rate': 0.001,
            'sample_rmse': 1.6357
        },
        {
            'name': '4위 (샘플)',
            'embedding_dim': 32,
            'mlp_layers': [256, 128, 64],
            'dropout': 0.2,
            'learning_rate': 0.001,
            'sample_rmse': 1.6358
        },
        {
            'name': '5위 (샘플)',
            'embedding_dim': 64,
            'mlp_layers': [256, 128, 64],
            'dropout': 0.1,
            'learning_rate': 0.001,
            'sample_rmse': 1.6362
        }
    ]
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}\n")
    
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
            sample_size=None  # 전체 데이터
        )
    
    print(f"학습 데이터: {len(train_dataset):,}개")
    print(f"테스트 데이터: {len(test_dataset):,}개")
    
    # 최적화 설정
    BATCH_SIZE = 4096
    EPOCHS = 10
    NUM_WORKERS = get_optimal_num_workers()
    
    print(f"\n배치 크기: {BATCH_SIZE}")
    print(f"워커 수: {NUM_WORKERS}")
    print(f"에폭: {EPOCHS}")
    
    # 결과 저장
    results = []
    
    # 각 설정 테스트
    for idx, config in enumerate(top_configs, 1):
        print(f"\n{'='*70}")
        print(f"테스트 {idx}/5: {config['name']}")
        print(f"{'='*70}")
        print(f"embedding_dim: {config['embedding_dim']}")
        print(f"mlp_layers: {config['mlp_layers']}")
        print(f"dropout: {config['dropout']}")
        print(f"learning_rate: {config['learning_rate']}")
        print(f"샘플 RMSE: {config['sample_rmse']}")
        
        # DataLoader (최적화)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,  # GPU 전송 속도 향상
            prefetch_factor=2,  # 미리 2개 배치 로딩
            persistent_workers=True  # 워커 재사용
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
        
        # 학습
        start_time = datetime.now()
        rmse, mae = await train_and_evaluate(
            train_loader=train_loader,
            test_loader=test_loader,
            n_users=n_users,
            n_items=n_items,
            config=config,
            epochs=EPOCHS,
            device=device
        )
        elapsed = (datetime.now() - start_time).total_seconds() / 3600
        
        # 결과 저장
        result = {
            **config,
            'full_data_rmse': rmse,
            'full_data_mae': mae,
            'improvement': config['sample_rmse'] - rmse,
            'training_hours': elapsed
        }
        results.append(result)
        
        print(f"\n결과:")
        print(f"  전체 데이터 RMSE: {rmse:.4f}")
        print(f"  샘플 데이터 RMSE: {config['sample_rmse']:.4f}")
        print(f"  개선도: {result['improvement']:.4f}")
        print(f"  소요 시간: {elapsed:.2f}시간")
        
        # 메모리 정리
        torch.cuda.empty_cache()
    
    # 최종 결과
    print("\n" + "=" * 70)
    print("전체 데이터 결과 비교")
    print("=" * 70)
    
    for idx, result in enumerate(results, 1):
        print(f"\n{idx}. {result['name']}")
        print(f"   샘플 RMSE: {result['sample_rmse']:.4f}")
        print(f"   전체 RMSE: {result['full_data_rmse']:.4f}")
        print(f"   개선도: {result['improvement']:.4f}")
        print(f"   소요 시간: {result['training_hours']:.2f}시간")
    
    # 최고 성능 찾기
    best = min(results, key=lambda x: x['full_data_rmse'])
    
    print("\n" + "=" * 70)
    print("최종 최적 설정 (전체 데이터)")
    print("=" * 70)
    print(f"embedding_dim: {best['embedding_dim']}")
    print(f"mlp_layers: {best['mlp_layers']}")
    print(f"dropout: {best['dropout']}")
    print(f"learning_rate: {best['learning_rate']}")
    print(f"\nBest RMSE: {best['full_data_rmse']:.4f}")
    print(f"Best MAE: {best['full_data_mae']:.4f}")
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"top5_validation_results_{timestamp}.json"
    
    with open(result_file, 'w') as f:
        json.dump({
            'best_config': best,
            'all_results': results
        }, f, indent=2)
    
    print(f"\n결과 저장: {result_file}")


if __name__ == "__main__":
    asyncio.run(main())