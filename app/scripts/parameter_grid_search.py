import asyncio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
from datetime import datetime
from itertools import product
import os

from app.core.database import AsyncSessionLocal
from app.ml.models.ncf import NCF
from app.ml.data.dataset import prepare_data_for_training


async def train_and_evaluate(
    train_loader,
    test_loader,
    n_users,
    n_items,
    embedding_dim,
    mlp_layers,
    dropout,
    learning_rate,
    epochs,
    device
):
    """단일 설정으로 학습 및 평가"""
    model = NCF(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=embedding_dim,
        mlp_layers=mlp_layers,
        dropout=dropout
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_rmse = float('inf')
    best_mae = float('inf')
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss = 0
        
        for user_ids, item_ids, ratings in train_loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            ratings = ratings.to(device)
            
            optimizer.zero_grad()
            predictions = model(user_ids, item_ids)
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Evaluate
        model.eval()
        total_mse = 0
        total_mae = 0
        
        with torch.no_grad():
            for user_ids, item_ids, ratings in test_loader:
                user_ids = user_ids.to(device)
                item_ids = item_ids.to(device)
                ratings = ratings.to(device)
                
                predictions = model(user_ids, item_ids)
                
                mse = ((predictions - ratings) ** 2).mean().item()
                mae = (predictions - ratings).abs().mean().item()
                
                total_mse += mse
                total_mae += mae
        
        rmse = (total_mse / len(test_loader)) ** 0.5
        mae = total_mae / len(test_loader)
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_mae = mae
    
    return best_rmse, best_mae


async def main():
    print("=" * 70)
    print("Grid Search 시작")
    print("=" * 70)
    
    # ==========================================
    # 옵션 선택
    # ==========================================
    print("\nGrid Search 옵션을 선택하세요:")
    print("1. 빠른 Grid Search (16개 조합, 5-6시간)")
    print("2. 중간 Grid Search (54개 조합, 18시간)")
    print("3. 전체 Grid Search (81개 조합, 27시간)")
    
    choice = input("\n선택 (1/2/3): ").strip()
    
    if choice == '1':
        # 빠른 Grid Search
        embedding_dims = [64, 128]
        mlp_layers_list = [[128, 64, 32], [256, 128, 64]]
        dropouts = [0.2, 0.3]
        learning_rates = [0.001, 0.002]
        search_name = "fast"
    elif choice == '2':
        # 중간 Grid Search
        embedding_dims = [32, 64, 128]
        mlp_layers_list = [[64, 32, 16], [128, 64, 32], [256, 128, 64]]
        dropouts = [0.2, 0.3]
        learning_rates = [0.0005, 0.001, 0.002]
        search_name = "medium"
    elif choice == '3':
        # 전체 Grid Search
        embedding_dims = [32, 64, 128]
        mlp_layers_list = [[64, 32, 16], [128, 64, 32], [256, 128, 64]]
        dropouts = [0.1, 0.2, 0.3]
        learning_rates = [0.0005, 0.001, 0.002]
        search_name = "full"
    else:
        print("잘못된 선택입니다. 빠른 Grid Search로 진행합니다.")
        embedding_dims = [64, 128]
        mlp_layers_list = [[128, 64, 32], [256, 128, 64]]
        dropouts = [0.2, 0.3]
        learning_rates = [0.001, 0.002]
        search_name = "fast"
    
    # 모든 조합 생성
    combinations = list(product(
        embedding_dims,
        mlp_layers_list,
        dropouts,
        learning_rates
    ))
    
    total_combinations = len(combinations)
    print(f"\n총 {total_combinations}개 조합 실험 시작")
    print("=" * 70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # 데이터 준비 (100만건 샘플)
    print("데이터 로딩 중...")
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
            sample_size=1000000
        )
    
    # 공통 설정
    BATCH_SIZE = 4096
    EPOCHS = 5
    NUM_WORKERS = 4
    
    # 결과 저장
    results = []
    best_rmse = float('inf')
    best_config = None
    
    # Grid Search 실행
    for idx, (emb_dim, mlp_layers, dropout, lr) in enumerate(combinations, 1):
        print(f"\n{'='*70}")
        print(f"실험 {idx}/{total_combinations}")
        print(f"{'='*70}")
        print(f"embedding_dim: {emb_dim}")
        print(f"mlp_layers: {mlp_layers}")
        print(f"dropout: {dropout}")
        print(f"learning_rate: {lr}")
        print(f"batch_size: {BATCH_SIZE}")
        
        # DataLoader 생성
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS
        )
        
        # 학습 및 평가
        try:
            rmse, mae = await train_and_evaluate(
                train_loader=train_loader,
                test_loader=test_loader,
                n_users=n_users,
                n_items=n_items,
                embedding_dim=emb_dim,
                mlp_layers=mlp_layers,
                dropout=dropout,
                learning_rate=lr,
                epochs=EPOCHS,
                device=device
            )
            
            # 결과 저장
            result = {
                'index': idx,
                'embedding_dim': emb_dim,
                'mlp_layers': mlp_layers,
                'dropout': dropout,
                'learning_rate': lr,
                'batch_size': BATCH_SIZE,
                'rmse': rmse,
                'mae': mae
            }
            results.append(result)
            
            print(f"\nResult: RMSE={rmse:.4f}, MAE={mae:.4f}")
            
            # 최고 성능 업데이트
            if rmse < best_rmse:
                best_rmse = rmse
                best_config = result
                print(f"*** 새로운 최고 성능! RMSE={rmse:.4f} ***")
            
            # 중간 저장 (10개마다)
            if idx % 10 == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_file = f"grid_search_temp_{search_name}_{timestamp}.json"
                with open(temp_file, 'w') as f:
                    json.dump({
                        'progress': f"{idx}/{total_combinations}",
                        'best_config': best_config,
                        'all_results': results
                    }, f, indent=2)
                print(f"중간 결과 저장: {temp_file}")
        
        except Exception as e:
            print(f"에러 발생: {e}")
            results.append({
                'index': idx,
                'embedding_dim': emb_dim,
                'mlp_layers': mlp_layers,
                'dropout': dropout,
                'learning_rate': lr,
                'batch_size': BATCH_SIZE,
                'rmse': None,
                'mae': None,
                'error': str(e)
            })
    
    # ==========================================
    # 최종 결과
    # ==========================================
    print("\n" + "=" * 70)
    print("Grid Search 완료!")
    print("=" * 70)
    
    # 상위 5개 결과
    valid_results = [r for r in results if r['rmse'] is not None]
    top_5 = sorted(valid_results, key=lambda x: x['rmse'])[:5]
    
    print("\n상위 5개 설정:")
    print("-" * 70)
    for i, result in enumerate(top_5, 1):
        print(f"\n{i}위:")
        print(f"  embedding_dim: {result['embedding_dim']}")
        print(f"  mlp_layers: {result['mlp_layers']}")
        print(f"  dropout: {result['dropout']}")
        print(f"  learning_rate: {result['learning_rate']}")
        print(f"  RMSE: {result['rmse']:.4f}")
        print(f"  MAE: {result['mae']:.4f}")
    
    # 최적 설정
    print("\n" + "=" * 70)
    print("최적 하이퍼파라미터:")
    print("=" * 70)
    print(f"embedding_dim: {best_config['embedding_dim']}")
    print(f"mlp_layers: {best_config['mlp_layers']}")
    print(f"dropout: {best_config['dropout']}")
    print(f"learning_rate: {best_config['learning_rate']}")
    print(f"batch_size: {best_config['batch_size']}")
    print(f"\nBest RMSE: {best_config['rmse']:.4f}")
    print(f"Best MAE: {best_config['mae']:.4f}")
    
    # 최종 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"grid_search_results_{search_name}_{timestamp}.json"
    
    with open(result_file, 'w') as f:
        json.dump({
            'search_type': search_name,
            'total_combinations': total_combinations,
            'completed': len(valid_results),
            'failed': len(results) - len(valid_results),
            'best_config': best_config,
            'top_5': top_5,
            'all_results': results
        }, f, indent=2)
    
    print(f"\n최종 결과 저장: {result_file}")
    print("=" * 70)
    
    # train_ncf.py 설정 코드 생성
    print("\ntrain_ncf.py에 사용할 최적 설정:")
    print("-" * 70)
    print(f"EMBEDDING_DIM = {best_config['embedding_dim']}")
    print(f"MLP_LAYERS = {best_config['mlp_layers']}")
    print(f"DROPOUT = {best_config['dropout']}")
    print(f"LEARNING_RATE = {best_config['learning_rate']}")
    print(f"BATCH_SIZE = {best_config['batch_size']}")
    print("-" * 70)


if __name__ == "__main__":
    asyncio.run(main())