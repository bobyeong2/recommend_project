#!/bin/bash

# Blue-Green 배포 스크립트
# Usage ./deploy.sh [blue|green]

set -e

COLOR=$1
PROJECT_DIR="/home/deploy/bob_project"

if [ "$COLOR" != "blue" ] && [ "$COLOR" != "green" ]; then
    echo "Usage: $0 [blue|green]"
    exit 1
fi

echo "===== Deploying $COLOR ====="
cd $PROJECT_DIR

# 최신코드 pull
echo "Pulling latest code ..."
git pull origin main

# container build and Run
echo "Building and starting bob_app_$COLOR..."
if [ "$COLOR" = "green" ]; then
    docker compose -f docker-compose.blue-green.yml --profile green up -d bob_app_green --build
else
    docker compose -f docker-compose.blue-green.yml --profile blue up -d bob_app_blue --build
fi

# Health check waiting
echo "Waiting for container to be healty....."
./scripts/health_check.sh $COLOR

if [ $? -eq 0 ]; then
    echo "===== DEPLOYMENT of $COLOR completed successfully ====="
else
    echo "===== DEPLOYMENT of $COLOR failed ====="
    exit 1
fi
