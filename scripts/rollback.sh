#!/bin/bash

# 롤백 스크립트
# Usage: ./rollback.sh [blue|green]

set -e

FROM_COLOR=$1

if [ "$FROM_COLOR" != "blue" ] && [ "$FROM_COLOR" != "green" ]; then
    echo "Usage: $0 [blue|green]"
    exit 1
fi

# 반대 색상 계산
if [ "$FROM_COLOR" = "blue" ]; then
    TO_COLOR="green"
else
    TO_COLOR="blue"
fi

echo "===== Rolling back from $FROM_COLOR to $TO_COLOR ====="

# 트래픽 전환
./scripts/switch_traffic.sh $TO_COLOR

# 실패한 컨테이너 중지
docker compose -f docker-compose.blue-green.yml stop bob_app_$FROM_COLOR

echo "✓ Rolled back to bob_app_$TO_COLOR"