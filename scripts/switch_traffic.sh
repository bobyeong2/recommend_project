#!/bin/bash

# Nginx Upstream 전환 스크립트
# Usage: ./switch_traffic.sh [blue|green]

set -e

COLOR=$1
PROJECT_DIR="/home/deploy/bob_project"
NGINX_CONF="$PROJECT_DIR/nginx/nginx.conf"

if [ "$COLOR" != "blue" ] && [ "$COLOR" != "green" ]; then
    echo "Usage: $0 [blue|green]"
    exit 1
fi

echo "===== Switching traffic to $COLOR ====="

# Nginx 설정 변경
if [ "$COLOR" = "blue" ]; then
    sed -i 's/# server bob_app_blue:8000;/server bob_app_blue:8000;/' $NGINX_CONF
    sed -i 's/server bob_app_green:8000;/# server bob_app_green:8000;/' $NGINX_CONF
else
    sed -i 's/server bob_app_blue:8000;/# server bob_app_blue:8000;/' $NGINX_CONF
    sed -i 's/# server bob_app_green:8000;/server bob_app_green:8000;/' $NGINX_CONF
fi

# Nginx reload
docker exec bob_nginx nginx -s reload

echo "✓ Traffic switched to bob_app_$COLOR"