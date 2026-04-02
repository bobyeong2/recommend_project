#!/bin/bash

# Health Check 스크립트
# Usage: ./health_check.sh [blue|green]

COLOR=$1

if [ "$COLOR" = "blue" ]; then
    PORT=8001
elif [ "$COLOR" = "green" ]; then
    PORT=8002
else
    echo "Usage: $0 [blue|green]"
    exit 1
fi

MAX_ATTEMPTS=60
ATTEMPT=0

echo "Checking health of bob_app_$COLOR on port $PORT..."

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if curl -f http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "✓ bob_app_$COLOR is healthy!"
        exit 0
    fi
    
    ATTEMPT=$((ATTEMPT + 1))
    echo "Attempt $ATTEMPT/$MAX_ATTEMPTS - Waiting..."
    sleep 10
done

echo "✗ bob_app_$COLOR failed to become healthy after $MAX_ATTEMPTS seconds"
exit 1