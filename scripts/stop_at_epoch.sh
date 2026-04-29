#!/bin/bash
# Monitors training and sends SIGINT (clean shutdown) at target epoch.
# Usage: nohup bash scripts/stop_at_epoch.sh &
# Runs on the training VM.

TARGET_EPOCH=${1:-150}
CHECK_INTERVAL=120  # seconds between checks

echo "Will stop training at epoch $TARGET_EPOCH (checking every ${CHECK_INTERVAL}s)"

while true; do
    CURRENT_EPOCH=$(grep -c '^Epoch' ~/training.log 2>/dev/null || echo 0)

    if [ "$CURRENT_EPOCH" -ge "$TARGET_EPOCH" ]; then
        echo "Epoch $CURRENT_EPOCH >= $TARGET_EPOCH — stopping training"
        PID=$(pgrep -f 'python3 main.py' | head -1)
        if [ -n "$PID" ]; then
            kill -INT "$PID"
            echo "Sent SIGINT to PID $PID"
        else
            echo "Training process not found — may have already stopped"
        fi
        exit 0
    fi

    echo "Epoch $CURRENT_EPOCH / $TARGET_EPOCH — waiting..."
    sleep "$CHECK_INTERVAL"
done
