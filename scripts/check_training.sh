#!/bin/bash
# Quick training status check
# Usage: bash scripts/check_training.sh

export PATH="$HOME/google-cloud-sdk/bin:$PATH"
ZONE="us-central1-b"
INSTANCE="diamond-cs2-train"

echo "╔══════════════════════════════════════════╗"
echo "║     DIAMOND CS2 Training Status          ║"
echo "╚══════════════════════════════════════════╝"

gcloud compute ssh "$INSTANCE" --zone="$ZONE" --tunnel-through-iap --ssh-flag="-o ConnectTimeout=30" --command="
# Current epoch
CURRENT_EPOCH=\$(grep -c '^Epoch' ~/training.log 2>/dev/null || echo 0)
TOTAL_EPOCHS=150

# Calculate progress
PCT=\$(( CURRENT_EPOCH * 100 / TOTAL_EPOCHS ))

# Progress bar
BAR_LEN=30
FILLED=\$(( PCT * BAR_LEN / 100 ))
EMPTY=\$(( BAR_LEN - FILLED ))
BAR=\$(printf '%0.s█' \$(seq 1 \$FILLED 2>/dev/null))
BAR=\$BAR\$(printf '%0.s░' \$(seq 1 \$EMPTY 2>/dev/null))

echo \"\"
echo \"  Progress: [\$BAR] \$PCT%\"
echo \"  Epoch:    \$CURRENT_EPOCH / \$TOTAL_EPOCHS\"
echo \"\"

# Time estimate
if [ \$CURRENT_EPOCH -gt 0 ]; then
    UPTIME_SECS=\$(ps -p \$(pgrep -f 'python3 main.py' | head -1) -o etimes= 2>/dev/null || echo 0)
    if [ \"\$UPTIME_SECS\" -gt 0 ] && [ \"\$CURRENT_EPOCH\" -gt 0 ]; then
        SECS_PER_EPOCH=\$(( UPTIME_SECS / CURRENT_EPOCH ))
        REMAINING_EPOCHS=\$(( TOTAL_EPOCHS - CURRENT_EPOCH ))
        REMAINING_SECS=\$(( REMAINING_EPOCHS * SECS_PER_EPOCH ))
        REMAINING_HRS=\$(( REMAINING_SECS / 3600 ))
        REMAINING_DAYS=\$(( REMAINING_HRS / 24 ))
        echo \"  Speed:    ~\$(( SECS_PER_EPOCH / 60 )) min/epoch\"
        echo \"  ETA:      ~\${REMAINING_HRS} hrs (\${REMAINING_DAYS} days)\"
        echo \"  Est cost: ~\\\$\$(( REMAINING_HRS * 120 / 100 )) remaining\"
    fi
fi

echo \"\"

# GPU status
echo \"  GPU:\"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null | while read line; do
    echo \"    \$line\"
done

echo \"\"

# Latest training activity
echo \"  Latest log:\"
tail -3 ~/training.log 2>/dev/null | while read line; do
    echo \"    \$line\"
done

echo \"\"
" 2>&1 | grep -v "WARNING:" | grep -v "please see" | grep -v "^$" | head -30
