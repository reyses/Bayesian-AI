#!/bin/bash
# Auto-launch full training after smoke test completes

echo "Waiting for smoke test to complete..."
echo "Monitoring: C:\Users\reyse\AppData\Local\Temp\claude\c--Users-reyse-OneDrive-Desktop-Bayesian-AI\tasks\be6f18c.output"

# Wait for smoke test completion
while ! grep -q "Walk-Forward Training Complete" "C:\Users\reyse\AppData\Local\Temp\claude\c--Users-reyse-OneDrive-Desktop-Bayesian-AI\tasks\be6f18c.output" 2>/dev/null; do
    sleep 60
    echo "Still running... $(date)"
done

echo ""
echo "=========================================="
echo "SMOKE TEST COMPLETE!"
echo "=========================================="
echo ""
echo "Launching FULL TRAINING..."
echo "27 days × 1000 iterations per day"
echo ""

cd "c:\Users\reyse\OneDrive\Desktop\Bayesian-AI"

python training/walk_forward_trainer.py \
    --data "DATA/glbx-mdp3-20251230-20260129.ohlcv-1s.parquet" \
    --iterations 1000 \
    --checkpoint-dir "checkpoints/full_training" \
    2>&1 | tee full_training.log

echo ""
echo "=========================================="
echo "FULL TRAINING COMPLETE!"
echo "=========================================="
