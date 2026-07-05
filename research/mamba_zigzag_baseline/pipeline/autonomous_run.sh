#!/bin/bash
export PYTHONPATH=.
# Disable debug overhead for the actual autonomous run
export TORCH_COMPILE_DEBUG=0

# Total number of epochs for the 1-week run
TOTAL_EPOCHS=200

echo "Starting Autonomous Training Loop for $TOTAL_EPOCHS epochs..."

while true; do
    # Check if the final epoch checkpoint exists, meaning we are completely done
    if [ -f "mamba_rl_checkpoint_ep$((TOTAL_EPOCHS-1)).pth" ]; then
        echo "Final checkpoint found. Autonomous run completed successfully!"
        break
    fi

    echo "Launching train_mamba_rl.py..."
    .venv_wsl/bin/python research/mamba_zigzag_baseline/pipeline/train_mamba_rl.py --num_episodes $TOTAL_EPOCHS
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "train_mamba_rl.py exited with 0 (Success). We should be done!"
        # The loop will check the final checkpoint in the next iteration and break if true.
    elif [ $EXIT_CODE -eq 88 ]; then
        echo "train_mamba_rl.py exited with 88 (E-EXIT / RAM Failsafe)."
        echo "Sleeping for 60 seconds to allow RAM to clear..."
        sleep 60
    else
        echo "train_mamba_rl.py crashed with exit code $EXIT_CODE (Segfault/OOM/Error)."
        echo "Sleeping for 30 seconds before auto-restarting..."
        sleep 30
    fi
done
