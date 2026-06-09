#!/bin/bash
# GCP Bootstrap Script for Bayesian-AI

echo "[1/4] Updating System & Installing Dependencies..."
sudo apt-get update -y
sudo apt-get install -y python3.11 python3.11-venv python3-pip git wget curl rclone unzip

echo "[2/4] Setting up Python Virtual Environment..."
python3.11 -m venv ~/bayesian_env
source ~/bayesian_env/bin/activate

echo "[3/4] Installing PyTorch & Data Science Libraries..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pandas numpy scikit-learn pyarrow fastparquet

echo "[4/4] Environment Ready!"
echo "Next Steps:"
echo "1. Run 'rclone config' to sync OneDrive data"
echo "2. Clone your GitHub repo"
echo "3. Run the pipeline!"
