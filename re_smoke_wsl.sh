export PYTHONPATH=/mnt/c/Users/reyse/OneDrive/Desktop/Bayesian-AI
V=/home/reyses/venvs/bayesian-ai/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$V/cuda_runtime/lib:$V/cublas/lib:$V/cudnn/lib:$LD_LIBRARY_PATH

echo "Running LLAMA.CPP smoke test (dummy/real)"
/home/reyses/venvs/bayesian-ai/bin/python -u -X faulthandler research/dojo_forge/pipeline/forge_harness.py --episodes 2025_01_02_1735831200_S 2025_01_03_1735934940_S 2025_01_06_1736182040_S 2025_01_07_1736261820_S --run-id f1-resmoke-llama --model-blob /mnt/d/ollama/models/blobs/sha256-a8cc1361f3145dc01f6d77c6c82c9116b9ffe3c97b34716fe20418455876c40e
