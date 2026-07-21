export PYTHONPATH=/mnt/c/Users/reyse/OneDrive/Desktop/Bayesian-AI
V=/home/reyses/venvs/bayesian-ai/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$V/cuda_runtime/lib:$V/cublas/lib:$V/cudnn/lib:$LD_LIBRARY_PATH
/home/reyses/venvs/bayesian-ai/bin/python -c '
from llama_cpp import Llama
try:
    print("Initializing...")
    llm = Llama(model_path="/mnt/d/ollama/models/blobs/sha256-a8cc1361f3145dc01f6d77c6c82c9116b9ffe3c97b34716fe20418455876c40e", n_gpu_layers=-1, n_ctx=4096, seed=42, temperature=0.0, logits_all=True)
    print("SUCCESS")
except Exception as e:
    import traceback
    traceback.print_exc()
'
