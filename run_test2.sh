export PYTHONPATH=/mnt/c/Users/reyse/OneDrive/Desktop/Bayesian-AI
V=/home/reyses/venvs/bayesian-ai/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$V/cuda_runtime/lib:$V/cublas/lib:$V/cudnn/lib:$LD_LIBRARY_PATH
/home/reyses/venvs/bayesian-ai/bin/python -X faulthandler research/dojo_forge/pipeline/test_native2.py
