#!/bin/bash
V=/home/reyses/venvs/bayesian-ai/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$V/cuda_runtime/lib:$V/cublas/lib:$V/cudnn/lib:$V/cuda_nvrtc/lib:$LD_LIBRARY_PATH

# Run whatever arguments are passed
exec /home/reyses/venvs/bayesian-ai/bin/python "$@"
