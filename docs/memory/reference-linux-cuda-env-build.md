---
name: reference-linux-cuda-env-build
description: "Native-Linux (Ubuntu 26.04) env rebuild recipe — conda py3.12, torch cu121, mamba/causal prebuilt wheels, and the gcc-13 fix for the llama-cpp-python CUDA compile (nvcc rejects Ubuntu 26.04's gcc-15). Verified working 2026-07-21."
metadata:
  node_type: memory
  type: reference
---

**Full env rebuild on native Ubuntu 26.04 (RTX 3060, migrated off Windows/WSL 2026-07-21). VERIFIED end-to-end (teacher GGUF loads + offloads to GPU).** See also [[ONBOARDING]], [[reference-mamba-ssm-wsl-perf]], [[project-data-locations]].

## Env
- **conda env `bayesian`, Python 3.12** at `~/miniforge3/envs/bayesian` — MUST be on ext4 (the repo drive is NTFS/fuseblk; NTFS can't hold venv symlinks → the old `.venv` was broken). Miniforge used because Ubuntu 26.04 is python-3.14-native (no apt python3.12).
- `pip install -r requirements.txt` → **torch 2.5.1+cu121**, `torch.cuda.is_available()==True` (torch's bundled cu121 runtime; no system CUDA needed for torch). numpy 1.26.4, pandas 3.0.3 (pin `<3` if pandas-3 API breaks bite).
- Extra prereqs not in requirements.txt (found by audit): `h5py optuna beautifulsoup4 markdownify python-dotenv pytz statsmodels seaborn pandas-market-calendars streamlit yfinance mcp`. (`google.antigravity` is Google's AG SDK, not pip-installable.)

## mamba stack — prebuilt wheels (NO compile)
- `causal_conv1d 1.5.0.post8` + `mamba_ssm 2.2.4`, tag `+cu12torch2.5cxx11abiFALSE-cp312` from the Dao-AILab / state-spaces GitHub releases. torch2.6 variants also exist.
- **transformers-5.x trap**: `import mamba_ssm` fails (`GreedySearchDecoderOnlyOutput`/`SampleDecoderOnlyOutput` removed). FIX = `sitecustomize.py` in site-packages aliasing both (+Beam variants) to `GenerateDecoderOnlyOutput`. Do NOT downgrade transformers.

## llama-cpp-python — MUST compile from source (the hard part, 5 attempts)
- The **prebuilt cu121 wheel SIGILLs** on load (CPU has avx2/fma, NO avx512; exit 132). So build from source.
- **THE root cause of repeated failures**: Ubuntu 26.04 ships **gcc-15**, and CUDA nvcc rejects it — `nvcc fatal: Failed to preprocess host compiler properties`. Conda-CUDA also fails cmake `FindCUDAToolkit`.
- **WORKING recipe** (COMPILE_V3_OK / TEACHER_V3_OK):
  ```
  sudo apt-get install -y nvidia-cuda-toolkit gcc-13 g++-13   # system CUDA 12.4 + a CUDA-compatible gcc
  export CUDACXX=/usr/bin/nvcc CUDAToolkit_ROOT=/usr CUDA_PATH=/usr
  export CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13
  export CMAKE_ARGS="-DGGML_CUDA=on -DGGML_NATIVE=off -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 -DCMAKE_C_COMPILER=/usr/bin/gcc-13 \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCUDAToolkit_ROOT=/usr"
  ~/miniforge3/envs/bayesian/bin/pip install --force-reinstall --no-cache-dir \
    --no-binary llama-cpp-python "llama-cpp-python==0.3.34"
  ```
  Key flags: **`GGML_NATIVE=off`** (portable runtime CPU dispatch → no avx512 SIGILL), **arch sm_86** (RTX 3060), **gcc-13 host compiler** (the nvcc/gcc-15 fix). Recipe basis: `research/dojo_forge/reports/gpu_wsl_build.md`.
- **Runtime**: llama's `libllama.so` needs `libcudart.so.12` — export `LD_LIBRARY_PATH=$(ls -d ~/miniforge3/envs/bayesian/lib/python3.12/site-packages/nvidia/*/lib | paste -sd:)` (or `conda env config vars set`).

## Verified
21/21 imports OK, CUDA True, teacher GGUF loads with layers on CUDA0. Teacher blobs at `/media/moi/WindowsCode/ollama/models/blobs/` (qwen3:14b = `sha256-a8cc…`, gemma4 = `sha256-4e30…`).
