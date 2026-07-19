export PYTHONPATH=/mnt/c/Users/reyse/OneDrive/Desktop/Bayesian-AI

echo "Running LLAMA.CPP smoke test (dummy/real)"
/home/reyses/venvs/bayesian-ai/bin/python research/dojo_forge/pipeline/forge_harness.py --episodes 2025_01_02_1735831200_S 2025_01_03_1735934940_S 2025_01_06_1736182040_S 2025_01_07_1736261820_S --run-id f1-resmoke-llama
