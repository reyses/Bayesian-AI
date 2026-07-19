from llama_cpp import Llama
model_blob = "/mnt/c/Users/reyse/OneDrive/Desktop/Bayesian-AI/research/dojo_forge/pipeline/model.gguf"
llm = Llama(model_path=model_blob, n_gpu_layers=0)
print('HOLD:', llm.tokenize(b'HOLD'))
print('EXIT:', llm.tokenize(b'EXIT'))
