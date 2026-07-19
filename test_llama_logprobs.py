import sys
from llama_cpp import Llama

llm = Llama(model_path='gemma4-e2b.gguf', n_gpu_layers=-1, verbose=False)

print("Tokens:")
t_exit = llm.tokenize(b"EXIT")
t_hold = llm.tokenize(b"HOLD")
print(f"EXIT: {t_exit}")
print(f"HOLD: {t_hold}")

print("Attributes of llm:")
print([a for a in dir(llm) if 'logit' in a or 'score' in a])
