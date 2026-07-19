from llama_cpp import Llama
import numpy as np

model_blob = "/mnt/c/Users/reyse/OneDrive/Desktop/Bayesian-AI/research/dojo_forge/pipeline/model.gguf"
llm = Llama(model_path=model_blob, n_gpu_layers=-1, n_ctx=512, logits_all=True)

tok_exit = llm.tokenize(b"EXIT")[0]
tok_hold = llm.tokenize(b"HOLD")[0]
print(f"tok_exit={tok_exit} tok_hold={tok_hold}")

text = b"<|im_start|>user\nHello\n<|im_end|>\n<|im_start|>assistant\n"
tokens = llm.tokenize(text, add_bos=False, special=True)
llm.eval(tokens)

logits = np.array(llm.eval_logits[-1])
print(f"logits shape: {logits.shape}")
print(f"logit_exit: {logits[tok_exit]}")
print(f"logit_hold: {logits[tok_hold]}")
