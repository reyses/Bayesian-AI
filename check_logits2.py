from llama_cpp import Llama
import numpy as np

model_blob = "/mnt/c/Users/reyse/OneDrive/Desktop/Bayesian-AI/research/dojo_forge/pipeline/model.gguf"
llm = Llama(model_path=model_blob, n_gpu_layers=-1, n_ctx=512, logits_all=True)

tok_exit = llm.tokenize(b"EXIT")[1] if len(llm.tokenize(b"EXIT")) > 1 else llm.tokenize(b"EXIT")[0]
tok_hold = llm.tokenize(b"HOLD")[1] if len(llm.tokenize(b"HOLD")) > 1 else llm.tokenize(b"HOLD")[0]
print(f"tok_exit={tok_exit} tok_hold={tok_hold}")

text = b"<|im_start|>user\nHello\n<|im_end|>\n<|im_start|>assistant\n"
tokens = llm.tokenize(text, add_bos=False, special=True)
llm.eval(tokens)

logits = np.array(llm.eval_logits[-1])
logit_exit = logits[tok_exit]
logit_hold = logits[tok_hold]
exp_exit = np.exp(logit_exit)
exp_hold = np.exp(logit_hold)
p_exit = exp_exit / (exp_exit + exp_hold + 1e-9)

print(f"logit_exit: {logit_exit}")
print(f"logit_hold: {logit_hold}")
print(f"exp_exit: {exp_exit}")
print(f"exp_hold: {exp_hold}")
print(f"p_exit: {p_exit:.10f}")
