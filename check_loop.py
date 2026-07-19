from llama_cpp import Llama
import numpy as np

model_blob = "/mnt/c/Users/reyse/OneDrive/Desktop/Bayesian-AI/research/dojo_forge/pipeline/model.gguf"
llm = Llama(model_path=model_blob, n_gpu_layers=-1, n_ctx=512, logits_all=True)

sys_text = f"<|im_start|>system\nYou are an agent.<|im_end|>\n"
sys_tokens = llm.tokenize(sys_text.encode('utf-8'), add_bos=False, special=True)
for i in range(0, len(sys_tokens), llm.n_batch):
    llm.eval(sys_tokens[i:i + llm.n_batch])

sys_len = len(sys_tokens)
llm._ctx.kv_cache_seq_rm(-1, sys_len, -1)
llm.n_tokens = sys_len

user_prefix = llm.tokenize(b"<|im_start|>user\n", add_bos=False, special=True)
assistant_suffix = llm.tokenize(b"<|im_end|>\n<|im_start|>assistant\n", add_bos=False, special=True)
frame_tokens = llm.tokenize(b"hello world", add_bos=False, special=False)

new_tokens = user_prefix + frame_tokens + assistant_suffix
llm.eval(new_tokens)

logits = np.array(llm.eval_logits[-1])
print(f"Logits shape: {logits.shape}")
print(f"Logits max: {np.max(logits)}")
print(f"Logits min: {np.min(logits)}")
print(f"Logit 49488: {logits[49488]}")
print(f"Logit 7863: {logits[7863]}")
