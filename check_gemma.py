from llama_cpp import Llama
import numpy as np

model_blob = "/mnt/c/Users/reyse/.ollama/models/blobs/sha256-4e30e2665218745ef463f722c0bf86be0cab6ee676320f1cfadf91e989107448"
llm = Llama(model_path=model_blob, n_gpu_layers=-1, n_ctx=512, logits_all=True)

tok_exit = llm.tokenize(b"EXIT")[1] if len(llm.tokenize(b"EXIT")) > 1 else llm.tokenize(b"EXIT")[0]
tok_hold = llm.tokenize(b"HOLD")[1] if len(llm.tokenize(b"HOLD")) > 1 else llm.tokenize(b"HOLD")[0]
print(f"tok_exit={tok_exit} tok_hold={tok_hold}")

text = b"<|im_start|>user\nHello\n<|im_end|>\n<|im_start|>assistant\n"
tokens = llm.tokenize(text, add_bos=False, special=True)
llm.eval(tokens)

logits = np.array(llm.eval_logits[-1])
print(f"Logits shape: {logits.shape}")
print(f"Logits max: {np.max(logits)}")
print(f"Logits min: {np.min(logits)}")
print(f"Logit EXIT: {logits[tok_exit]}")
print(f"Logit HOLD: {logits[tok_hold]}")
