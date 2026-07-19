import os
import time
import math
from llama_cpp import Llama

def extract_p_exit(llm, prompt_text):
    start = time.time()
    
    # We want to force the model to output one token and get the logprobs of the top candidates
    response = llm.create_completion(
        prompt_text,
        max_tokens=1,
        logprobs=5,
        temperature=0.0
    )
    
    dur = time.time() - start
    
    logprobs = response['choices'][0]['logprobs']['top_logprobs'][0]
    
    # The keys in top_logprobs will be strings, so we need to look for 'EXIT' and 'HOLD'
    # Sometimes it might have a leading space like ' EXIT' or ' HOLD'
    logit_exit = logprobs.get('EXIT', logprobs.get(' EXIT', -100.0))
    logit_hold = logprobs.get('HOLD', logprobs.get(' HOLD', -100.0))
    
    # Check lower case as fallback
    if logit_exit == -100.0: logit_exit = logprobs.get('exit', logprobs.get(' exit', -100.0))
    if logit_hold == -100.0: logit_hold = logprobs.get('hold', logprobs.get(' hold', -100.0))
    
    # If the token is totally missing from top 5, it gets -100
    p_exit = math.exp(logit_exit) / (math.exp(logit_exit) + math.exp(logit_hold))
    
    print(f"Time taken: {dur:.3f}s")
    print("Top 5 logprobs: ", {k.encode("ascii", "ignore").decode("ascii"): v for k,v in logprobs.items()})
    print(f"Logits -> EXIT: {logit_exit:.4f} HOLD: {logit_hold:.4f}")
    print(f"P(EXIT)  -> {p_exit:.6f}")
    return p_exit

llm = Llama(model_path=r"D:\ollama\models\blobs\sha256-a8cc1361f3145dc01f6d77c6c82c9116b9ffe3c97b34716fe20418455876c40e", n_gpu_layers=-1, n_ctx=4096, seed=42, temperature=0.0, logits_all=True)

# Simulate system prompt + frame 1
text1 = "<|im_start|>system\nYou are an expert trader. Given the state, respond with EXIT or HOLD.<|im_end|>\n<|im_start|>user\nFrame 1: Price 100. Action:<|im_end|>\n<|im_start|>assistant\n"
print("\n--- Cold Cache (System Prompt + Frame 1) ---")
extract_p_exit(llm, text1)

# Simulate frame 2
text2 = text1 + "HOLD<|im_end|>\n<|im_start|>user\nFrame 2: Price 90. Action:<|im_end|>\n<|im_start|>assistant\n"
print("\n--- Warm Cache (Frame 2) ---")
extract_p_exit(llm, text2)
