import os
import time
import math
from llama_cpp import Llama

def extract_p_exit(llm, tokens):
    start = time.time()
    llm.eval(tokens)
    dur = time.time() - start
    
    # We want logits for the last token
    logits = llm.eval_logits
    
    if hasattr(logits, '__len__') and len(logits) > 0:
        last_logits = logits[-1]
    else:
        last_logits = logits
        
    # Get vocab IDs for EXIT and HOLD
    token_exit = llm.tokenize(b"EXIT", add_bos=False, special=False)[0]
    token_hold = llm.tokenize(b"HOLD", add_bos=False, special=False)[0]
    
    logit_exit = last_logits[token_exit]
    logit_hold = last_logits[token_hold]
    
    p_exit = math.exp(logit_exit) / (math.exp(logit_exit) + math.exp(logit_hold))
    
    print(f"Tokens eval: {len(tokens)}")
    print(f"Time taken: {dur:.3f}s")
    print(f"Logits -> EXIT: {logit_exit:.4f} HOLD: {logit_hold:.4f}")
    print(f"P(EXIT)  -> {p_exit:.6f}")
    return p_exit

llm = Llama(model_path="research/dojo_forge/pipeline/model.gguf", n_gpu_layers=-1, n_ctx=4096, seed=42, temperature=0.0)

# Simulate system prompt
system_text = b"System: You are an expert trader. Given the state, respond with EXIT or HOLD."
sys_tokens = llm.tokenize(system_text, add_bos=True, special=True)
print("\n--- Cold Cache (System Prompt) ---")
extract_p_exit(llm, sys_tokens)

# Simulate state frame
frame_text = b"Frame 1: Price 100. Action:"
frame_tokens = llm.tokenize(frame_text, add_bos=False, special=False)
print("\n--- Warm Cache (Frame 1) ---")
extract_p_exit(llm, frame_tokens)

frame_text2 = b"Frame 2: Price 90. Action:"
frame_tokens2 = llm.tokenize(frame_text2, add_bos=False, special=False)
print("\n--- Warm Cache (Frame 2) ---")
extract_p_exit(llm, frame_tokens2)
