import os
import json
import glob
import math
import time
from llama_cpp import Llama

DOJO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PACKETS_DIR = os.path.join(DOJO_ROOT, 'reports', 'gen0', 'packets')
GENOME_PATH = os.path.join(DOJO_ROOT, 'genome', 'GENOME.md')

if os.path.exists(GENOME_PATH):
    with open(GENOME_PATH, 'r', encoding='utf-8') as f:
        genome_text = f.read()
else:
    genome_text = ""

SYSTEM_PROMPT = f"Decide to HOLD or EXIT based on the frame. If EXIT, provide a reason.\n\nRULES (Genome):\n{genome_text}"
NUM_CTX = 8192

def get_p_exit(llm, prompt_text):
    try:
        response = llm.create_completion(
            prompt_text,
            max_tokens=1,
            logprobs=50,
            temperature=0.0
        )
    except ValueError as e:
        if "exceed context window" in str(e):
            return -100.0, -100.0, NUM_CTX + 1
        raise e
    
    prompt_eval_count = response['usage']['prompt_tokens']
    
    logprobs = response['choices'][0]['logprobs']['top_logprobs'][0]
    logit_exit = logprobs.get('EXIT', logprobs.get(' EXIT', -100.0))
    logit_hold = logprobs.get('HOLD', logprobs.get(' HOLD', -100.0))
    
    if logit_exit == -100.0: logit_exit = logprobs.get('exit', logprobs.get(' exit', -100.0))
    if logit_hold == -100.0: logit_hold = logprobs.get('hold', logprobs.get(' hold', -100.0))
    
    return logit_exit, logit_hold, prompt_eval_count

def main():
    packet_files = glob.glob(os.path.join(PACKETS_DIR, "*.json"))
    if not packet_files:
        print("No packet files found.")
        return
        
    print(f"Loading Qwen3 14b with n_ctx={NUM_CTX}...")
    llm = Llama(
        model_path=r"D:\ollama\models\blobs\sha256-a8cc1361f3145dc01f6d77c6c82c9116b9ffe3c97b34716fe20418455876c40e", 
        n_gpu_layers=-1, 
        n_ctx=NUM_CTX, 
        seed=42, 
        temperature=0.0, 
        logits_all=True
    )
    print("Model loaded.")
    
    out_path = os.path.join(DOJO_ROOT, 'reports', 'acceptance_native_gen0.csv')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("eid,frame_idx,p_exit,prompt_tokens,tainted\n")
    
    total = len(packet_files)
    for idx, pkt_path in enumerate(packet_files):
        eid = os.path.basename(pkt_path).replace('.json', '')
        print(f"\n[{idx+1}/{total}] Processing {eid}...")
        
        with open(pkt_path, 'r', encoding='utf-8') as f:
            packet = json.load(f)
            
        frames = packet.get('frames', [])
        
        # Reset context
        llm.reset()
        
        prompt_text = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        
        tainted = False
        
        for i, frame in enumerate(frames):
            frame_text = frame['text']
            prompt_text += f"<|im_start|>user\n{frame_text}<|im_end|>\n<|im_start|>assistant\n<think>\nDecision bypassed for native logprobs.\n</think>\n"
            
            logit_exit, logit_hold, pt_tokens = get_p_exit(llm, prompt_text)
            
            if pt_tokens >= NUM_CTX:
                print(f"  Frame {i} TAINTED: {pt_tokens} tokens >= {NUM_CTX}!")
                tainted = True
                with open(out_path, 'a', encoding='utf-8') as f:
                    f.write(f"{eid},{i},NaN,{pt_tokens},Y\n")
                break
                
            if logit_exit == -100.0 or logit_hold == -100.0:
                print(f"  Frame {i} TAINTED: missing candidate logprob (EXIT: {logit_exit}, HOLD: {logit_hold})")
                tainted = True
                with open(out_path, 'a', encoding='utf-8') as f:
                    f.write(f"{eid},{i},NaN,{pt_tokens},Y\n")
                break
                
            p_exit = math.exp(logit_exit) / (math.exp(logit_exit) + math.exp(logit_hold))
                
            with open(out_path, 'a', encoding='utf-8') as f:
                f.write(f"{eid},{i},{p_exit:.6f},{pt_tokens},N\n")
                
            # Append the chosen decision to prompt context for the next frame
            decision = "EXIT" if p_exit > 0.5 else "HOLD"
            prompt_text += f"{decision}<|im_end|>\n"

if __name__ == '__main__':
    main()
