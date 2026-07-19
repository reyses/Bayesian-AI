import os
import json
import glob
import requests

DOJO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
GATE_STATE_DIR = os.path.join(DOJO_ROOT, 'gate_state', 'gen0')
PACKETS_DIR = os.path.join(DOJO_ROOT, 'reports', 'gen0', 'packets')
GENOME_PATH = os.path.join(DOJO_ROOT, 'genome', 'GENOME.md')

if os.path.exists(GENOME_PATH):
    with open(GENOME_PATH, 'r', encoding='utf-8') as f:
        genome_text = f.read()
else:
    genome_text = ""

system_prompt = f"Decide to HOLD or EXIT based on the frame. If EXIT, provide a reason.\n\nRULES (Genome):\n{genome_text}"

def main():
    state_files = glob.glob(os.path.join(GATE_STATE_DIR, "*.state.json"))
    if not state_files:
        print("No state files found in gen0.")
        return

    print(f"Auditing {len(state_files)} episodes...")
    
    results = []
    tainted_count = 0
    
    for state_file in state_files:
        eid = os.path.basename(state_file).replace('.state.json', '')
        
        with open(state_file, 'r', encoding='utf-8') as f:
            state = json.load(f)
            
        pkt_path = os.path.join(PACKETS_DIR, f"{eid}.json")
        if not os.path.exists(pkt_path):
            print(f"Skipping {eid}: Packet not found.")
            continue
            
        with open(pkt_path, 'r', encoding='utf-8') as f:
            packet = json.load(f)
            
        frames = packet['frames']
        
        # Find the longest frame by string length
        longest_frame_text = ""
        for frame in frames:
            if len(frame['text']) > len(longest_frame_text):
                longest_frame_text = frame['text']
                
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": longest_frame_text}
        ]
        
        # Test true context via Ollama
        try:
            resp = requests.post("http://localhost:11434/api/chat", json={
                "model": "gemma4:e2b",
                "messages": messages,
                "stream": False,
                "options": {"num_ctx": 32000, "num_predict": 1}
            })
            if resp.status_code != 200:
                print(f"Ollama API error on {eid}: {resp.status_code}")
                continue
                
            data = resp.json()
            prompt_tokens = data.get('prompt_eval_count', 0)
        except Exception as e:
            print(f"Request failed for {eid}: {e}")
            continue
            
        # The previous default context limit in Ollama is typically 2048 or 4096. 
        # The forge_harness used default until the patch in doc 116.
        # We will check if prompt_tokens > 4096.
        is_tainted = prompt_tokens > 4096
        if is_tainted:
            tainted_count += 1
            
        results.append({
            "eid": eid,
            "true_tokens": prompt_tokens,
            "effective_ctx": 4096,
            "tainted": "Y" if is_tainted else "N",
            "rerun": "Y" if is_tainted else "N"
        })
        print(f"{eid}: {prompt_tokens} tokens - Tainted: {is_tainted}")

    print(f"\nTotal episodes audited: {len(results)}")
    print(f"Tainted episodes (>4096 tokens): {tainted_count}")
    
    # Save results
    out_path = os.path.join(DOJO_ROOT, 'reports', 'truncation_audit_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}")

if __name__ == "__main__":
    main()
