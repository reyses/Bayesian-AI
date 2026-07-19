import os
import json
import glob
import requests

DOJO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), 'research', 'dojo_forge'))
PACKETS_DIR = os.path.join(DOJO_ROOT, 'reports', 'gen0', 'packets')
GEN0_DIR = os.path.join(DOJO_ROOT, 'gate_state', 'gen0')

def get_genome():
    with open(os.path.join(DOJO_ROOT, 'genome', 'GENOME.md'), 'r', encoding='utf-8') as f:
        return f.read()

def main():
    system_prompt = f"Decide to HOLD or EXIT based on the frame. If EXIT, provide a reason.\n\nRULES (Genome):\n{get_genome()}"
    
    contaminated = []
    total = 0
    
    state_files = glob.glob(os.path.join(GEN0_DIR, '*.state.json'))
    print(f"Auditing {len(state_files)} gen-0 episodes...", flush=True)
    
    for sf in state_files:
        with open(sf, 'r') as f:
            state = json.load(f)
            
        eid = state['episode_id']
        # The longest frame is the last one in 'served'
        if not state.get('served'):
            continue
            
        last_served = state['served'][-1]
        frame_idx = last_served['frame']
        
        # load packet
        pkt_path = os.path.join(PACKETS_DIR, f"{eid}.json")
        with open(pkt_path, 'r') as f:
            packet = json.load(f)
            
        frame_text = packet['frames'][frame_idx]['text']
        
        # Send to Ollama to measure true prompt size by giving it a huge num_ctx
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": frame_text}
        ]
        
        # Use options.num_ctx = 32000 to ensure no truncation and get TRUE prompt_eval_count
        resp = requests.post("http://localhost:11434/api/chat", json={
            "model": "gemma4:e2b",
            "messages": messages,
            "stream": False,
            "options": {"num_ctx": 32000}
        }, headers={"Connection": "close"}, timeout=120)
        
        if resp.status_code == 200:
            data = resp.json()
            true_count = data.get('prompt_eval_count', 0)
            
            total += 1
            if true_count > 4096:
                print(f"[{eid}] CONTAMINATED: True prompt size {true_count} > 4096", flush=True)
                contaminated.append(eid)
            else:
                pass
                #print(f"[{eid}] OK: True prompt size {true_count} <= 4096", flush=True)
        else:
            print(f"Error from Ollama: {resp.status_code}", flush=True)
            
    print(f"Total audited: {total}")
    print(f"Contaminated episodes: {len(contaminated)}")
    
    with open('contaminated.json', 'w') as f:
        json.dump(contaminated, f)

if __name__ == '__main__':
    main()
