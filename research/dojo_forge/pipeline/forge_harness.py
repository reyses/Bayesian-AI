"""
DOJO FORGE - Phase F1
Local-model generational dojo harness.
"""
import os
import sys
import json
import time
import secrets
import argparse
import requests

NONCE_BYTES = 8
DOJO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
EXIT_DOJO_ROOT = os.path.abspath(os.path.join(DOJO_ROOT, '..', 'exit_dojo'))
FULL_RUN_DIR = os.environ.get('DOJO_RUN_DIR') or os.path.join(EXIT_DOJO_ROOT, 'reports', 'full_run')
PACKETS_DIR = os.path.join(FULL_RUN_DIR, 'packets')
GATE_STATE_DIR = os.path.join(DOJO_ROOT, 'gate_state')

GBNF_GRAMMAR = r'''
root ::= "HOLD" | "EXIT: " string
string ::= [^\n]+
'''

def _paths(run_id: str, eid: str):
    gate_dir = os.path.join(GATE_STATE_DIR, run_id)
    return (os.path.join(PACKETS_DIR, f'{eid}.json'),
            os.path.join(gate_dir, f'{eid}.state.json'),
            os.path.join(gate_dir, f'{eid}.transcript.jsonl'))

def _load_packet(eid: str) -> dict:
    pkt_path = os.path.join(PACKETS_DIR, f'{eid}.json')
    if not os.path.exists(pkt_path):
        raise FileNotFoundError(f"Missing packet JSON {pkt_path}")
    with open(pkt_path, encoding='utf-8') as f:
        return json.load(f)

def _save_state(run_id: str, eid: str, state: dict):
    gate_dir = os.path.join(GATE_STATE_DIR, run_id)
    os.makedirs(gate_dir, exist_ok=True)
    _, st_path, _ = _paths(run_id, eid)
    tmp = st_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, st_path)

def _log(run_id: str, eid: str, event: dict):
    gate_dir = os.path.join(GATE_STATE_DIR, run_id)
    os.makedirs(gate_dir, exist_ok=True)
    _, _, tr_path = _paths(run_id, eid)
    event = dict(ts=time.time(), **event)
    with open(tr_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(event) + '\n')

class InProcessGate:
    def __init__(self, run_id: str, eid: str):
        self.run_id = run_id
        self.eid = eid
        self.packet = _load_packet(eid)
        self.frames = self.packet['frames']
        self.state = dict(episode_id=eid, served=[], commits=[], pending=None, closed=False,
                          exit_frame=None, finished=False, summary=None)
        
        _, st_path, tr_path = _paths(run_id, eid)
        if os.path.exists(st_path): os.remove(st_path)
        if os.path.exists(tr_path): os.remove(tr_path)

    def serve(self):
        if self.state['finished']:
            raise ValueError("Episode already finished")
        k = len(self.state['commits'])
        if self.state['closed'] or k >= len(self.frames):
            return None, None
        
        frame = self.frames[k]
        nonce = secrets.token_hex(NONCE_BYTES)
        self.state['pending'] = dict(frame=k, nonce=nonce)
        self.state['served'].append(dict(frame=k, nonce=nonce, ts=time.time()))
        _save_state(self.run_id, self.eid, self.state)
        _log(self.run_id, self.eid, dict(event='serve', frame=k, nonce=nonce))
        return k, frame['text'], nonce

    def commit(self, nonce: str, decision: str, reason: str, p_exit: float = None):
        if self.state['finished']:
            raise ValueError("Episode already finished")
        if not self.state['pending']:
            raise ValueError("No frame pending")
        
        k = self.state['pending']['frame']
        expected_nonce = self.state['pending']['nonce']
        
        if nonce != expected_nonce:
            _log(self.run_id, self.eid, dict(event='error', reason='nonce mismatch', expected=expected_nonce, got=nonce))
            raise ValueError(f"Nonce mismatch! Expected {expected_nonce}, got {nonce}")
        
        commit = dict(frame=k, nonce=nonce, decision=decision, reason=reason)
        if p_exit is not None:
            commit['p_exit'] = p_exit

        self.state['commits'].append(commit)
        self.state['pending'] = None
        
        if decision == 'EXIT':
            self.state['closed'] = True
            self.state['exit_frame'] = k
            
        _save_state(self.run_id, self.eid, self.state)
        _log(self.run_id, self.eid, dict(event='commit', **commit))
        return self.state['closed']

    def finish(self):
        if self.state['finished']:
            return
        self.state['finished'] = True
        self.state['summary'] = 'Finished by dojo_forge'
        _save_state(self.run_id, self.eid, self.state)
        _log(self.run_id, self.eid, dict(event='finish', summary=self.state['summary'],
                            exit_frame=self.state['exit_frame'], n_commits=len(self.state['commits'])))


def run_episode_llama(gate: InProcessGate, llm, system_prompt: str, grammar):
    print(f"--- Running {gate.eid} via llama_cpp ---")
    start_time = time.time()
    total_tokens = 0

    llm.reset()
    sys_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    sys_tokens = llm.tokenize(sys_text.encode('utf-8'), add_bos=False, special=True)
    t0 = time.time()
    for i in range(0, len(sys_tokens), llm.n_batch):
        llm.eval(sys_tokens[i:i + llm.n_batch])
    t1 = time.time()
    cold_time = t1 - t0
    prefix_time = 0.0

    print(f"Prefix Cache - Cold: {cold_time:.3f}s | Cached (restored): {prefix_time:.3f}s")
    
    sys_len = len(sys_tokens)
    
    # Get token IDs for "EXIT" and "HOLD"
    token_exit = llm.tokenize(b"EXIT", add_bos=False)[0]
    token_hold = llm.tokenize(b"HOLD", add_bos=False)[0]

    while True:
        k, frame_text, nonce = gate.serve()
        if k is None:
            break
            
        llm._ctx.kv_cache_seq_rm(-1, sys_len, -1)
        llm.n_tokens = sys_len
        
        user_prefix = llm.tokenize(b"<|im_start|>user\n", add_bos=False, special=True)
        assistant_suffix = llm.tokenize(b"<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n", add_bos=False, special=True)
        frame_tokens = llm.tokenize(frame_text.encode("utf-8"), add_bos=False, special=False)
        
        max_frame_len = llm._n_ctx - sys_len - len(user_prefix) - len(assistant_suffix) - 128
        if max_frame_len < 0:
            max_frame_len = 0
            
        if len(frame_tokens) > max_frame_len:
            frame_tokens = frame_tokens[-max_frame_len:]
            
        new_tokens = user_prefix + frame_tokens + assistant_suffix
        
        for i in range(0, len(new_tokens), llm.n_batch):
            llm.eval(new_tokens[i:i + llm.n_batch])
        
        import numpy as np
        # Extract P(EXIT) from logits
        logits = np.array(llm.eval_logits[-1]) if hasattr(llm, 'eval_logits') else np.array(llm.scores) if hasattr(llm, 'scores') else np.array(llm._scores) if hasattr(llm, '_scores') else None
        
        p_exit = 0.5
        if logits is not None:
            logit_exit = logits[token_exit]
            logit_hold = logits[token_hold]
            exp_exit = np.exp(logit_exit)
            exp_hold = np.exp(logit_hold)
            p_exit = exp_exit / (exp_exit + exp_hold + 1e-9)
            print(f"Frame {k}: P(EXIT)={p_exit:.4f} (logits: EXIT={logit_exit:.2f}, HOLD={logit_hold:.2f})")

        output = ""
        while True:
            if grammar is not None:
                token = llm.sample(grammar=grammar)
            else:
                token = llm.sample()
            if token == llm.token_eos():
                break
            output += llm.detokenize([token]).decode('utf-8', errors='ignore')
            llm.eval([token])
            if len(output) > 50:
                break

        decision = "HOLD"
        reason = ""
        if output.startswith("EXIT"):
            decision = "EXIT"
            if ":" in output:
                reason = output.split(":", 1)[1].strip()

        closed = gate.commit(nonce, decision, reason, p_exit)
        total_tokens += len(frame_tokens)
        
        if closed:
            break

    gate.finish()
    run_time = time.time() - start_time
    print(f"Episode {gate.eid} finished in {run_time:.2f}s. Tokens/s: {total_tokens/run_time:.2f}")

def run_episode_ollama(gate: InProcessGate, url: str, system_prompt: str):
    print(f"--- Running {gate.eid} via ollama HTTP fallback ---")
    start_time = time.time()
    total_tokens = 0
    
    while True:
        k, frame_text, nonce = gate.serve()
        if k is None:
            break
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": frame_text}
        ]
        
        resp = requests.post(url, json={
            "model": "gemma4:e2b",
            "messages": messages,
            "stream": False,
            "format": {
                "type": "object",
                "properties": {
                    "decision": {"type": "string", "enum": ["HOLD", "EXIT"]},
                    "reason": {"type": "string"}
                },
                "required": ["decision", "reason"]
            },
            "options": {"temperature": 0.0, "num_ctx": 8192}
        })
        
        if resp.status_code != 200:
            print(f"Ollama API error: {resp.text}")
            break
            
        data = resp.json()
        
        prompt_eval = data.get('prompt_eval_count', 0)
        if prompt_eval >= 8192:
            raise RuntimeError(f"Ollama silent truncation detected: prompt size {prompt_eval} exceeded num_ctx!")
            
        output = data['message']['content'].strip()
        
        try:
            data_json = json.loads(output)
            decision = data_json.get("decision", "HOLD")
            reason = data_json.get("reason", "")
        except json.JSONDecodeError:
            decision = "HOLD"
            reason = "Failed to parse json"
            if "EXIT" in output.upper():
                decision = "EXIT"
        
        closed = gate.commit(nonce, decision, reason, None)
        if closed:
            break
            
    gate.finish()
    print(f"Episode {gate.eid} finished via fallback in {time.time()-start_time:.2f}s")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', nargs='+', required=True)
    # qwen3:14b (a8cc) on the native-Linux ollama store (2026-07-22). Was the gemma4 (4e30)
    # blob on a dead /mnt/c path — gemma4 is also truncated; qwen3 is THE teacher.
    ap.add_argument('--model-blob', default='/media/moi/WindowsCode/ollama/models/blobs/sha256-a8cc1361f3145dc01f6d77c6c82c9116b9ffe3c97b34716fe20418455876c40e')
    ap.add_argument('--fallback-url', default=None, help='http://localhost:11434/api/chat')
    ap.add_argument('--run-id', required=True, help='Unique ID for this run (e.g. F1-run-1)')
    args = ap.parse_args()

    genome_path = os.path.join(DOJO_ROOT, 'genome', 'GENOME.md')
    if os.path.exists(genome_path):
        with open(genome_path, 'r', encoding='utf-8') as f:
            genome_text = f.read()
    else:
        genome_text = ""

    system_prompt = f"Decide to HOLD or EXIT based on the frame. If EXIT, provide a reason.\n\nRULES (Genome):\n{genome_text}"

    if args.fallback_url:
        for eid in args.episodes:
            gate = InProcessGate(args.run_id, eid)
            run_episode_ollama(gate, args.fallback_url, system_prompt)
    else:
        from llama_cpp import Llama
        print(f"DEBUG: Starting Llama initialization with {args.model_blob}...")
        llm = Llama(model_path=args.model_blob, n_gpu_layers=20, n_ctx=4096, seed=42, temperature=0.0, logits_all=True, verbose=True)
        print("DEBUG: Llama initialization complete!")
        
        for eid in args.episodes:
            print(f"DEBUG: Running episode {eid}...")
            gate = InProcessGate(args.run_id, eid)
            run_episode_llama(gate, llm, system_prompt, None)
