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
    system_tokens = llm.tokenize(f"<start_of_turn>user\n{system_prompt}\n<end_of_turn>\n".encode("utf-8"))
    
    t0 = time.time()
    llm.eval(system_tokens)
    t1 = time.time()
    cold_time = t1 - t0
    
    state = llm.save_state()
    
    llm.reset()
    t0 = time.time()
    llm.load_state(state)
    t1 = time.time()
    prefix_time = t1 - t0

    print(f"Prefix Cache - Cold: {cold_time:.3f}s | Cached (restored): {prefix_time:.3f}s")
    
    while True:
        k, frame_text, nonce = gate.serve()
        if k is None:
            break
            
        llm.load_state(state)
        frame_tokens = llm.tokenize(f"<start_of_turn>user\n{frame_text}\n<end_of_turn>\n<start_of_turn>model\n".encode("utf-8"))
        
        llm.eval(frame_tokens)
        
        output = ""
        while True:
            token = llm.sample(grammar=grammar)
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

        p_exit = 0.5
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
            "options": {"temperature": 0.0, "num_ctx": 4096}
        })
        
        if resp.status_code != 200:
            print(f"Ollama API error: {resp.text}")
            break
            
        data = resp.json()
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
    ap.add_argument('--model-blob', default='/mnt/c/Users/reyse/.ollama/models/blobs/sha256-4e30e2665218745ef463f722c0bf86be0cab6ee676320f1cfadf91e989107448')
    ap.add_argument('--fallback-url', default=None, help='http://localhost:11434/api/chat')
    ap.add_argument('--run-id', required=True, help='Unique ID for this run (e.g. F1-run-1)')
    args = ap.parse_args()

    system_prompt = "Decide to HOLD or EXIT based on the frame. If EXIT, provide a reason."

    if args.fallback_url:
        for eid in args.episodes:
            gate = InProcessGate(args.run_id, eid)
            run_episode_ollama(gate, args.fallback_url, system_prompt)
    else:
        from llama_cpp import Llama, LlamaGrammar
        grammar = LlamaGrammar.from_string(GBNF_GRAMMAR)
        llm = Llama(model_path=args.model_blob, n_gpu_layers=-1, n_ctx=4096, seed=42, temperature=0.0)
        
        for eid in args.episodes:
            gate = InProcessGate(args.run_id, eid)
            run_episode_llama(gate, llm, system_prompt, grammar)
