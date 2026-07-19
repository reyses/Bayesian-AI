import os
import sys
from research.dojo_forge.pipeline.forge_harness import InProcessGate

if __name__ == '__main__':
    eid = '2025_01_02_1735831200_S'
    run_id = 'test-wrong-nonce'
    gate = InProcessGate(run_id, eid)
    
    k, frame_text, nonce = gate.serve()
    print(f"Served frame {k} with nonce {nonce}")
    
    wrong_nonce = "deadbeef12345678"
    print(f"Committing with WRONG nonce: {wrong_nonce}")
    
    try:
        gate.commit(wrong_nonce, "HOLD", "", p_exit=0.5)
    except ValueError as e:
        print(f"Successfully caught expected ValueError: {e}")
        tr_path = os.path.join('research', 'dojo_forge', 'gate_state', run_id, f'{eid}.transcript.jsonl')
        with open(tr_path, 'r') as f:
            lines = f.readlines()
            if 'error' in lines[-1] and 'nonce mismatch' in lines[-1]:
                print("Error event successfully logged.")
            else:
                print("Error event not found in transcript!")
                print(lines[-1])
                sys.exit(1)
        sys.exit(0)
    
    print("FAILED! No exception was raised.")
    sys.exit(1)
