#!/usr/bin/env python3
"""E2c — isolate the E2b divergence: save_state/load_state vs chunk-layout.

Three passes on the same 3 frames (subset for speed):
  A  one-shot eval (the E2 baseline)                       [reference]
  B  two-phase eval: eval(prefix); eval(remainder) — NO save/load
  C  cached: eval(prefix); save; per frame load + eval(remainder)
If B==C != A  -> chunk layout causes divergence; save/load faithful.
   Fix: standardize production labeling on the two-phase layout; cache
   is then exactly equivalent by construction.
If B==A != C  -> save_state/load_state is unfaithful -> E2 dead on this stack.
"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'tools'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'dojo_forge', 'pipeline'))
import bench_common as bc

def main():
    llm, _, _ = bc.load_teacher(11264, engine='cuda')
    import eval_native_ckpt as base
    sys_read = bc.system_prompt_readout()
    reader, method, _, _ = base.resolve_and_selftest(llm, llm.n_vocab(), sys_read)
    print(f"[readout] {method}", flush=True)
    pk = bc.load_packet('2025_01_21_1737469980_S')      # same episode E2 benched
    frames = pk['frames']
    idxs = list(bc.bench_frame_indices(len(frames)))[:3]
    full = {i: llm.tokenize(bc.logit_prompt(frames, i, sys_read).encode('utf-8'),
                            add_bos=True, special=True) for i in idxs}
    from e2_anchor_cache import longest_common_prefix
    plen = longest_common_prefix([full[i] for i in idxs])
    print(f"[e2c] prefix={plen} frames={idxs}")

    def logits_of_oneshot(toks):
        llm.reset(); llm.eval(toks); return np.asarray(reader(), dtype=np.float64)
    def logits_of_twophase(toks):
        llm.reset(); llm.eval(toks[:plen]); llm.eval(list(toks[plen:])); return np.asarray(reader(), dtype=np.float64)

    A = {i: logits_of_oneshot(full[i]) for i in idxs}
    B = {i: logits_of_twophase(full[i]) for i in idxs}
    llm.reset(); llm.eval(full[idxs[0]][:plen]); st = llm.save_state()
    C = {}
    for i in idxs:
        llm.load_state(st); llm.eval(list(full[i][plen:])); C[i] = np.asarray(reader(), dtype=np.float64)

    for i in idxs:
        ab = np.max(np.abs(A[i]-B[i])); bcd = np.max(np.abs(B[i]-C[i])); ac = np.max(np.abs(A[i]-C[i]))
        print(f"f{i:02d}: |A-B|={ab:.2e}  |B-C|={bcd:.2e}  |A-C|={ac:.2e}")
    print("VERDICT: B==C -> layout causes divergence (fix=standardize two-phase); "
          "B==A -> save/load unfaithful (E2 dead)")

if __name__ == '__main__':
    main()
