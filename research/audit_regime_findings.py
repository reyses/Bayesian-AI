"""Audit of the segment-regime findings (phases 2-4, 2026-06-12 session).

Verifies the claims in the 2026-06-12 exit report against the artifacts:
  CLAIM 1  "80,717 segments -> 3,029 regime buckets"  (composition check)
  CLAIM 2  "Law of Inertia: Regime 1 perpetuates 63%" (vs BASE RATE - inertia
           is only real if P(R1->R1) materially exceeds the unconditional P(R1))
  CLAIM 3  "Chaos Resolution: NOISE snaps into Regime 1 45%"  (same base-rate test;
           also: phase4's NOISE = unbucketed-but-VALID segments, NOT the removed
           chaos blocks - checked and reported)
  CHECK 4  timeline integrity: phase4 treats consecutive entries of the
           PRISTINE/RECOVERED-filtered list as physically adjacent. Counts how
           many "transitions" actually span removed segments or day boundaries.
  CHECK 5  adjacency self-match: adj[i,i] should be tier 1-2 for (nearly) all i
           (curve i evaluated on its own segment). debug_segment_self_match.py
           records at least one failure ("seg 6 - the one that failed").

Output: reports/findings/2026-06-12_segment_regime_audit.md
"""
import os
import json
import numpy as np
import torch
from collections import Counter
from tqdm import tqdm

ART = 'artifacts'
OUT = 'reports/findings/2026-06-12_segment_regime_audit.md'
DIAG_TIER_OK = (1, 2)  # stage1 break rule tolerates <=5 consecutive out-of-band bars,
                       # so a legitimate self-fit can land in tier 2; 8 = broken mapping.

L = []
def log(s=''):
    print(s)
    L.append(s)

# ── 1. stage2_year_segments.json: composition + timeline integrity ─────────
log('# Segment-Regime Findings Audit (2026-06-12)')
log()
with open(f'{ART}/stage2_year_segments.json') as f:
    segments = json.load(f)
status_counts = Counter(s['status'] for s in segments)
valid = [s for s in segments if s['status'] in ('PRISTINE', 'RECOVERED')]
N = len(valid)
log('## 1. Composition')
log(f'- total segments in stage2_year_segments.json: {len(segments):,}')
log(f'- status counts: {dict(status_counts)}')
log(f'- valid (PRISTINE|RECOVERED) = phase2/3/4 universe: {N:,}')
log()

# timeline integrity: is the valid-list chronologically ordered, and are
# consecutive valid segments physically adjacent (same day, end==start)?
days = [s['day'] for s in valid]
day_sorted = all(days[i] <= days[i+1] for i in range(N-1))
same_day, contiguous, gapped, cross_day = 0, 0, 0, 0
gap_sizes = []
for i in range(N-1):
    if valid[i]['day'] != valid[i+1]['day']:
        cross_day += 1
        continue
    same_day += 1
    gap = valid[i+1]['start_idx'] - valid[i]['end_idx']
    if gap == 0:
        contiguous += 1
    else:
        gapped += 1
        gap_sizes.append(gap)
log('## 2. Phase-4 timeline integrity (transitions = consecutive valid entries)')
log(f'- valid list chronologically day-ordered: {day_sorted}')
log(f'- consecutive pairs: {N-1:,} total = {same_day:,} same-day + {cross_day:,} CROSS-DAY (spurious overnight "transitions")')
if same_day:
    log(f'- same-day pairs physically contiguous (end_idx==next start_idx): {contiguous:,} ({100*contiguous/same_day:.1f}%)')
    log(f'- same-day pairs with a GAP (removed/failed segment between them): {gapped:,} ({100*gapped/same_day:.1f}%)')
if gap_sizes:
    gs = np.array(gap_sizes)
    log(f'- gap size (5s bars): median {np.median(gs):.0f}, p90 {np.percentile(gs, 90):.0f}, max {gs.max()}')
log()

# ── 2. regime_buckets.json: bucket structure ───────────────────────────────
with open(f'{ART}/regime_buckets.json') as f:
    buckets = json.load(f)
sizes = {}
tier12_sizes = {}
for b_id, d in buckets.items():
    sizes[int(b_id)] = d['total_members']
    tier12_sizes[int(b_id)] = len(d['members_tier_1']) + len(d['members_tier_2'])
size_arr = np.array(sorted(sizes.values(), reverse=True))
classified = int(size_arr.sum())
log('## 3. Bucket structure')
log(f'- buckets: {len(buckets):,}; classified segments: {classified:,}/{N:,} ({100*classified/N:.1f}%); NOISE(unbucketed): {N-classified:,}')
log(f'- top-10 bucket sizes: {size_arr[:10].tolist()}')
log(f'- bucket-size median: {np.median(size_arr):.0f}; buckets with <10 members: {(size_arr < 10).sum():,} ({100*(size_arr<10).sum()/len(size_arr):.1f}%)')
log(f'- Regime 1: total {sizes.get(1, 0):,} members of which tier1+2 {tier12_sizes.get(1, 0):,} '
    f'(tier3+4 = loose membership: {sizes.get(1, 0) - tier12_sizes.get(1, 0):,})')
log(f'- share of ALL segments held by Regime 1: {100*sizes.get(1, 0)/N:.1f}%   <- the base-rate that the "inertia" claim must beat')
log()

# ── 3. transition matrix: verify claims vs base rates ──────────────────────
tm = np.load(f'{ART}/transition_matrix.npy')
state = np.zeros(N, dtype=np.int32)
for b_id, d in buckets.items():
    rid = int(b_id)
    for m in d['members_tier_1'] + d['members_tier_2'] + d['members_tier_3'] + d['members_tier_4']:
        state[m] = rid

marg = np.bincount(state, minlength=tm.shape[0]).astype(float) / N
log('## 4. Claim verification (as-built: ALL consecutive pairs, incl. cross-day/gapped)')
for s0, name, claim in ((1, 'Regime 1 -> Regime 1 ("Law of Inertia")', 63.0),
                        (0, 'NOISE -> Regime 1 ("Chaos Resolution")', 45.0)):
    row = tm[s0]
    tot = row.sum()
    p_r1 = 100 * row[1] / tot if tot else float('nan')
    p_stay = 100 * row[s0] / tot if tot else float('nan')
    log(f'- {name}: claimed {claim:.0f}%')
    log(f'    measured P(next=R1 | cur={s0}) = {p_r1:.2f}%  (P(stay)={p_stay:.2f}%, n={tot:,})')
    log(f'    base rate P(any segment = R1) = {100*marg[1]:.2f}%  ->  LIFT = {p_r1 - 100*marg[1]:+.2f} pp '
        f'({p_r1/(100*marg[1]):.2f}x)')
log(f'- transition matrix sparsity: {tm.shape[0]}x{tm.shape[1]} = {tm.size:,} cells, '
    f'nonzero {np.count_nonzero(tm):,} ({100*np.count_nonzero(tm)/tm.size:.2f}%); total transitions {tm.sum():,}')
log()

# honest re-count: same-day AND physically contiguous pairs only
tc_honest = np.zeros_like(tm)
for i in range(N-1):
    if valid[i]['day'] == valid[i+1]['day'] and valid[i+1]['start_idx'] == valid[i]['end_idx']:
        tc_honest[state[i], state[i+1]] += 1
log('## 5. Claim verification (HONEST timeline: same-day, contiguous pairs only)')
for s0, name in ((1, 'Regime 1 -> Regime 1'), (0, 'NOISE -> Regime 1')):
    row = tc_honest[s0]
    tot = row.sum()
    p_r1 = 100 * row[1] / tot if tot else float('nan')
    p_stay = 100 * row[s0] / tot if tot else float('nan')
    log(f'- {name}: P(next=R1) = {p_r1:.2f}%  (P(stay)={p_stay:.2f}%, n={tot:,})  '
    f'vs base rate {100*marg[1]:.2f}%  ->  LIFT {p_r1 - 100*marg[1]:+.2f} pp')
log()

# ── 4. adjacency self-match (diagonal) ──────────────────────────────────────
log('## 6. Adjacency self-match (diagonal = curve i on its own segment)')
try:
    adj = torch.load(f'{ART}/adjacency_matrix.pt', weights_only=False, mmap=True)
    diag = np.empty(N, dtype=np.uint8)
    CH = 4000
    for i in tqdm(range(0, N, CH), desc='diag', ncols=80):
        j = min(i+CH, N)
        diag[i:j] = adj[i:j, i:j].diagonal().numpy() if False else np.diagonal(adj[i:j, :].numpy()[:, i:j])
    dc = Counter(diag.tolist())
    ok = sum(v for k, v in dc.items() if k in DIAG_TIER_OK)
    log(f'- diagonal tier distribution: {dict(sorted(dc.items()))}')
    log(f'- self-match in tier 1-2: {ok:,}/{N:,} ({100*ok/N:.2f}%)')
    log(f'- self-match BROKEN (tier 8 on own data): {dc.get(8, 0):,} ({100*dc.get(8, 0)/N:.2f}%)')
except Exception as e:
    log(f'- SKIPPED (load failed: {e})')
log()

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L) + '\n')
print(f'\nReport written to {OUT}')
