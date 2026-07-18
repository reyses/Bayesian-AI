"""
Exit Dojo -- CAUSAL telescope packet builder
(research/exit_dojo/builders/telescope_packet_builder.py)

Builds per-episode "telescope" frame packets for the STEPWISE-BLIND full-run dojo.
Unlike the pilot (episode_builder.py -> one ep_NN.md served single-prompt), these
packets are served ONE FRAME AT A TIME by tools/dojo_gate.py, so the agent
mechanically cannot see a future frame before committing the current one.

THE CAUSALITY LAW (non-negotiable, see doc 097 / build_dataset.py:96-111):
    a bar labeled B covers [B, B+period) and CLOSES at B+period. Every row/bar
    that appears in a frame at wall-clock frame_ts MUST satisfy
        row_ts + period <= frame_ts
    at EVERY timeframe INCLUDING the 5s base layer (the phold bug was an omitted
    -period shift at the 5s layer -- it must not recur here). This is ASSERTED
    per-TF, per-frame; any violation is a hard build failure.

NESTED-CADENCE TELESCOPE (memory `telescope-nested-cadence`):
    - Frame 0 = full wide field: every TF's last-CLOSED V2 feature block (named &
      grouped by TF) + per-TF last-closed OHLC context (points-from-entry,
      favorable-signed) + entry info (dir, entry P, price anchored to 0) + the
      pilot per-frame block.
    - Frames k>=1: a TF's block is RE-EMITTED only when that TF's bar has CLOSED
      since the previous frame (its last-closed index advanced). The current
      forming bar at any TF never appears as a bar -- only its closed sub-bars.
    - The pilot per-frame block (drift, leg age/amp/giveback, ER10, vol(5m)+delta,
      KMDR/CLIMAX/HA/PROPP fires w/ age and with/against, close-in-range) is
      emitted EVERY frame (it is the fast local state).

FEATURE SOURCE. The V2 feature store (DATA/ATLAS/FEATURES_5s_v2/<family>/<day>.parquet,
41 families = L0 + L1..L5 x 8 TFs) is already last-closed-bar aligned per higher TF
by build_dataset._last_closed_idx. We read the feature ROW at the last-CLOSED 5s
anchor `ai = searchsorted(ts5, frame_ts - 5) - 1`; every higher-TF family value in
that row is therefore last-closed-as-of ts5[ai] <= frame_ts-5 < frame_ts (causal).
OHLC context + the telescope close-detection use the RAW per-TF stores loaded with a
trailing multi-day buffer (so the last-closed bar exists even for 1D/4h intraday),
indexed store-consistently via `searchsorted(tf_ts, ts5[ai] - period) - 1`.

POPULATION (full run). The phold engagement population (phold_exit_model.py::
engagements): econ_drift_rows.parquet, split=='test', P >= p90(train P) frozen on
train, de-duped to one fire per 60s / day / direction. Restricted to 2025-26 days,
EXCLUDING the 10 pilot days.

SAMPLING. 200 episodes, 60/60/40/40 winner/midflip/instantfail/chop by post-entry
label geometry (the pilot taxonomy: natural_buckets), ONE DISTINCT day per episode.
chop is severely scarce (only 8 distinct non-pilot test days have any 15-min stretch
within +-4pts, 31 within +-5, 60 within +-6); the chop tolerance is graduated to the
MINIMAL value in {4,5,6} that yields >= the chop target of distinct days -- a
documented extension of the pilot's own +-3->+-4 widening. See selection_table.md.

Run:
    python3.11 research/exit_dojo/builders/telescope_packet_builder.py [--seed N]
        [--n-target 200] [--limit K] [--select-only]
Outputs (all under research/exit_dojo/reports/full_run/):
    selection.json                 full 200-episode manifest (params per episode)
    selection_table.md             human-readable selection table + deviations
    packets/<eid>.json             agent-facing telescope frames (gate serves these)
    truth/<eid>.json               ground-truth sidecar (scorer-only, NEVER served)
"""
import os
import sys
import json
import glob
import argparse
from collections import defaultdict
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE)
import episode_builder as eb   # reuse the pilot's verified helpers

# ---- paths ---------------------------------------------------------------------------
FEATROOT = os.path.join(ROOT, 'DATA', 'ATLAS', 'FEATURES_5s_v2')
ATLAS_DIR = os.path.join(ROOT, 'DATA', 'ATLAS')
ECON_DRIFT_PATH = eb.ECON_DRIFT_PATH
FULL_RUN_DIR = os.path.join(eb.DOJO_ROOT, 'reports', 'full_run')
PACKETS_DIR = os.path.join(FULL_RUN_DIR, 'packets')
TRUTH_DIR = os.path.join(FULL_RUN_DIR, 'truth')
SELECTION_JSON = os.path.join(FULL_RUN_DIR, 'selection.json')
SELECTION_TABLE = os.path.join(FULL_RUN_DIR, 'selection_table.md')

# ---- constants (house rule: no bare magic numbers) -----------------------------------
BAR_S = 5                                   # 5s base bar; row B closes at B+BAR_S (build_dataset.py:96)
# 8 timeframes carried by the V2 store, coarse->fine irrelevant; period in seconds.
TF_PERIODS: Dict[str, int] = {
    '5s': 5, '15s': 15, '1m': 60, '5m': 300,
    '15m': 900, '1h': 3600, '4h': 14400, '1D': 86400,
}
TF_ORDER = ['5s', '15s', '1m', '5m', '15m', '1h', '4h', '1D']
FEATURE_LEVELS = ['L1', 'L2', 'L3', 'L4', 'L5']   # per-TF feature families
TF_BUFFER_DAYS = 4                          # trailing raw-TF day files to preload so the last-closed
                                            # bar exists intraday even for 1D/4h (period up to 1 day)

P_PCTL = 90                                 # entry-P percentile defining an engagement (phold: P_PCTL)
DEDUP_S = 60                                # co-fires within this many s / same day / same dir = one eng
TAU_PAD = 10                                # minutes past label end to keep watching (phold: TAU_PAD)
WINDOW_NOFLIP = 45                          # frames for a non-flipping (ride) episode ("up to ~45")
WINDOW_CAP = 60                             # hard cap (label_end+10 capped at 60)
MIN_WINDOW_MIN = 15                         # selection floor -- enough frames to measure an exit
VOL_WINDOW_5S_BARS = eb.VOL_WINDOW_5S_BARS  # 60 x 5s = 5min vol window (pilot vol(5m))

# sampling targets + graduated chop tolerance
BUCKET_TARGETS = {'winner': 60, 'midflip': 60, 'instantfail': 40, 'chop': 40}
CHOP_TOL_LADDER = [4.0, 5.0, 6.0]           # minimal-widening ladder (pilot used 4.0; extend as needed)
SELECTION_SEED = 20260717

PILOT_DAYS = {                              # the 10 pilot days (excluded from the full run)
    '2025_02_04', '2025_05_07', '2025_07_24', '2025_12_02', '2025_08_21',
    '2025_07_01', '2026_01_14', '2026_02_27', '2025_07_25', '2025_07_21',
}

DECISION_CONTRACT = (
    "You are drilling EXITS on a real historical trade replay, served ONE FRAME AT A "
    "TIME through a gate you cannot skip ahead in. Every price number is FAVORABLE-"
    "SIGNED points from entry (entry = 0.00): positive = good for this position, "
    "negative = bad, regardless of LONG/SHORT. For each frame decide HOLD (stay in) or "
    "EXIT (close now). Your FIRST EXIT is binding and ends the episode. Commit HOLD or "
    "EXIT every frame with a one-line reason, then request the next frame. Finish with a "
    "2-3 sentence summary of what signature drove your exit and what you'd watch next."
)


# ================= feature store ======================================================
_FEATCOLS_CACHE: Optional[List[str]] = None
_TF_COLGROUPS_CACHE: Optional[Dict[str, List[str]]] = None


def build_featcols() -> Tuple[List[str], Dict[str, List[str]]]:
    """Ordered V2 feature columns across all 41 families + a TF -> [cols] grouping
    (L1..L5 for each TF). L0 (global time-of-day) is grouped under key 'L0'."""
    global _FEATCOLS_CACHE, _TF_COLGROUPS_CACHE
    if _FEATCOLS_CACHE is not None:
        return _FEATCOLS_CACHE, _TF_COLGROUPS_CACHE
    families = sorted(os.listdir(FEATROOT))
    cols: List[str] = []
    groups: Dict[str, List[str]] = defaultdict(list)
    for fam in families:
        fs = sorted(glob.glob(os.path.join(FEATROOT, fam, '*.parquet')))
        if not fs:
            continue
        c = [x for x in pd.read_parquet(fs[0]).columns if x != 'timestamp']
        cols.extend(c)
        key = 'L0' if fam == 'L0' else fam.split('_', 1)[1]   # 'L1_1m' -> '1m'
        groups[key].extend(c)
    _FEATCOLS_CACHE = cols
    _TF_COLGROUPS_CACHE = dict(groups)
    return cols, _TF_COLGROUPS_CACHE


def load_feature_panel(day: str, featcols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """(ts5 int64, feat float32 [n, len(featcols)]) for one day: all 41 families merged
    on the 5s grid timestamp, reindexed to the stable featcol order. Same construction
    as phold_exit_model.load_day (the feature store is already higher-TF last-closed)."""
    d5 = pd.read_parquet(os.path.join(ATLAS_DIR, '5s', f'{day}.parquet'))
    master = d5['timestamp'].values.astype(np.int64)
    mi = pd.Index(master)
    mats = []
    for fam in sorted(os.listdir(FEATROOT)):
        f = os.path.join(FEATROOT, fam, f'{day}.parquet')
        if not os.path.exists(f):
            continue
        df = pd.read_parquet(f).set_index('timestamp').reindex(mi)
        mats.append(df)
    merged = pd.concat(mats, axis=1).reindex(columns=featcols)
    return master, merged.values.astype(np.float32)


# ================= raw per-TF stores (buffered) =======================================
def _tf_day_files(tf: str) -> List[str]:
    return sorted(glob.glob(os.path.join(ATLAS_DIR, tf, '*.parquet')))


def load_tf_bars(tf: str, day: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Raw OHLC bars for a TF, loaded with a trailing TF_BUFFER_DAYS buffer so the
    last-closed bar exists intraday even for 1D/4h. Returns (ts,o,h,l,c) int64/f64.
    5s is the base grid -- callers use ts5 directly, but this supports it uniformly."""
    files = _tf_day_files(tf)
    names = [os.path.splitext(os.path.basename(p))[0] for p in files]
    if day not in names:
        raise FileNotFoundError(f'{tf} store has no day file for {day}')
    j = names.index(day)
    lo = max(0, j - TF_BUFFER_DAYS)
    frames = [pd.read_parquet(files[k]) for k in range(lo, j + 1)]
    df = pd.concat(frames, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    df = df.drop_duplicates(subset=['timestamp'], keep='last')
    return (df['timestamp'].values.astype(np.int64),
            df['open'].values.astype(np.float64), df['high'].values.astype(np.float64),
            df['low'].values.astype(np.float64), df['close'].values.astype(np.float64))


# ================= population (phold engagements) =====================================
def engagements() -> pd.DataFrame:
    """phold_exit_model.engagements() replica: test split, P>=p90(train), 60s/day/dir
    de-dup, 2025-26, EXCLUDING the pilot days."""
    econ = pd.read_parquet(ECON_DRIFT_PATH, columns=['ts', 'day', 'det', 'is_long', 'P', 'split'])
    thr = float(np.percentile(econ.loc[econ.split == 'train', 'P'].values, P_PCTL))
    sub = econ[(econ.split == 'test') & (econ.P >= thr) &
               (econ.day.str[:4].isin(['2025', '2026'])) &
               (~econ.day.isin(PILOT_DAYS))].copy()
    sub = sub.sort_values(['day', 'is_long', 'ts', 'det']).reset_index(drop=True)
    last: Dict[Tuple[str, bool], int] = {}
    keep = []
    for r in sub.itertuples():
        k = (r.day, bool(r.is_long))
        if k in last and r.ts - last[k] <= DEDUP_S:
            continue
        last[k] = r.ts
        keep.append(r.Index)
    dd = sub.loc[keep].reset_index(drop=True)
    dd.attrs['p90_thr'] = thr
    return dd


def _window_minutes(session_end: int, entry_ts: int, lem: Optional[int]) -> int:
    session_frames = (session_end - entry_ts) // 60
    base = (lem + TAU_PAD) if lem is not None else WINDOW_NOFLIP
    return int(min(WINDOW_CAP, session_frames, max(MIN_WINDOW_MIN, base)))


# ================= sampling ===========================================================
def select_full_run(seed: int = SELECTION_SEED, targets: Dict[str, int] = None):
    """Scan every non-pilot test day once; record per-day first-qualifying engagement per
    bucket; allocate ONE distinct day per episode, scarcest bucket first. Graduate the
    chop tolerance to the minimal ladder value yielding >= the chop target distinct days.
    Returns (selected list, meta dict)."""
    targets = targets or dict(BUCKET_TARGETS)
    eng = engagements()
    thr = eng.attrs['p90_thr']
    days = sorted(eng['day'].unique())

    # ---- pass 1: per-day candidate params + chop capability at each ladder tol --------
    # day -> bucket -> params;  day -> set(tol where chop-capable)
    day_bucket_params: Dict[str, Dict[str, dict]] = defaultdict(dict)
    day_chop_tol: Dict[str, set] = defaultdict(set)
    day_groups = {d: g for d, g in eng.groupby('day', sort=False)}
    for day in tqdm(days, desc='scan days'):
        dd = eb.load_day_data(day)
        if dd is None:
            continue
        for r in day_groups[day].itertuples(index=False):
            ets, isl = int(r.ts), bool(r.is_long)
            lem = eb.label_flip_minute(dd.oracle_ivals, ets, isl,
                                       int(min(WINDOW_CAP, (dd.session_end - ets) // 60)))
            wm = _window_minutes(dd.session_end, ets, lem)
            if wm < MIN_WINDOW_MIN:
                continue
            dp, entry_price = eb.signed_drift_path(dd.ts5, dd.c5, ets, isl, wm)
            # non-chop buckets from label geometry (natural_buckets ignores chop when False)
            for b in eb.natural_buckets(lem, chop=False):
                if b not in day_bucket_params[day]:
                    om = lem if lem is not None else wm
                    day_bucket_params[day][b] = dict(
                        day=day, ts=ets, is_long=isl, P=float(r.P), det=r.det, type=b,
                        window_minutes=wm, label_end_minute=lem, oracle_minute=om,
                        oracle_capture=float(dp[om]), per_minute_forward_drift=dp,
                        entry_price=entry_price, chop_tol=None)
            # chop capability at each ladder tolerance (record the tightest that qualifies)
            for tol in CHOP_TOL_LADDER:
                if eb.is_chop(dp, tol=tol):
                    day_chop_tol[day].add(tol)
                    key = ('chop', tol)
                    if key not in day_bucket_params[day]:
                        om = lem if lem is not None else wm
                        day_bucket_params[day][key] = dict(
                            day=day, ts=ets, is_long=isl, P=float(r.P), det=r.det, type='chop',
                            window_minutes=wm, label_end_minute=lem, oracle_minute=om,
                            oracle_capture=float(dp[om]), per_minute_forward_drift=dp,
                            entry_price=entry_price, chop_tol=tol)

    # ---- choose the minimal chop tolerance that unlocks >= target distinct days -------
    chop_target = targets['chop']
    chop_tol = None
    chop_capacity = {}
    for tol in CHOP_TOL_LADDER:
        cap = sum(1 for d in days if tol in day_chop_tol[d])
        chop_capacity[tol] = cap
        if chop_tol is None and cap >= chop_target:
            chop_tol = tol
    if chop_tol is None:
        chop_tol = CHOP_TOL_LADDER[-1]   # best effort at the widest rung

    # ---- allocation: scarcest bucket first, one distinct day each ---------------------
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(days))
    used_days: set = set()
    selected: List[dict] = []
    order = ['chop', 'instantfail', 'midflip', 'winner']   # scarcest -> most common
    shortfalls = {}
    for bucket in order:
        want = targets[bucket]
        got = 0
        for day in perm:
            if got >= want:
                break
            if day in used_days:
                continue
            key = ('chop', chop_tol) if bucket == 'chop' else bucket
            params = day_bucket_params[day].get(key)
            if params is None:
                continue
            selected.append(params)
            used_days.add(day)
            got += 1
        if got < want:
            shortfalls[bucket] = want - got

    # order deterministically by bucket then day
    order_idx = {b: i for i, b in enumerate(['winner', 'midflip', 'instantfail', 'chop'])}
    selected.sort(key=lambda s: (order_idx[s['type']], s['day']))
    meta = dict(seed=seed, p90_thr=thr, n_days_scanned=len(days),
                chop_tol=chop_tol, chop_capacity=chop_capacity, targets=targets,
                shortfalls=shortfalls, n_selected=len(selected))
    return selected, meta


# ================= telescope frame construction =======================================
def _fmt_feat_block(tf: str, feat_row: np.ndarray, featcol_idx: Dict[str, int],
                    tf_cols: List[str]) -> str:
    parts = []
    for c in tf_cols:
        v = feat_row[featcol_idx[c]]
        short = c.split('_', 2)[-1] if c.count('_') >= 2 else c   # drop 'L3_1m_' prefix noise
        parts.append(f"{short}={'na' if not np.isfinite(v) else f'{v:+.3f}'}")
    return f"  [{tf}] " + " ".join(parts)


def _ohlc_ctx(sign: float, entry_price: float, o, h, l, c) -> Tuple[str, float]:
    bo, bh, bl, bc = (eb.clean0(sign * (o - entry_price)), eb.clean0(sign * (h - entry_price)),
                      eb.clean0(sign * (l - entry_price)), eb.clean0(sign * (c - entry_price)))
    rng = (h - l)
    cir = float((c - l) / rng) if rng > 0 else 0.5   # close-position-in-bar-range [0,1]
    return f"O{bo:+.2f} H{bh:+.2f} L{bl:+.2f} C{bc:+.2f} clsInRng {cir:.2f}", cir


def build_packet(sel: dict, featcols: List[str], tf_groups: Dict[str, List[str]],
                 aux_data: Dict[str, pd.DataFrame], dd) -> Tuple[dict, dict, int]:
    """Returns (packet dict, truth dict, n_causality_checks). Raises AssertionError on
    ANY causality violation (row_ts + period > frame_ts at any TF, any frame)."""
    ts5 = dd.ts5
    sign = 1.0 if sel['is_long'] else -1.0
    entry_ts = sel['ts']
    entry_price = sel['entry_price']
    window = sel['window_minutes']

    featcol_idx = {c: i for i, c in enumerate(featcols)}
    ts5_panel, feat = load_feature_panel(sel['day'], featcols)
    assert np.array_equal(ts5_panel, ts5), 'feature-panel 5s grid != 5s OHLC grid'

    # raw per-TF stores (buffered) for OHLC context + telescope close-detection
    tf_raw = {tf: (ts5, dd.o5, dd.h5, dd.l5, dd.c5) if tf == '5s' else load_tf_bars(tf, sel['day'])
              for tf in TF_ORDER}

    # pilot local-state series
    piv_ts_arr, amp_arr, giveback_arr = eb.track_leg_state(ts5, dd.c5)
    ohlc1m = eb.build_1m_ohlc(ts5, dd.o5, dd.h5, dd.l5, dd.c5)
    er10 = eb.compute_er10_series(ohlc1m['c'])
    day_aux = {name: df[df['day'] == sel['day']] for name, df in aux_data.items()}

    n_checks = 0
    prev_tf_idx: Dict[str, int] = {tf: -2 for tf in TF_ORDER}
    prev_vol5m = None
    frames = []

    for m in range(0, window + 1):
        frame_ts = entry_ts + m * 60

        # --- base 5s anchor (THE phold bug site): last CLOSED 5s bar ---
        ai = int(np.searchsorted(ts5, frame_ts - BAR_S, side='right') - 1)
        if ai < 0:
            # pre-warmup frame: emit a minimal placeholder, no feature blocks
            frames.append({'frame': m, 'text': f"[t={m}m] (warmup: no closed 5s bar yet)"})
            continue
        assert ts5[ai] + BAR_S <= frame_ts, (
            f'CAUSALITY[5s] ep {sel["day"]}@{entry_ts} frame {m}: '
            f'ts5[ai]={ts5[ai]}+{BAR_S} > frame_ts={frame_ts}')
        n_checks += 1
        anchor_ts = int(ts5[ai])
        feat_row = feat[ai]

        # --- per-TF last-closed index (store-consistent) + causality assert + telescope ---
        tf_blocks = []
        for tf in TF_ORDER:
            period = TF_PERIODS[tf]
            tts, to, th, tl, tc = tf_raw[tf]
            if tf == '5s':
                ti = ai
            else:
                ti = int(np.searchsorted(tts, anchor_ts - period, side='right') - 1)
            if ti < 0:
                continue
            assert tts[ti] + period <= frame_ts, (
                f'CAUSALITY[{tf}] ep {sel["day"]}@{entry_ts} frame {m}: '
                f'bar_ts={tts[ti]}+{period} > frame_ts={frame_ts}')
            n_checks += 1
            if ti == prev_tf_idx[tf] and m > 0:
                continue                              # bar did not close since last frame -> telescope skip
            prev_tf_idx[tf] = ti
            ohlc_txt, _ = _ohlc_ctx(sign, entry_price, to[ti], th[ti], tl[ti], tc[ti])
            feat_txt = _fmt_feat_block(tf, feat_row, featcol_idx, tf_groups.get(tf, []))
            tf_blocks.append(f"  [{tf}] closed-bar (pts fr entry): {ohlc_txt}\n{feat_txt}")

        # --- pilot local-state block (every frame) ---
        px = sel['per_minute_forward_drift'][m]
        leg_age_min = (anchor_ts - int(piv_ts_arr[ai])) / 60.0
        amp = float(amp_arr[ai])
        giveback = float(giveback_arr[ai])
        vlo = max(0, ai - VOL_WINDOW_5S_BARS + 1)
        vwin = dd.c5[vlo: ai + 1]
        vol5m = float(np.std(vwin, ddof=1)) if len(vwin) >= 2 else float('nan')
        vol_delta = (vol5m - prev_vol5m) if (prev_vol5m is not None and np.isfinite(prev_vol5m)
                                             and np.isfinite(vol5m)) else float('nan')
        prev_vol5m = vol5m
        cur_bucket = int(anchor_ts // 60)
        er_val = er10.get(cur_bucket - 1, np.nan)
        er_txt = f"{er_val:.2f}" if np.isfinite(er_val) else "n/a"
        fires_txt = eb.aux_fires_text(day_aux, frame_ts, sel['is_long'])
        vol_delta_txt = f"{vol_delta:+.1f}" if np.isfinite(vol_delta) else "n/a"
        local = (f"  local: px {eb.clean0(px):+.2f}pts | leg age {leg_age_min:.0f}m amp {amp:.1f}pts "
                 f"giveback {giveback * 100:.0f}% | vol(5m) {vol5m:.1f}pts (d {vol_delta_txt}) | "
                 f"ER10 {er_txt} | fires<=3m: {fires_txt}")

        header = (f"[t={m}m]" + ("  == WIDE FIELD (frame 0) ==" if m == 0 else
                  ("  (TF blocks: " + ", ".join(b.strip().split(']')[0].strip('[') for b in tf_blocks)
                   + " re-emitted)" if tf_blocks else "  (no TF bar closed this minute)")))
        parts = [header]
        if tf_blocks:
            parts.extend(tf_blocks)
        parts.append(local)
        frames.append({'frame': m, 'text': "\n".join(parts)})

    packet = dict(
        episode_id=_eid(sel),
        meta=dict(direction='LONG' if sel['is_long'] else 'SHORT',
                  entry_P=round(sel['P'], 4), window_minutes=window,
                  n_frames=len(frames), n_causality_checks=n_checks,
                  sign_convention='favorable-signed points from entry (entry=0.00)',
                  decision_contract=DECISION_CONTRACT),
        frames=frames,
    )
    truth = dict(
        episode_id=_eid(sel), type=sel['type'], is_long=sel['is_long'], entry_ts=entry_ts,
        entry_price=entry_price, det=sel['det'], P=sel['P'], window_minutes=window,
        label_end_minute=sel['label_end_minute'], oracle_capture=sel['oracle_capture'],
        oracle_minute=sel['oracle_minute'], per_minute_forward_drift=sel['per_minute_forward_drift'],
        chop_tol=sel['chop_tol'], real_day=sel['day'], n_causality_checks=n_checks,
    )
    return packet, truth, n_checks


def _eid(sel: dict) -> str:
    d = sel['day']
    return f"{d}_{sel['ts']}_{'L' if sel['is_long'] else 'S'}"


# ================= writers ============================================================
def write_selection_table(selected: List[dict], meta: dict):
    L = []
    A = L.append
    A('# Exit Dojo -- full-run selection table (causal telescope sandbox)')
    A('')
    A(f"Seed={meta['seed']}; scanned {meta['n_days_scanned']} non-pilot 2025-26 test days; "
      f"phold engagement population (entry-P p{P_PCTL} frozen on train = {meta['p90_thr']:.5f}, "
      f"60s/day/dir de-dup). Target {sum(meta['targets'].values())} episodes "
      f"({', '.join(f'{v} {k}' for k, v in meta['targets'].items())}), ONE DISTINCT DAY each, "
      f"pilot's 10 days excluded.")
    A('')
    A('## Declared deviation -- graduated chop tolerance')
    A(f"chop is extremely scarce in the high-confidence entry pool: distinct non-pilot "
      f"test days with ANY 15-min stretch within tolerance = "
      f"{', '.join(f'+-{int(t)}pt: {c}' for t, c in meta['chop_capacity'].items())}. The chop "
      f"target ({meta['targets']['chop']} distinct days) is unreachable at the pilot's +-4pt; the "
      f"tolerance is graduated to the MINIMAL ladder value that reaches it: "
      f"**+-{int(meta['chop_tol'])}pt**. This extends the pilot's own +-3->+-4 widening "
      f"(episode_builder.CHOP_TOL_PTS) on the same rationale -- decile-9/p90 fires essentially "
      f"never stay flat, so 'chop' must be defined at a looser flatness band to be populatable.")
    if meta['shortfalls']:
        A('')
        A(f"**SHORTFALL (reviewer note):** could not fully fill {meta['shortfalls']} from distinct "
          f"days even at the widest rung; table below is what was fillable.")
    A('')
    A('| ep | eid | type | real day | det | entry ts (UTC) | dir | P | window(min) | '
      'label_end_min | oracle cap (pts) | chop_tol |')
    A('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for i, s in enumerate(selected, 1):
        ctol = '' if s['chop_tol'] is None else f"+-{int(s['chop_tol'])}pt"
        A(f"| {i:03d} | {_eid(s)} | {s['type']} | {s['day']} | {s['det']} | {s['ts']} | "
          f"{'LONG' if s['is_long'] else 'SHORT'} | {s['P']:.3f} | {s['window_minutes']} | "
          f"{s['label_end_minute']} | {s['oracle_capture']:+.2f} | {ctol} |")
    os.makedirs(FULL_RUN_DIR, exist_ok=True)
    with open(SELECTION_TABLE, 'w', encoding='utf-8') as f:
        f.write("\n".join(L))


def _manifest_row(s: dict) -> dict:
    r = dict(s)
    r['eid'] = _eid(s)
    return r


# ================= main ===============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=SELECTION_SEED)
    ap.add_argument('--n-target', type=int, default=None,
                    help='override total (keeps the 60/60/40/40 ratio scaled)')
    ap.add_argument('--limit', type=int, default=None,
                    help='build only the first K packets (selection + table still full)')
    ap.add_argument('--select-only', action='store_true',
                    help='write selection.json + selection_table.md only; build no packets')
    ap.add_argument('--only', default=None,
                    help='comma-separated eids to build (subset; selection/table still full)')
    ap.add_argument('--selection', default=None,
                    help='external selection.json (e.g. wrongdir) to build packets FROM; '
                         'skips select_full_run + selection writers (doc 099 reuse path)')
    ap.add_argument('--outdir', default=None,
                    help='override output dir; packets/ + truth/ are created under it '
                         '(default: reports/full_run/)')
    args = ap.parse_args()

    # --outdir redirects the packet/truth targets (additive; default keeps full_run) ----
    global PACKETS_DIR, TRUTH_DIR
    out_root = os.path.abspath(args.outdir) if args.outdir else FULL_RUN_DIR
    PACKETS_DIR = os.path.join(out_root, 'packets')
    TRUTH_DIR = os.path.join(out_root, 'truth')
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(PACKETS_DIR, exist_ok=True)
    os.makedirs(TRUTH_DIR, exist_ok=True)

    # --selection: consume an externally-built manifest (wrongdir), build packets only ---
    if args.selection:
        with open(args.selection, encoding='utf-8') as f:
            ext = json.load(f)
        selected = ext['episodes']
        print(f'[select] external selection: {len(selected)} episodes from {args.selection}; '
              f'building into {out_root} (no selection writers)')
    else:
        targets = dict(BUCKET_TARGETS)
        if args.n_target:
            scale = args.n_target / sum(BUCKET_TARGETS.values())
            targets = {k: int(round(v * scale)) for k, v in BUCKET_TARGETS.items()}

        print(f'[select] scanning for {sum(targets.values())} episodes {targets} seed={args.seed} ...')
        selected, meta = select_full_run(seed=args.seed, targets=targets)
        print(f"[select] chose {len(selected)} episodes; chop_tol=+-{int(meta['chop_tol'])}pt; "
              f"shortfalls={meta['shortfalls'] or 'none'}")

        write_selection_table(selected, meta)
        with open(SELECTION_JSON, 'w', encoding='utf-8') as f:
            json.dump(dict(meta=meta, episodes=[_manifest_row(s) for s in selected]), f, indent=2)
        print(f'[select] wrote {SELECTION_TABLE}\n[select] wrote {SELECTION_JSON}')

        if args.select_only:
            print('[select-only] done (no packets built).')
            return

    featcols, tf_groups = build_featcols()
    aux_data = eb.load_aux_data()
    if args.only:
        want = set(args.only.split(','))
        build_list = [s for s in selected if _eid(s) in want]
        missing = want - {_eid(s) for s in build_list}
        if missing:
            print(f'[build] WARN: --only eids not in selection: {missing}')
    elif args.limit is not None:
        build_list = selected[:args.limit]
    else:
        build_list = selected
    print(f'[build] building {len(build_list)} packets (resume-safe) ...')
    day_cache = {}
    built = skipped = 0
    for sel in tqdm(build_list, desc='packets'):
        eid = _eid(sel)
        pkt_path = os.path.join(PACKETS_DIR, f'{eid}.json')
        if os.path.exists(pkt_path):
            skipped += 1
            continue
        dd = day_cache.get(sel['day'])
        if dd is None:
            dd = eb.load_day_data(sel['day'])
            day_cache[sel['day']] = dd
        packet, truth, n = build_packet(sel, featcols, tf_groups, aux_data, dd)
        with open(pkt_path, 'w', encoding='utf-8') as f:
            json.dump(packet, f)
        with open(os.path.join(TRUTH_DIR, f'{eid}.json'), 'w', encoding='utf-8') as f:
            json.dump(truth, f, indent=2)
        built += 1
    print(f'[build] done: built {built}, skipped(existing) {skipped}, total packets '
          f'{built + skipped} in {PACKETS_DIR}')


if __name__ == '__main__':
    main()
