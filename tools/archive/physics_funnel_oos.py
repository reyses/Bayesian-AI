"""Physics Funnel OOS — Match enriched IS seeds against OOS using full physics.

Each IS seed has entry_fm, entry_dmi_diff, entry_vol, entry_adx, entry_z.
At each OOS bar, compute the current 1m state and find seeds with similar
physics at their entry point.

Direction from seed consensus (not FM sign alone).
Hold from matched seed median duration.
"""
import os, sys, glob, json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core_v2.statistical_field_engine import StatisticalFieldEngine

TICK = 0.25
TICK_VALUE = 0.50

# Load enriched IS seeds
print('Loading enriched IS seeds...')
seed_path = 'DATA/regime_seeds/auto_seeds_all_20260322_161825.json'
with open(seed_path) as f:
    raw = json.load(f)

# Handle both formats
if 'seeds' in raw:
    all_seeds_raw = raw['seeds']
elif 'days' in raw:
    all_seeds_raw = []
    for d in raw['days'].values():
        all_seeds_raw.extend(d.get('seeds', []))
else:
    all_seeds_raw = raw

# Filter to seeds with physics
seeds = [s for s in all_seeds_raw if 'entry_fm' in s]
print(f'Enriched seeds: {len(seeds)} (of {len(all_seeds_raw)} total)')

# Build seed trajectory matrix: 10-bar lookback x 12 features per seed
TRAJ_KEYS = ['fm', 'z', 'dmi_p', 'dmi_m', 'adx', 'vel', 'vol', 'hurst',
             'P_center', 'coherence', 'sigma', 'pid',
             'entropy', 'net_force', 'mr_force', 'rev_prob',
             'P_upper', 'P_lower', 'dmi_p_prev', 'dmi_m_prev', 'adx_prev', 'noise']
TRAJ_LEN = 10
N_FEAT = len(TRAJ_KEYS)

# Only keep seeds with full 10-bar lookback
full_seeds = [s for s in seeds if len(s.get('lookback', [])) >= TRAJ_LEN]
print(f'Seeds with full lookback: {len(full_seeds)} (of {len(seeds)})')

# Build (n_seeds, TRAJ_LEN, N_FEAT) matrix
seed_trajs = np.zeros((len(full_seeds), TRAJ_LEN, N_FEAT))
for si, s in enumerate(full_seeds):
    lb = s['lookback'][-TRAJ_LEN:]  # last 10 bars of lookback
    for bi, bar in enumerate(lb):
        for fi, key in enumerate(TRAJ_KEYS):
            seed_trajs[si, bi, fi] = bar.get(key, 0.0)

# Normalize: z-score per feature across all seeds (flatten to 2D for std)
flat = seed_trajs.reshape(-1, N_FEAT)
feat_means = flat.mean(axis=0)
feat_stds = flat.std(axis=0)
feat_stds[feat_stds < 1e-8] = 1.0
seed_trajs_normed = (seed_trajs - feat_means) / feat_stds
# Flatten to (n_seeds, TRAJ_LEN * N_FEAT) for fast distance
seed_flat = seed_trajs_normed.reshape(len(full_seeds), -1)

seed_mag = np.array([abs(s['change_ticks']) for s in full_seeds])
seed_dir = np.array([1 if s['direction'] == 'LONG' else -1 for s in full_seeds])
seed_bars = np.array([s['n_bars'] for s in full_seeds])

print(f'Trajectory matrix: {seed_flat.shape} ({TRAJ_LEN} bars x {N_FEAT} features)')
print(f'Feature stds: {", ".join(f"{k}={s:.1f}" for k, s in zip(TRAJ_KEYS, feat_stds))}')

# Load OOS 1m
print('Loading OOS...')
oos_files = sorted(glob.glob('DATA/ATLAS_OOS/1m/*.parquet'))
oos_1m = pd.concat([pd.read_parquet(f) for f in oos_files], ignore_index=True)
print(f'OOS bars: {len(oos_1m)}')

engine = StatisticalFieldEngine()
print('Computing OOS states...')
raw_states = engine.batch_compute_states(oos_1m)
states = [s['state'] if s and isinstance(s, dict) and 'state' in s else None for s in raw_states]
closes = oos_1m['close'].values
timestamps = oos_1m['timestamp'].values

# Match parameters
K = 20  # number of nearest seeds to consider
MIN_CONSENSUS = 0.65  # 65%+ must agree on direction
MIN_MAG_PCTILE = 0.25  # skip tiny prior moves

# Prior magnitude distribution (rolling)
mag_window = deque(seed_mag[-200:], maxlen=200)

# Simulate
trades = []
in_trade = False
entry_price = 0.0
trade_dir = ''
entry_bar = 0
entry_ts = 0.0
trade_hold = 0

move_start = closes[0]

traj_buffer = deque(maxlen=TRAJ_LEN)

print(f'Running funnel (K={K}, consensus>{MIN_CONSENSUS}, {TRAJ_LEN}x{N_FEAT} trajectory)...')
for i in range(1, len(states)):
    s = states[i]
    if s is None:
        continue

    # Exit check
    if in_trade and i >= entry_bar + trade_hold:
        ep = closes[i]
        if trade_dir == 'LONG':
            pnl = (ep - entry_price) / TICK
        else:
            pnl = (entry_price - ep) / TICK
        date = datetime.fromtimestamp(entry_ts, tz=timezone.utc).strftime('%Y-%m-%d')
        trades.append({'pnl': pnl, 'dir': trade_dir, 'date': date,
                       'hold': i - entry_bar, 'consensus': trade_consensus})
        in_trade = False
        move_start = closes[i]

    if in_trade:
        continue

    # Build current 10-bar trajectory from rolling buffer
    bar_feats = [
        s.F_momentum, s.z_score, s.dmi_plus, s.dmi_minus,
        s.adx_strength, s.velocity, s.volume_delta, s.hurst_exponent,
        s.P_at_center, getattr(s, 'oscillation_entropy_normalized', 0),
        s.regression_sigma, s.term_pid,
        s.entropy_normalized, s.net_force, s.mean_reversion_force,
        s.reversion_probability, s.P_near_upper, s.P_near_lower,
        s.di_plus_prev, s.di_minus_prev, s.adx_prev, s.swing_noise_ticks,
    ]
    traj_buffer.append(bar_feats)

    if len(traj_buffer) < TRAJ_LEN:
        continue

    # Prior move magnitude
    prior_move = abs(closes[i] - move_start) / TICK
    mag_pctile = sum(1 for m in mag_window if m < prior_move) / max(len(mag_window), 1)
    if mag_pctile < MIN_MAG_PCTILE:
        continue

    # Build normalized trajectory vector
    current_traj = np.array(list(traj_buffer))  # (10, 12)
    current_normed = (current_traj - feat_means) / feat_stds
    current_flat = current_normed.reshape(1, -1)  # (1, 120)

    # Find K nearest seeds by Euclidean distance on full trajectory
    dist = np.linalg.norm(seed_flat - current_flat, axis=1)
    nearest_idx = np.argpartition(dist, K)[:K]

    # Direction consensus from K nearest
    nearest_dirs = seed_dir[nearest_idx]
    n_long = (nearest_dirs > 0).sum()
    n_short = (nearest_dirs < 0).sum()
    consensus = max(n_long, n_short) / K

    if consensus < MIN_CONSENSUS:
        continue

    # Direction from consensus
    if n_long > n_short:
        trade_dir = 'LONG'
    else:
        trade_dir = 'SHORT'

    # Hold from median of matched seeds
    trade_hold = int(np.median(seed_bars[nearest_idx]))
    trade_hold = max(3, min(trade_hold, 20))

    entry_price = closes[i]
    entry_bar = i
    entry_ts = timestamps[i]
    trade_consensus = consensus
    in_trade = True
    mag_window.append(prior_move)
    move_start = closes[i]

t = pd.DataFrame(trades)
if len(t) == 0:
    print('No trades')
    sys.exit()

wins = t[t.pnl > 0]
total = t.pnl.sum() * TICK_VALUE
n_days = t.date.nunique()

print()
print('=' * 70)
print('PHYSICS FUNNEL OOS — Enriched IS seeds, K-NN matching')
print(f'K={K} nearest seeds | Consensus>{MIN_CONSENSUS} | Mag>p{MIN_MAG_PCTILE*100:.0f}')
print('=' * 70)
print(f'  Trades: {len(t):,}')
print(f'  WR: {len(wins)/len(t)*100:.1f}%')
print(f'  Total: {t.pnl.sum():,.0f}t (${total:,.0f})')
print(f'  Avg: {t.pnl.mean():.1f}t (${t.pnl.mean()*TICK_VALUE:.2f})')
print(f'  Avg hold: {t.hold.mean():.1f} bars')
print(f'  Avg consensus: {t.consensus.mean():.2f}')
print(f'  Days: {n_days}')
print(f'  Per day: ${total/n_days:,.0f}')
print()

for d in ['LONG', 'SHORT']:
    sub = t[t.dir == d]
    if len(sub) > 0:
        print(f'  {d}: {len(sub)} trades, WR={(sub.pnl>0).mean()*100:.0f}%, '
              f'avg={sub.pnl.mean():.1f}t, total=${sub.pnl.sum()*TICK_VALUE:,.0f}')

# Consensus buckets
print()
print('  By consensus level:')
for lo, hi, label in [(0.65, 0.75, '65-75%'), (0.75, 0.85, '75-85%'),
                       (0.85, 0.95, '85-95%'), (0.95, 1.01, '95-100%')]:
    sub = t[(t.consensus >= lo) & (t.consensus < hi)]
    if len(sub) > 0:
        wr = (sub.pnl > 0).mean() * 100
        print(f'    {label}: {len(sub)} trades, WR={wr:.0f}%, avg={sub.pnl.mean():.1f}t')

print()
print('  COMPARISON:')
print(f'    Blind flip:             $7,967  $73/day')
print(f'    Current system:         $1,844  $56/day')
print(f'    Pure physics 5m:        $-3,420  $-90/day')
print(f'    Seed funnel (price):    $48  $1/day')
print(f'    Physics funnel (this):  ${total:,.0f}  ${total/n_days:,.0f}/day')
