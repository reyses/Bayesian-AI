"""Run regret on new entry strategies to find optimal PnL and direction."""
import pickle, numpy as np, pandas as pd, sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.features_79d import FEATURE_NAMES_79D
from training.regret import compute_all_regrets
from tqdm import tqdm

FEAT_IDX = {name: i for i, name in enumerate(FEATURE_NAMES_79D)}
FEATURES_DIR = 'DATA/FEATURES_79D_5s'
ATLAS_1M = 'DATA/ATLAS/1m'
TICK = 0.25
TV = 0.50


def collect_trades(name, entry_fn, dir_fn, exit_fn):
    feat_files = sorted([f for f in glob.glob(os.path.join(FEATURES_DIR, '*.parquet'))
                         if '2025_' in os.path.basename(f)])
    trades = []
    for fpath in tqdm(feat_files, desc=name, unit='day'):
        day_name = os.path.basename(fpath).replace('.parquet', '')
        df = pd.read_parquet(fpath)
        price_file = os.path.join(ATLAS_1M, f'{day_name}.parquet')
        if not os.path.exists(price_file):
            continue
        prices_df = pd.read_parquet(price_file).sort_values('timestamp')
        price_ts = prices_df['timestamp'].values
        price_close = prices_df['close'].values
        feat_cols = [c for c in df.columns if c != 'timestamp']
        timestamps = df['timestamp'].values

        in_pos = False
        entry_bar = 0
        entry_price = 0
        entry_ts = 0
        direction = ''
        entry_feat = None

        for i in range(len(df)):
            ts = timestamps[i]
            if int(ts) % 60 >= 5:
                continue
            feat = np.array([df.iloc[i][c] for c in feat_cols], dtype=np.float32)
            pidx = int(np.searchsorted(price_ts, ts, side='right')) - 1
            if pidx < 0 or pidx >= len(price_close):
                continue
            price = price_close[pidx]

            if in_pos:
                bars = i - entry_bar
                if exit_fn(feat, entry_feat) or bars > 500:
                    d = direction
                    pnl = ((price - entry_price) / TICK * TV) if d == 'long' else ((entry_price - price) / TICK * TV)
                    trades.append({
                        'day': day_name, 'dir': d, 'entry_price': entry_price,
                        'pnl': pnl, 'held': bars, 'timestamp': entry_ts,
                        'entry_79d': entry_feat.tolist(), 'entry_tier': name,
                        'exit_reason': 'physics', 'path': [], 'approach': [],
                    })
                    in_pos = False
            elif not in_pos:
                if entry_fn(feat):
                    direction = dir_fn(feat)
                    entry_price = price
                    entry_feat = feat.copy()
                    entry_bar = i
                    entry_ts = float(ts)
                    in_pos = True
    return trades


def analyze(name, trades):
    print(f'\n{"="*60}')
    print(f'{name}: {len(trades)} trades')
    print(f'{"="*60}')
    if not trades:
        return

    regret_df = compute_all_regrets(trades)
    actual = regret_df['actual_pnl'].sum()
    optimal = regret_df['best_pnl'].sum()

    print(f'  Actual:  ${actual:,.0f}')
    print(f'  Optimal: ${optimal:,.0f}')
    print(f'  Capture: {actual/max(optimal,1)*100:.1f}%')
    print(f'  Optimal avg: ${regret_df["best_pnl"].mean():.1f}/trade (${optimal/277:.0f}/day)')

    print(f'  Best action:')
    for action, count in regret_df['best_action'].value_counts().items():
        pct = count / len(regret_df) * 100
        avg = regret_df[regret_df['best_action'] == action]['best_pnl'].mean()
        print(f'    {action:<22} {count:>5} ({pct:>4.0f}%)  avg=${avg:.1f}')

    n_counter = regret_df['best_action'].str.contains('counter').sum()
    print(f'  Counter: {n_counter} ({n_counter/len(regret_df)*100:.0f}%) vs Same: {len(regret_df)-n_counter} ({(len(regret_df)-n_counter)/len(regret_df)*100:.0f}%)')

    early = regret_df['best_early_bars_before']
    n_early = (early > 0).sum()
    print(f'  Early entry: {n_early} ({n_early/len(regret_df)*100:.0f}%), mean={early[early>0].mean():.0f} bars')


# REGIME_FLIP
trades = collect_trades('REGIME_FLIP',
    entry_fn=lambda f: f[FEAT_IDX['1m_variance_ratio']] < 0.35 and f[FEAT_IDX['1m_hurst']] < 0.45 and abs(f[FEAT_IDX['1m_z_se']]) < 2.0 and f[FEAT_IDX['1m_variance_ratio']] < 1.0,
    dir_fn=lambda f: 'short' if f[FEAT_IDX['1m_z_se']] > 0 else 'long',
    exit_fn=lambda f, ef: f[FEAT_IDX['1m_variance_ratio']] > 0.7 or f[FEAT_IDX['1m_hurst']] > 0.55 or abs(f[FEAT_IDX['1m_z_se']]) < 0.3,
)
analyze('REGIME_FLIP', trades)

# EXHAUSTION_BAR
trades = collect_trades('EXHAUSTION_BAR',
    entry_fn=lambda f: f[FEAT_IDX['1m_bar_range']] > 80 and abs(f[FEAT_IDX['1m_acceleration']]) > 2.0 and f[FEAT_IDX['1m_acceleration']] * f[FEAT_IDX['1m_velocity']] < 0 and abs(f[FEAT_IDX['1m_z_se']]) < 2.0 and f[FEAT_IDX['1m_variance_ratio']] < 1.0,
    dir_fn=lambda f: 'short' if f[FEAT_IDX['1m_velocity']] > 0 else 'long',
    exit_fn=lambda f, ef: f[FEAT_IDX['1m_bar_range']] < 30 or abs(f[FEAT_IDX['1m_velocity']]) < 0.3,
)
analyze('EXHAUSTION_BAR', trades)

# ABSORPTION
trades = collect_trades('ABSORPTION',
    entry_fn=lambda f: f[FEAT_IDX['1m_vol_rel']] > 1.5 and f[FEAT_IDX['1m_bar_range']] < 20 and f[FEAT_IDX['1m_wick_ratio']] > 0.5 and abs(f[FEAT_IDX['1m_z_se']]) < 2.0 and f[FEAT_IDX['1m_variance_ratio']] < 1.0,
    dir_fn=lambda f: 'short' if f[FEAT_IDX['1m_z_se']] > 0 else 'long',
    exit_fn=lambda f, ef: f[FEAT_IDX['1m_vol_rel']] < 0.5 or f[FEAT_IDX['1m_bar_range']] > 50 or f[FEAT_IDX['1m_wick_ratio']] < 0.25,
)
analyze('ABSORPTION', trades)
