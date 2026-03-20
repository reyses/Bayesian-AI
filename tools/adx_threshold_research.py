"""ADX threshold research — sweep ADX thresholds against trade outcomes.

Computes 1m ADX for all ATLAS data, matches to IS/OOS trades,
and shows PnL/WR/PF at different ADX thresholds.

Usage: python tools/adx_threshold_research.py [--oos]
"""
import pandas as pd
import numpy as np
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def compute_adx(df, period=14):
    """Compute ADX, DI+, DI- from OHLC dataframe."""
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    n = len(df)

    tr = np.zeros(n)
    dm_plus = np.zeros(n)
    dm_minus = np.zeros(n)

    for i in range(1, n):
        h0, l0, c1 = high[i], low[i], close[i-1]
        h1, l1 = high[i-1], low[i-1]
        tr[i] = max(h0 - l0, abs(h0 - c1), abs(l0 - c1))
        dm_plus[i] = max(h0 - h1, 0) if (h0 - h1 > l1 - l0) else 0
        dm_minus[i] = max(l1 - l0, 0) if (l1 - l0 > h0 - h1) else 0

    # Wilder smoothing (same as NT8 ADX)
    sum_tr = np.zeros(n)
    sum_dp = np.zeros(n)
    sum_dm = np.zeros(n)
    adx = np.full(n, 50.0)

    for i in range(1, n):
        if i < period:
            sum_tr[i] = sum_tr[i-1] + tr[i]
            sum_dp[i] = sum_dp[i-1] + dm_plus[i]
            sum_dm[i] = sum_dm[i-1] + dm_minus[i]
        else:
            sum_tr[i] = sum_tr[i-1] - sum_tr[i-1] / period + tr[i]
            sum_dp[i] = sum_dp[i-1] - sum_dp[i-1] / period + dm_plus[i]
            sum_dm[i] = sum_dm[i-1] - sum_dm[i-1] / period + dm_minus[i]

        if sum_tr[i] > 0:
            di_plus = 100 * sum_dp[i] / sum_tr[i]
            di_minus = 100 * sum_dm[i] / sum_tr[i]
        else:
            di_plus = di_minus = 0

        diff = abs(di_plus - di_minus)
        s = di_plus + di_minus
        dx = 100 * diff / s if s > 0 else 50

        if i >= period:
            adx[i] = ((period - 1) * adx[i-1] + dx) / period

    return adx


def main():
    oos_mode = '--oos' in sys.argv

    # Load trade log
    if oos_mode:
        log_path = 'checkpoints/oos_trade_log.csv'
        data_root = 'DATA/ATLAS_OOS'
        label = 'OOS'
    else:
        log_path = 'checkpoints/oracle_trade_log.csv'
        data_root = 'DATA/ATLAS'
        label = 'IS'

    if not os.path.exists(log_path):
        print(f'ERROR: {log_path} not found')
        return

    trades = pd.read_csv(log_path)
    print(f'{label}: {len(trades)} trades')

    # Load 1m data and compute ADX
    tf_dir = os.path.join(data_root, '1m')
    if not os.path.isdir(tf_dir):
        print(f'ERROR: {tf_dir} not found')
        return

    files = sorted(f for f in os.listdir(tf_dir) if f.endswith('.parquet'))
    print(f'Loading {len(files)} 1m parquet files...')

    chunks = []
    for fn in files:
        df = pd.read_parquet(os.path.join(tf_dir, fn))
        if 'timestamp' in df.columns and not np.issubdtype(df['timestamp'].dtype, np.number):
            df['timestamp'] = df['timestamp'].apply(lambda x: x.timestamp())
        chunks.append(df)
    df_1m = pd.concat(chunks, ignore_index=True).sort_values('timestamp')
    print(f'1m bars: {len(df_1m):,}')

    # Compute ADX
    print('Computing ADX(14)...')
    adx = compute_adx(df_1m, period=14)
    df_1m['adx'] = adx
    timestamps = df_1m['timestamp'].values

    # Match each trade to the 1m ADX at entry time
    print('Matching trades to 1m ADX...')
    trade_adx = []
    for _, t in trades.iterrows():
        entry_ts = t['entry_time']
        idx = np.searchsorted(timestamps, entry_ts, side='right') - 1
        idx = max(0, min(idx, len(adx) - 1))
        trade_adx.append(adx[idx])

    trades['adx_1m'] = trade_adx

    # Summary stats
    print(f'\nADX Distribution at Entry:')
    print(f'  Mean: {trades["adx_1m"].mean():.1f}')
    print(f'  P10:  {trades["adx_1m"].quantile(0.10):.1f}')
    print(f'  P25:  {trades["adx_1m"].quantile(0.25):.1f}')
    print(f'  P50:  {trades["adx_1m"].quantile(0.50):.1f}')
    print(f'  P75:  {trades["adx_1m"].quantile(0.75):.1f}')
    print(f'  P90:  {trades["adx_1m"].quantile(0.90):.1f}')

    # Sweep thresholds
    print(f'\n{"Threshold":>10} {"Trades":>7} {"Blocked":>8} {"WR%":>6} {"PnL":>12} {"$/tr":>8} {"PF":>6}')
    print('-' * 62)

    for thresh in [0, 5, 8, 10, 12, 13, 14, 15, 16, 17, 18, 20, 25, 30]:
        kept = trades[trades['adx_1m'] >= thresh]
        blocked = trades[trades['adx_1m'] < thresh]
        n = len(kept)
        if n == 0:
            continue
        wr = (kept['result'] == 'WIN').mean() * 100
        pnl = kept['actual_pnl'].sum()
        avg = kept['actual_pnl'].mean()
        gw = kept[kept['actual_pnl'] > 0]['actual_pnl'].sum()
        gl = abs(kept[kept['actual_pnl'] < 0]['actual_pnl'].sum())
        pf = gw / gl if gl > 0 else float('inf')
        print(f'{thresh:>10} {n:>7} {len(blocked):>8} {wr:>5.1f}% ${pnl:>10,.2f} ${avg:>7.2f} {pf:>5.2f}')

    # Blocked trade analysis
    print(f'\n--- Blocked trade quality (what we lose at each threshold) ---')
    print(f'{"Threshold":>10} {"Blocked":>8} {"Blk WR%":>8} {"Blk PnL":>12} {"Blk $/tr":>9}')
    print('-' * 52)
    for thresh in [10, 12, 14, 15, 16, 18, 20]:
        blocked = trades[trades['adx_1m'] < thresh]
        if len(blocked) == 0:
            continue
        wr = (blocked['result'] == 'WIN').mean() * 100
        pnl = blocked['actual_pnl'].sum()
        avg = blocked['actual_pnl'].mean()
        print(f'{thresh:>10} {len(blocked):>8} {wr:>7.1f}% ${pnl:>10,.2f} ${avg:>8.2f}')

    # Save results
    out_path = f'reports/findings/adx_threshold_{label.lower()}.txt'
    print(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    main()
