"""Bar color continuation: payoff asymmetry and simple strategy PnL."""
import numpy as np, pandas as pd, sys, os
from scipy import stats
from scipy.stats import pointbiserialr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

tf = sys.argv[1] if len(sys.argv) > 1 else '1m'
import glob
files = sorted(glob.glob(f'DATA/ATLAS/{tf}/*.parquet'))
df = pd.concat([pd.read_parquet(f) for f in files[-3:]], ignore_index=True)
df = df.sort_values('timestamp').reset_index(drop=True)

closes = df['close'].values
opens = df['open'].values
highs = df['high'].values
lows = df['low'].values
n = len(df)

green = closes > opens
red = closes < opens

# Continuation vs reversal bar sizes
same_body, flip_body = [], []
for i in range(n - 1):
    if not (green[i] or red[i]):
        continue
    body = abs(closes[i+1] - opens[i+1])
    is_same = (green[i] and green[i+1]) or (red[i] and red[i+1])
    if is_same:
        same_body.append(body)
    elif (green[i] and red[i+1]) or (red[i] and green[i+1]):
        flip_body.append(body)

sb = np.array(same_body)
fb = np.array(flip_body)
t, p = stats.ttest_ind(sb, fb)

print(f'PAYOFF ASYMMETRY ({tf}, {n:,} bars)')
print(f'  Continuation bar body: mean={sb.mean():.2f}  median={np.median(sb):.2f}  (N={len(sb)})')
print(f'  Reversal bar body:     mean={fb.mean():.2f}  median={np.median(fb):.2f}  (N={len(fb)})')
print(f'  Ratio: {sb.mean()/fb.mean():.3f}x  p={p:.2e}')
print()

# Simple strategy: bet same color, settle at next close
pnls = []
for i in range(n - 1):
    if not (green[i] or red[i]):
        continue
    if green[i]:
        pnl = closes[i+1] - closes[i]
    else:
        pnl = closes[i] - closes[i+1]
    pnls.append(pnl)

pnls = np.array(pnls)
wins = pnls > 0
losses = pnls < 0
flat = pnls == 0

print(f'SIMPLE STRATEGY: bet same color, exit next bar close')
print(f'  Trades: {len(pnls):,}')
print(f'  Win rate: {wins.mean()*100:.1f}%')
print(f'  Avg win:  {pnls[wins].mean():.2f} pts (${pnls[wins].mean()*2:.2f})')
print(f'  Avg loss: {pnls[losses].mean():.2f} pts (${pnls[losses].mean()*2:.2f})')
print(f'  Win/Loss ratio: {pnls[wins].mean()/abs(pnls[losses].mean()):.2f}')
print(f'  Avg PnL/trade: {pnls.mean():.4f} pts (${pnls.mean()*2:.4f})')
print(f'  Total PnL: {pnls.sum():.1f} pts (${pnls.sum()*2:.1f})')
print(f'  PF: {pnls[wins].sum() / abs(pnls[losses].sum()):.3f}')
print()

# R2
signal = np.array([1 if green[i] else -1 for i in range(n-1) if green[i] or red[i]])
outcome = np.array([closes[i+1] - closes[i] for i in range(n-1) if green[i] or red[i]])
r, p_r = pointbiserialr(signal, outcome)
print(f'R2 ANALYSIS')
print(f'  Point-biserial r: {r:.4f}')
print(f'  R2: {r**2:.6f} ({r**2*100:.3f}%)')
print(f'  p-value: {p_r:.2e}')
print(f'  Bar color explains {r**2*100:.3f}% of next-bar price change')
