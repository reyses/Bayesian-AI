# JULES SPEC: Trade Analytics — Statistical Exploration Suite

## Goal
After every forward pass, generate a deep statistical report comparing good trades vs bad
trades across every dimension available in oracle_trade_log.csv. The user wants to stop
flying blind — this suite replaces intuition with sigma-level evidence.

## Output
Appended to `checkpoints/phase4_report.txt` AND saved as `checkpoints/trade_analytics.txt`
(standalone, so it can be read without the full report).

---

## Part 1: Feature Engineering from oracle_trade_log.csv

Load the log and derive these columns:

```python
df['entry_dt']      = pd.to_datetime(df['entry_time'], unit='s', utc=True).dt.tz_convert('US/Eastern')
df['hour_et']       = df['entry_dt'].dt.hour
df['dow']           = df['entry_dt'].dt.dayofweek          # 0=Mon
df['month']         = df['entry_dt'].dt.month
df['hold_mins']     = df['hold_bars'] * 0.25               # 15s bars → minutes
df['is_win']        = (df['actual_pnl'] > 0).astype(int)
df['capture_pct']   = df['capture_rate'].clip(-1, 2) * 100
df['signed_oracle'] = df['oracle_mfe'] * df['oracle_label'].apply(lambda x: 1 if x > 0 else -1)

# Session label
def session(h):
    if h in range(1, 9):   return 'Europe/PreMkt'
    if h in (9, 10):       return 'US_Open'
    if h in (11,12,13):    return 'US_Mid'
    if h in (14,15,16):    return 'US_Close'
    return 'Overnight'
df['session'] = df['hour_et'].apply(session)

# Direction correctness
df['dir_correct'] = (
    ((df['direction']=='LONG')  & (df['oracle_label'] > 0)) |
    ((df['direction']=='SHORT') & (df['oracle_label'] < 0))
).astype(int)

# Parse entry_workers JSON for 1h/5m dir_prob and conviction
import json
def _parse_workers(s):
    try: return json.loads(s)
    except: return {}
df['ew'] = df['entry_workers'].apply(_parse_workers)
df['w1h_d']   = df['ew'].apply(lambda w: w.get('1h',{}).get('d', 0.5))
df['w1h_c']   = df['ew'].apply(lambda w: w.get('1h',{}).get('c', 0.0))
df['w5m_d']   = df['ew'].apply(lambda w: w.get('5m',{}).get('d', 0.5))
df['w5m_c']   = df['ew'].apply(lambda w: w.get('5m',{}).get('c', 0.0))
df['w1h_agree'] = (
    ((df['direction']=='LONG')  & (df['w1h_d'] > 0.5)) |
    ((df['direction']=='SHORT') & (df['w1h_d'] < 0.5))
).astype(int)
df['w5m_agree'] = (
    ((df['direction']=='LONG')  & (df['w5m_d'] > 0.5)) |
    ((df['direction']=='SHORT') & (df['w5m_d'] < 0.5))
).astype(int)
```

---

## Part 2: Good Trade vs Bad Trade Comparison

Split: `winners = df[df['is_win']==1]`, `losers = df[df['is_win']==0]`

For each numeric feature, compute mean ± std and run a two-sample t-test:

**Features to compare:**
- `belief_conviction`, `wave_maturity`, `decision_wave_maturity`
- `oracle_mfe`, `oracle_mae`, `capture_pct`, `hold_mins`
- `dmi_diff`, `long_bias`, `short_bias`
- `w1h_d`, `w1h_c`, `w5m_d`, `w5m_c`
- `w1h_agree`, `w5m_agree`
- `entry_depth`

**Output format:**
```
GOOD vs BAD TRADE COMPARISON  (t-test, two-tailed)
  Feature               Winners(n=X)       Losers(n=Y)     t-stat    p-value  sig
  ------------------    ---------------    ---------------  ------    -------  ---
  belief_conviction     0.612 ± 0.091      0.548 ± 0.102    5.21    <0.001   ***
  wave_maturity         0.089 ± 0.041      0.104 ± 0.052   -2.31     0.021   *
  ...
```
Stars: *** p<0.001, ** p<0.01, * p<0.05, (blank) p≥0.05

---

## Part 3: ANOVA — Categorical Variables

For each categorical variable, run one-way ANOVA of `actual_pnl` across groups.
Also show group means as a mini table.

**Categoricals:**
1. `session` (5 groups: Overnight, Europe/PreMkt, US_Open, US_Mid, US_Close)
2. `dow` (Mon-Fri)
3. `entry_depth` (0-12)
4. `direction` (LONG/SHORT)
5. `exit_reason` (trail_stop, take_profit, belief_flip, eod)
6. `dir_correct` (0/1)

**Output:**
```
ANOVA: actual_pnl ~ session
  F=14.23  p<0.001 ***  (significant session effect)
  Session          n    Mean PnL   Std PnL
  Overnight      152    -$12.40   $84.20
  Europe/PreMkt  398    +$18.32   $61.50  <- best
  US_Open        310    +$8.14    $72.10
  ...
```

---

## Part 4: Linear Regression — What Predicts PnL?

OLS regression: `actual_pnl ~ features`

Features (standardized): `belief_conviction`, `wave_maturity`, `oracle_mfe`,
`oracle_mae`, `dmi_diff`, `w1h_c`, `w5m_c`, `w1h_agree`, `w5m_agree`,
`entry_depth`, `hold_mins`, `hour_et`

**Output:**
```
LINEAR REGRESSION: actual_pnl ~ features  (OLS, standardized coefficients)
  R² = 0.183  Adj R² = 0.176  (n=1544)
  Feature               Coef    Std Err   t-stat   p-value  sig
  belief_conviction    +8.42     1.21      6.96    <0.001   ***
  oracle_mfe           +5.31     0.88      6.03    <0.001   ***
  w1h_agree            +3.14     0.94      3.34     0.001   **
  wave_maturity        -2.87     1.02     -2.81     0.005   **
  ...
```

---

## Part 5: Logistic Regression — What Predicts Win/Loss?

Logistic regression: `is_win ~ features` (same features as Part 4)

**Output:**
```
LOGISTIC REGRESSION: is_win ~ features
  Accuracy = 76.3%  AUC-ROC = 0.71
  Feature               Coef    Odds Ratio   p-value  sig
  belief_conviction    +1.24      3.46       <0.001   ***
  w1h_agree            +0.87      2.39        0.002   **
  wave_maturity        -0.64      0.53        0.008   **
  ...
```

---

## Part 6: Capture Rate Deep Dive

For correct-direction trades only, regress `capture_pct` on exit features:

```python
exit_df = df[df['dir_correct']==1].copy()
# parse exit_workers for 5m flip
exit_df['ew_exit'] = exit_df['exit_workers'].apply(_parse_workers)
exit_df['w5m_exit_d'] = exit_df['ew_exit'].apply(lambda w: w.get('5m',{}).get('d', 0.5))
exit_df['w5m_flipped'] = (
    ((exit_df['direction']=='LONG')  & (exit_df['w5m_exit_d'] < 0.5)) |
    ((exit_df['direction']=='SHORT') & (exit_df['w5m_exit_d'] > 0.5))
).astype(int)
```

OLS: `capture_pct ~ hold_mins + exit_conviction + exit_wave_maturity + w5m_flipped + oracle_mfe`

**Output:**
```
CAPTURE RATE REGRESSION (correct-direction trades, n=830)
  R² = 0.241
  hold_mins        -0.31  p<0.001 ***  <- longer hold = less captured
  w5m_flipped      -0.18  p=0.003 **   <- 5m flip during trade hurts capture
  oracle_mfe       -0.14  p=0.011 *    <- bigger move = harder to capture fully
  exit_conviction  +0.09  p=0.041 *
```

---

## Part 7: Session × Direction Interaction Table

Cross-tab: session rows × direction columns, showing mean PnL per cell.
Highlights if LONG works in morning but SHORT works at close, etc.

```
SESSION × DIRECTION  (mean PnL per trade)
  Session           LONG    SHORT   Delta
  Overnight        -$18     -$6    -$12
  Europe/PreMkt    +$22    +$14    +$8   <- LONG edge in Europe
  US_Open          +$12    +$4     +$8
  US_Mid           +$9     +$11    -$2
  US_Close         +$3     -$14   +$17   <- LONG only at close
```

---

## Implementation

### New file: `training/trade_analytics.py`

```python
def run_trade_analytics(log_path: str, report_path: str) -> str:
    """
    Loads oracle_trade_log.csv, runs the full statistical suite,
    appends to report_path, and returns the analytics text.
    """
    ...
```

### Call site: `training/orchestrator.py`

At the very end of `run_forward_pass`, after writing the main report:

```python
from training.trade_analytics import run_trade_analytics
analytics_txt = run_trade_analytics(
    log_path=os.path.join(self.checkpoint_dir, 'oracle_trade_log.csv'),
    report_path=_report_path,
)
# also save standalone
with open(os.path.join(self.checkpoint_dir, 'trade_analytics.txt'), 'w') as f:
    f.write(analytics_txt)
```

### Dependencies (all already in requirements):
- `pandas`, `numpy`, `scipy.stats` (ttest_ind, f_oneway), `sklearn.linear_model`
  (LinearRegression, LogisticRegression), `sklearn.preprocessing` (StandardScaler),
  `sklearn.metrics` (roc_auc_score)

---

## Baseline (last run for reference)
- 1,544 trades, 78.9% WR, +$15,298
- Worst hours: 20:00 ET (WR=54%, -$1,515), 23:00 ET (WR=82%, -$1,494)
- Best session: Europe/PreMkt (WR=91% at 05:00)
- Wrong direction: 36.6% — key feature to predict
