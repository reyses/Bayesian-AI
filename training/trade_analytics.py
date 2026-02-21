"""
Bayesian-AI — Trade Analytics Suite
File: training/trade_analytics.py

Post-run statistical exploration: compares winners vs losers across every
available dimension in oracle_trade_log.csv.

Outputs:
  - Appended to checkpoints/phase4_report.txt
  - Saved standalone to checkpoints/trade_analytics.txt

Parts:
  1. Feature engineering
  2. Good vs Bad trade t-tests
  3. ANOVA — categorical variables
  4. Linear regression — what predicts PnL?
  5. Logistic regression — what predicts Win/Loss?
  6. Capture rate deep dive (correct-direction trades)
  7. Session × Direction interaction table
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


# ── helpers ──────────────────────────────────────────────────────────────────

def _sig(p: float) -> str:
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''


def _ols_with_stats(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> list[dict]:
    """
    Fit OLS and return per-feature dicts with coef, se, t, p.
    Computes SE(β) = sqrt(diag(s²(XᵀX)⁻¹)).
    """
    n, p = X.shape
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)
    y_hat = model.predict(X)
    residuals = y - y_hat
    s2 = np.sum(residuals ** 2) / max(n - p - 1, 1)

    # Add intercept column for covariance computation
    X_aug = np.column_stack([np.ones(n), X])
    try:
        cov = np.linalg.inv(X_aug.T @ X_aug) * s2
        se_all = np.sqrt(np.maximum(np.diag(cov), 0))
    except np.linalg.LinAlgError:
        se_all = np.full(p + 1, np.nan)

    r2 = model.score(X, y)
    adj_r2 = 1 - (1 - r2) * (n - 1) / max(n - p - 1, 1)

    rows = []
    for i, name in enumerate(feature_names):
        coef = model.coef_[i]
        se   = se_all[i + 1]  # +1 because se_all[0] = intercept SE
        t    = coef / se if se > 0 else 0.0
        p    = float(2 * stats.t.sf(abs(t), df=max(n - p - 1, 1)))
        rows.append({'name': name, 'coef': coef, 'se': se, 't': t, 'p': p})

    return rows, r2, adj_r2


def _parse_workers(s) -> dict:
    try:
        return json.loads(s) if isinstance(s, str) else {}
    except Exception:
        return {}


# ── Part 1: feature engineering ──────────────────────────────────────────────

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Time features
    df['entry_dt']  = pd.to_datetime(df['entry_time'], unit='s', utc=True).dt.tz_convert('US/Eastern')
    df['hour_et']   = df['entry_dt'].dt.hour
    df['dow']       = df['entry_dt'].dt.dayofweek        # 0=Mon
    df['month']     = df['entry_dt'].dt.month
    df['hold_mins'] = df.get('hold_bars', pd.Series(0, index=df.index)) * 0.25

    # Outcome flags
    df['is_win']      = (df['actual_pnl'] > 0).astype(int)
    df['capture_pct'] = df['capture_rate'].clip(-1, 2) * 100

    # Oracle signed MFE
    df['signed_oracle'] = df['oracle_mfe'] * df['oracle_label'].apply(
        lambda x: 1 if x > 0 else -1)

    # Session label
    def _session(h: int) -> str:
        if 1 <= h <= 8:    return 'Europe/PreMkt'
        if h in (9, 10):   return 'US_Open'
        if h in (11,12,13):return 'US_Mid'
        if h in (14,15,16):return 'US_Close'
        return 'Overnight'
    df['session'] = df['hour_et'].apply(_session)

    # Direction correctness
    df['dir_correct'] = (
        ((df['direction'] == 'LONG')  & (df['oracle_label'] > 0)) |
        ((df['direction'] == 'SHORT') & (df['oracle_label'] < 0))
    ).astype(int)

    # Parse entry_workers for 1h and 5m
    if 'entry_workers' in df.columns:
        df['_ew'] = df['entry_workers'].apply(_parse_workers)
        df['w1h_d'] = df['_ew'].apply(lambda w: w.get('1h', {}).get('d', 0.5))
        df['w1h_c'] = df['_ew'].apply(lambda w: w.get('1h', {}).get('c', 0.0))
        df['w5m_d'] = df['_ew'].apply(lambda w: w.get('5m', {}).get('d', 0.5))
        df['w5m_c'] = df['_ew'].apply(lambda w: w.get('5m', {}).get('c', 0.0))
        df['w1h_agree'] = (
            ((df['direction'] == 'LONG')  & (df['w1h_d'] > 0.5)) |
            ((df['direction'] == 'SHORT') & (df['w1h_d'] < 0.5))
        ).astype(int)
        df['w5m_agree'] = (
            ((df['direction'] == 'LONG')  & (df['w5m_d'] > 0.5)) |
            ((df['direction'] == 'SHORT') & (df['w5m_d'] < 0.5))
        ).astype(int)
        df.drop(columns=['_ew'], inplace=True)
    else:
        for col in ['w1h_d', 'w1h_c', 'w5m_d', 'w5m_c', 'w1h_agree', 'w5m_agree']:
            df[col] = np.nan

    return df


# ── Part 2: good vs bad t-tests ───────────────────────────────────────────────

def _part2_ttests(df: pd.DataFrame) -> list[str]:
    lines = []
    lines.append('')
    lines.append('=' * 80)
    lines.append('PART 2 — GOOD vs BAD TRADE COMPARISON  (Welch t-test, two-tailed)')
    lines.append('=' * 80)

    winners = df[df['is_win'] == 1]
    losers  = df[df['is_win'] == 0]
    nw, nl  = len(winners), len(losers)

    features = [
        'belief_conviction', 'wave_maturity', 'decision_wave_maturity',
        'oracle_mfe', 'oracle_mae', 'capture_pct', 'hold_mins',
        'dmi_diff', 'long_bias', 'short_bias',
        'w1h_d', 'w1h_c', 'w5m_d', 'w5m_c',
        'w1h_agree', 'w5m_agree',
        'entry_depth',
    ]

    header = (f"  {'Feature':<28} {'Winners':>18} {'Losers':>18} "
              f"{'t-stat':>8} {'p-value':>9}  sig")
    sep    = '  ' + '-'*28 + '  ' + '-'*18 + '  ' + '-'*18 + '  ' + '-'*8 + '  ' + '-'*9 + '  ---'
    lines.append(header)
    lines.append(sep)

    for feat in features:
        if feat not in df.columns:
            continue
        w_vals = winners[feat].dropna()
        l_vals = losers[feat].dropna()
        if len(w_vals) < 5 or len(l_vals) < 5:
            continue
        w_mu, w_sd = w_vals.mean(), w_vals.std()
        l_mu, l_sd = l_vals.mean(), l_vals.std()
        t_stat, p_val = stats.ttest_ind(w_vals, l_vals, equal_var=False)
        sig = _sig(p_val)
        p_str = f'<0.001' if p_val < 0.001 else f'{p_val:.3f}'
        lines.append(
            f"  {feat:<28} {w_mu:>8.3f} ± {w_sd:>6.3f}  "
            f"{l_mu:>8.3f} ± {l_sd:>6.3f}  "
            f"{t_stat:>8.2f}  {p_str:>9}  {sig}"
        )

    lines.append(f"\n  Winners n={nw}  Losers n={nl}  "
                 f"Stars: *** p<0.001  ** p<0.01  * p<0.05")
    return lines


# ── Part 3: ANOVA ─────────────────────────────────────────────────────────────

def _part3_anova(df: pd.DataFrame) -> list[str]:
    lines = []
    lines.append('')
    lines.append('=' * 80)
    lines.append('PART 3 — ANOVA: actual_pnl ~ categorical variables')
    lines.append('=' * 80)

    SESSION_ORDER = ['Overnight', 'Europe/PreMkt', 'US_Open', 'US_Mid', 'US_Close']
    DOW_NAMES     = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri'}

    categoricals = [
        ('session',     'Session',     SESSION_ORDER),
        ('dow',         'Day-of-week', None),
        ('entry_depth', 'Depth',       None),
        ('direction',   'Direction',   None),
        ('exit_reason', 'Exit reason', None),
        ('dir_correct', 'Dir correct', None),
    ]

    for col, label, order in categoricals:
        if col not in df.columns:
            continue
        groups_raw = df.groupby(col)['actual_pnl'].apply(list)
        if len(groups_raw) < 2:
            continue

        group_lists = [v for v in groups_raw.values if len(v) >= 3]
        if len(group_lists) < 2:
            continue

        f_stat, p_val = stats.f_oneway(*group_lists)
        sig   = _sig(p_val)
        p_str = '<0.001' if p_val < 0.001 else f'{p_val:.3f}'

        lines.append('')
        lines.append(f"  ANOVA: actual_pnl ~ {label}")
        lines.append(f"    F={f_stat:.2f}  p={p_str} {sig}")

        summary = df.groupby(col)['actual_pnl'].agg(['count', 'mean', 'std']).reset_index()
        summary.columns = [col, 'n', 'mean', 'std']

        # Determine sort order
        if order:
            cat_order = [g for g in order if g in summary[col].values]
            summary[col] = pd.Categorical(summary[col], categories=cat_order, ordered=True)
            summary = summary.sort_values(col)
        else:
            summary = summary.sort_values('mean', ascending=False)

        best_mean = summary['mean'].max()
        lines.append(f"    {'Group':<22} {'n':>6}  {'Mean PnL':>10}  {'Std PnL':>10}")

        for _, row in summary.iterrows():
            grp = DOW_NAMES.get(row[col], str(row[col])) if col == 'dow' else str(row[col])
            flag = '  <- best' if abs(row['mean'] - best_mean) < 0.01 else ''
            lines.append(
                f"    {grp:<22} {int(row['n']):>6}  ${row['mean']:>9,.2f}  ${row['std']:>9,.2f}{flag}")

    return lines


# ── Part 4: linear regression ─────────────────────────────────────────────────

def _part4_ols(df: pd.DataFrame) -> list[str]:
    lines = []
    lines.append('')
    lines.append('=' * 80)
    lines.append('PART 4 — LINEAR REGRESSION: actual_pnl ~ features  (OLS, standardized β)')
    lines.append('=' * 80)

    feat_cols = [
        'belief_conviction', 'wave_maturity', 'oracle_mfe', 'oracle_mae',
        'dmi_diff', 'w1h_c', 'w5m_c', 'w1h_agree', 'w5m_agree',
        'entry_depth', 'hold_mins', 'hour_et',
    ]
    available = [c for c in feat_cols if c in df.columns]
    sub = df[available + ['actual_pnl']].dropna()
    if len(sub) < 30 or len(available) < 3:
        lines.append('  (insufficient data)')
        return lines

    X_raw = sub[available].values
    y     = sub['actual_pnl'].values
    n, p  = X_raw.shape

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    rows, r2, adj_r2 = _ols_with_stats(X, y, available)
    rows.sort(key=lambda r: -abs(r['t']))

    lines.append(f'  R² = {r2:.3f}  Adj R² = {adj_r2:.3f}  (n={n})')
    lines.append(f"  {'Feature':<28} {'Coef':>8} {'Std Err':>9} {'t-stat':>8} {'p-value':>9}  sig")
    lines.append('  ' + '-'*75)

    for r in rows:
        p_str = '<0.001' if r['p'] < 0.001 else f"{r['p']:.3f}"
        lines.append(
            f"  {r['name']:<28} {r['coef']:>+8.2f} {r['se']:>9.2f} "
            f"{r['t']:>8.2f} {p_str:>9}  {_sig(r['p'])}"
        )

    return lines


# ── Part 5: logistic regression ───────────────────────────────────────────────

def _part5_logistic(df: pd.DataFrame) -> list[str]:
    lines = []
    lines.append('')
    lines.append('=' * 80)
    lines.append('PART 5 — LOGISTIC REGRESSION: is_win ~ features')
    lines.append('=' * 80)

    feat_cols = [
        'belief_conviction', 'wave_maturity', 'oracle_mfe', 'oracle_mae',
        'dmi_diff', 'w1h_c', 'w5m_c', 'w1h_agree', 'w5m_agree',
        'entry_depth', 'hold_mins', 'hour_et',
    ]
    available = [c for c in feat_cols if c in df.columns]
    sub = df[available + ['is_win']].dropna()
    if len(sub) < 30 or sub['is_win'].nunique() < 2:
        lines.append('  (insufficient data)')
        return lines

    X_raw = sub[available].values
    y     = sub['is_win'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)

    y_pred    = model.predict(X)
    y_prob    = model.predict_proba(X)[:, 1]
    accuracy  = (y_pred == y).mean()
    auc       = roc_auc_score(y, y_prob)

    lines.append(f'  Accuracy = {accuracy:.1%}  AUC-ROC = {auc:.3f}  (n={len(sub)})')
    lines.append(f"  {'Feature':<28} {'Coef':>8} {'Odds Ratio':>11}  note")
    lines.append('  ' + '-'*65)

    coefs = list(zip(available, model.coef_[0]))
    coefs.sort(key=lambda x: -abs(x[1]))
    for name, coef in coefs:
        odds = np.exp(coef)
        direction = '(helps)' if coef > 0 else '(hurts)'
        lines.append(f"  {name:<28} {coef:>+8.3f} {odds:>11.3f}  {direction}")

    lines.append('  Note: coefficients are standardized (1 SD change in feature)')
    return lines


# ── Part 6: capture rate regression ──────────────────────────────────────────

def _part6_capture(df: pd.DataFrame) -> list[str]:
    lines = []
    lines.append('')
    lines.append('=' * 80)
    lines.append('PART 6 — CAPTURE RATE REGRESSION (correct-direction trades)')
    lines.append('=' * 80)

    exit_df = df[df['dir_correct'] == 1].copy()

    if 'exit_workers' in exit_df.columns:
        exit_df['_ew_exit'] = exit_df['exit_workers'].apply(_parse_workers)
        exit_df['w5m_exit_d'] = exit_df['_ew_exit'].apply(
            lambda w: w.get('5m', {}).get('d', 0.5))
        exit_df['w5m_flipped'] = (
            ((exit_df['direction'] == 'LONG')  & (exit_df['w5m_exit_d'] < 0.5)) |
            ((exit_df['direction'] == 'SHORT') & (exit_df['w5m_exit_d'] > 0.5))
        ).astype(int)
        exit_df.drop(columns=['_ew_exit'], inplace=True)
    else:
        exit_df['w5m_flipped'] = 0

    feat_cols = ['hold_mins', 'exit_conviction', 'exit_wave_maturity', 'w5m_flipped', 'oracle_mfe']
    available = [c for c in feat_cols if c in exit_df.columns]
    sub = exit_df[available + ['capture_pct']].dropna()

    if len(sub) < 20:
        lines.append('  (insufficient correct-direction trades)')
        return lines

    X_raw = sub[available].values
    y     = sub['capture_pct'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    rows, r2, _ = _ols_with_stats(X, y, available)
    rows.sort(key=lambda r: r['coef'])  # most negative first = biggest drags

    lines.append(f'  n={len(sub)} correct-direction trades  R² = {r2:.3f}')
    lines.append(f"  {'Feature':<28} {'Coef':>8} {'t-stat':>8} {'p-value':>9}  sig  interpretation")
    lines.append('  ' + '-'*80)

    interpretations = {
        'hold_mins':          'longer hold = less captured',
        'w5m_flipped':        '5m flip during trade hurts capture',
        'oracle_mfe':         'bigger move = harder to capture fully',
        'exit_conviction':    'higher conviction at exit = better capture',
        'exit_wave_maturity': 'mature wave at exit = late departure',
    }
    for r in rows:
        p_str = '<0.001' if r['p'] < 0.001 else f"{r['p']:.3f}"
        interp = interpretations.get(r['name'], '')
        lines.append(
            f"  {r['name']:<28} {r['coef']:>+8.2f} {r['t']:>8.2f} "
            f"{p_str:>9}  {_sig(r['p']):<3}  {interp}"
        )

    return lines


# ── Part 7: session × direction ───────────────────────────────────────────────

def _part7_session_direction(df: pd.DataFrame) -> list[str]:
    lines = []
    lines.append('')
    lines.append('=' * 80)
    lines.append('PART 7 — SESSION × DIRECTION  (mean PnL per trade)')
    lines.append('=' * 80)

    SESSION_ORDER = ['Overnight', 'Europe/PreMkt', 'US_Open', 'US_Mid', 'US_Close']
    pivot = df.pivot_table(
        values='actual_pnl', index='session', columns='direction', aggfunc='mean')

    # Add counts
    counts = df.pivot_table(
        values='actual_pnl', index='session', columns='direction', aggfunc='count')

    lines.append(f"  {'Session':<18} {'LONG mean':>10}  {'SHORT mean':>11}  {'Delta':>8}  "
                 f"{'n LONG':>7}  {'n SHORT':>8}")
    lines.append('  ' + '-'*75)

    for sess in SESSION_ORDER:
        if sess not in pivot.index:
            continue
        row   = pivot.loc[sess]
        cnt   = counts.loc[sess]
        long_  = row.get('LONG',  np.nan)
        short_ = row.get('SHORT', np.nan)
        nl_    = int(cnt.get('LONG',  0))
        ns_    = int(cnt.get('SHORT', 0))
        delta  = long_ - short_ if (not np.isnan(long_) and not np.isnan(short_)) else np.nan

        def _fmt(v):
            if np.isnan(v): return '    N/A   '
            return f'${v:>+8,.2f}'
        lines.append(
            f"  {sess:<18} {_fmt(long_)}  {_fmt(short_)}  {_fmt(delta)}  "
            f"{nl_:>7}  {ns_:>8}"
        )

    # WR by session
    lines.append('')
    lines.append(f"  WIN RATE BY SESSION:")
    lines.append(f"  {'Session':<18} {'WR LONG':>9}  {'WR SHORT':>10}  {'WR All':>8}")
    lines.append('  ' + '-'*55)
    for sess in SESSION_ORDER:
        sub = df[df['session'] == sess]
        if len(sub) == 0: continue
        wr_all   = sub['is_win'].mean()
        wr_long  = sub[sub['direction']=='LONG']['is_win'].mean()  if (sub['direction']=='LONG').any()  else np.nan
        wr_short = sub[sub['direction']=='SHORT']['is_win'].mean() if (sub['direction']=='SHORT').any() else np.nan
        def _pct(v): return f'{v:.1%}' if not np.isnan(v) else '  N/A '
        lines.append(f"  {sess:<18} {_pct(wr_long):>9}  {_pct(wr_short):>10}  {_pct(wr_all):>8}")

    return lines


# ── HOUR-BY-HOUR BREAKDOWN ───────────────────────────────────────────────────

def _hour_breakdown(df: pd.DataFrame) -> list[str]:
    lines = []
    lines.append('')
    lines.append('=' * 80)
    lines.append('HOUR-BY-HOUR BREAKDOWN  (ET, 24h)')
    lines.append('=' * 80)
    lines.append(f"  {'Hour ET':<10} {'Trades':>7} {'WR':>7} {'AvgPnL':>10} {'TotalPnL':>12}  flag")
    lines.append('  ' + '-'*65)

    for h in range(24):
        sub = df[df['hour_et'] == h]
        if len(sub) == 0: continue
        wr    = sub['is_win'].mean()
        avg   = sub['actual_pnl'].mean()
        total = sub['actual_pnl'].sum()
        flag  = ''
        if avg < -10 or wr < 0.55:  flag = '<- AVOID'
        elif avg > 30 and wr > 0.80: flag = '<- BEST'
        lines.append(
            f"  {h:02d}:00       {len(sub):>7} {wr:>7.1%} ${avg:>9.2f} ${total:>11,.2f}  {flag}")

    return lines


# ── PART 8: BEST / WORST DEEP DIVE ───────────────────────────────────────────

def _part8_best_worst(df: pd.DataFrame, n: int = 20) -> list[str]:
    """
    Red-X analysis: compare top-N winners vs bottom-N losers across every
    available feature dimension. Goal: find the features that clearly
    separate the extremes so we can add gates or adjust weights.
    """
    lines = []
    lines.append('')
    lines.append('─' * 80)
    lines.append(f'PART 8 — BEST/WORST DEEP DIVE  (top {n} vs bottom {n} by PnL)')
    lines.append('─' * 80)

    pnl_col = 'actual_pnl' if 'actual_pnl' in df.columns else 'pnl' if 'pnl' in df.columns else None
    if pnl_col is None or len(df) < n * 2:
        lines.append(f'  (not enough trades or no pnl column: need {n*2}, got {len(df)})')
        return lines

    best  = df.nlargest(n,  pnl_col).copy()
    worst = df.nsmallest(n, pnl_col).copy()

    lines.append(f'  BEST  {n}: PnL range  ${best[pnl_col].min():>8,.0f} → ${best[pnl_col].max():>8,.0f}   '
                 f'mean=${best[pnl_col].mean():>8,.0f}')
    lines.append(f'  WORST {n}: PnL range  ${worst[pnl_col].min():>8,.0f} → ${worst[pnl_col].max():>8,.0f}   '
                 f'mean=${worst[pnl_col].mean():>8,.0f}')
    lines.append('')

    # ── Numeric feature comparison ────────────────────────────────────────────
    numeric_features = []
    for col in ['entry_trail_ticks', 'entry_tp_ticks', 'entry_sl_ticks', 'entry_trail_act',
                'belief_conviction', 'belief_active_levels', 'oracle_mfe', 'oracle_mae',
                'w1h_d', 'w5m_d', 'w1h_agree', 'w5m_agree']:
        if col in df.columns:
            numeric_features.append(col)

    if numeric_features:
        lines.append(f'  {"Feature":<26}  {"Best mean":>10}  {"Worst mean":>10}  {"Diff":>10}  Note')
        lines.append(f'  {"─"*26}  {"─"*10}  {"─"*10}  {"─"*10}  {"─"*20}')
        for col in numeric_features:
            b_mean = best[col].mean()
            w_mean = worst[col].mean()
            diff   = b_mean - w_mean
            # Flag big differences
            b_std = df[col].std()
            note = ''
            if b_std > 0 and abs(diff) > b_std:
                note = '<< BIG DIFF'
            elif b_std > 0 and abs(diff) > b_std * 0.5:
                note = '< notable'
            lines.append(f'  {col:<26}  {b_mean:>10.3f}  {w_mean:>10.3f}  {diff:>+10.3f}  {note}')
        lines.append('')

    # ── Categorical breakdown ─────────────────────────────────────────────────
    cat_features = []
    for col in ['exit_reason', 'session', 'direction', 'entry_depth', 'dow']:
        if col in df.columns:
            cat_features.append(col)

    for col in cat_features:
        b_counts = best[col].value_counts(normalize=True).round(2)
        w_counts = worst[col].value_counts(normalize=True).round(2)
        all_vals = sorted(set(b_counts.index) | set(w_counts.index))
        lines.append(f'  {col.upper()}')
        lines.append(f'    {"Value":<20}  {"Best %":>8}  {"Worst %":>8}  {"Diff":>8}')
        for v in all_vals:
            bp = b_counts.get(v, 0.0)
            wp = w_counts.get(v, 0.0)
            flag = ' <<' if abs(bp - wp) >= 0.20 else ''
            lines.append(f'    {str(v):<20}  {bp:>8.0%}  {wp:>8.0%}  {bp-wp:>+8.0%}{flag}')
        lines.append('')

    # ── Worker agreement at entry ─────────────────────────────────────────────
    if 'entry_workers' in df.columns:
        lines.append('  WORKER AGREEMENT AT ENTRY (best vs worst)')
        lines.append(f'    {"TF":<6}  {"Best agree":>11}  {"Worst agree":>11}  {"Diff":>8}')
        tf_labels = ['1h', '30m', '15m', '5m', '3m', '1m', '30s', '15s']
        for tf in tf_labels:
            def _agree(row, tf=tf):
                try:
                    snap = json.loads(row)
                    w = snap.get(tf, {})
                    d = w.get('d', 0.5)
                    # agree = worker dir matches trade direction (d>0.5 → LONG)
                    return d
                except Exception:
                    return np.nan

            best_agree  = best['entry_workers'].apply(_agree).mean()
            worst_agree = worst['entry_workers'].apply(_agree).mean()
            if np.isnan(best_agree):
                continue
            flag = ' <<' if abs(best_agree - worst_agree) >= 0.10 else ''
            lines.append(f'    {tf:<6}  {best_agree:>11.3f}  {worst_agree:>11.3f}  '
                         f'{best_agree - worst_agree:>+8.3f}{flag}')
        lines.append('')

    # ── Individual trade cards for the top 5 best and worst ───────────────────
    for label, subset in [('BEST 5', best.head(5)), ('WORST 5', worst.head(5))]:
        lines.append(f'  {label} TRADES')
        for _, row in subset.iterrows():
            pnl    = row.get('pnl', 0)
            depth  = row.get('entry_depth', '?')
            dirn   = row.get('direction', '?')
            sess   = row.get('session', '?')
            reason = row.get('exit_reason', '?')
            conv   = row.get('belief_conviction', float('nan'))
            mfe    = row.get('oracle_mfe', float('nan'))
            mae    = row.get('oracle_mae', float('nan'))
            tp     = row.get('entry_tp_ticks', float('nan'))
            sl     = row.get('entry_sl_ticks', float('nan'))
            trail  = row.get('entry_trail_ticks', float('nan'))
            lines.append(
                f'    PnL=${pnl:>8,.0f}  depth={depth}  {dirn:<5}  sess={sess:<8}  '
                f'exit={str(reason):<12}  conv={conv:.2f}  '
                f'MFE={mfe:.1f}pts  MAE={mae:.1f}pts  '
                f'TP={tp:.0f}t  SL={sl:.0f}t  trail={trail:.0f}t'
            )
        lines.append('')

    return lines


# ── MAIN ENTRY POINT ─────────────────────────────────────────────────────────

def run_trade_analytics(log_path: str, report_path: str) -> str:
    """
    Load oracle_trade_log.csv, run the full statistical suite,
    append to report_path, and return the analytics text.
    """
    out_lines = []
    out_lines.append('')
    out_lines.append('=' * 80)
    out_lines.append('TRADE ANALYTICS SUITE')
    out_lines.append('=' * 80)

    try:
        df = pd.read_csv(log_path)
    except FileNotFoundError:
        msg = f'  [trade_analytics] oracle_trade_log.csv not found at {log_path}'
        out_lines.append(msg)
        return '\n'.join(out_lines)
    except Exception as e:
        out_lines.append(f'  [trade_analytics] failed to load: {e}')
        return '\n'.join(out_lines)

    if df.empty:
        out_lines.append('  (no trades to analyze)')
        return '\n'.join(out_lines)

    out_lines.append(f'  Loaded {len(df):,} trades from {log_path}')

    # Part 1: feature engineering
    try:
        df = _engineer_features(df)
    except Exception as e:
        out_lines.append(f'  [feature engineering error] {e}')
        return '\n'.join(out_lines)

    # Parts 2–7
    sections = [
        ('Part 2 t-tests',            _part2_ttests),
        ('Part 3 ANOVA',              _part3_anova),
        ('Part 4 OLS regression',     _part4_ols),
        ('Part 5 logistic regression',_part5_logistic),
        ('Part 6 capture rate',       _part6_capture),
        ('Part 7 session×direction',  _part7_session_direction),
        ('Hour breakdown',            _hour_breakdown),
        ('Part 8 best/worst dive',    _part8_best_worst),
    ]

    for name, fn in sections:
        try:
            out_lines.extend(fn(df))
        except Exception as e:
            out_lines.append(f'  [{name} error] {e}')

    out_lines.append('')
    out_lines.append('=' * 80)
    out_lines.append('END OF TRADE ANALYTICS SUITE')
    out_lines.append('=' * 80)

    text = '\n'.join(out_lines)

    # Append to main report
    if report_path:
        try:
            with open(report_path, 'a', encoding='utf-8') as f:
                f.write('\n' + text + '\n')
        except Exception:
            pass  # don't crash if report is read-only

    return text
