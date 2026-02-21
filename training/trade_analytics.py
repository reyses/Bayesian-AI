"""
JULES SPEC: Trade Analytics — Statistical Exploration Suite

Generates a deep statistical report comparing good trades vs bad trades
across every dimension available in oracle_trade_log.csv.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# --- Constants to avoid magic numbers ---
BAR_SECONDS = 15
HOLD_MINS_FACTOR = 0.25  # 15s bars -> minutes (1/4)
CAPTURE_CLIP_MIN = -1
CAPTURE_CLIP_MAX = 2
CAPTURE_SCALE = 100

# Significance levels
P_VALUE_SIG_001 = 0.001
P_VALUE_SIG_01 = 0.01
P_VALUE_SIG_05 = 0.05

# Categorical mapping
SESSION_MAP = {
    'Europe/PreMkt': range(1, 9),
    'US_Open': range(9, 11),
    'US_Mid': range(11, 14),
    'US_Close': range(14, 17)
    # Overnight is everything else
}

NEUTRAL_THRESHOLD = 0.5

# Features for comparison
COMPARE_FEATURES = [
    'belief_conviction', 'wave_maturity', 'decision_wave_maturity',
    'oracle_mfe', 'oracle_mae', 'capture_pct', 'hold_mins',
    'dmi_diff', 'long_bias', 'short_bias',
    'w1h_d', 'w1h_c', 'w5m_d', 'w5m_c',
    'w1h_agree', 'w5m_agree',
    'entry_depth'
]

REGRESSION_FEATURES = [
    'belief_conviction', 'wave_maturity', 'oracle_mfe',
    'oracle_mae', 'dmi_diff', 'w1h_c', 'w5m_c', 'w1h_agree', 'w5m_agree',
    'entry_depth', 'hold_mins', 'hour_et'
]

CAPTURE_REGRESSION_FEATURES = [
    'hold_mins', 'exit_conviction', 'exit_wave_maturity', 'w5m_flipped', 'oracle_mfe'
]

def _parse_workers(s):
    try:
        if isinstance(s, str):
            return json.loads(s)
        return {}
    except:
        return {}

def _get_session(h):
    for session_name, hours in SESSION_MAP.items():
        if h in hours:
            return session_name
    return 'Overnight'

def run_trade_analytics(log_path: str) -> str:
    """
    Loads oracle_trade_log.csv, runs the full statistical suite,
    and returns the analytics text.
    """
    if not os.path.exists(log_path):
        return "Trade analytics skipped: oracle_trade_log.csv not found."

    try:
        df = pd.read_csv(log_path)
    except Exception as e:
        return f"Trade analytics failed to load log: {e}"

    if df.empty:
        return "Trade analytics skipped: empty log."

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("JULES TRADE ANALYTICS SUITE")
    report_lines.append("=" * 80)

    # --- Part 1: Feature Engineering ---
    # entry_time is unix timestamp
    if 'entry_time' in df.columns:
        df['entry_dt'] = pd.to_datetime(df['entry_time'], unit='s', utc=True).dt.tz_convert('US/Eastern')
        df['hour_et'] = df['entry_dt'].dt.hour
        df['dow'] = df['entry_dt'].dt.dayofweek
        df['month'] = df['entry_dt'].dt.month
        df['session'] = df['hour_et'].apply(_get_session)
    else:
        # Fallback if entry_time missing
        df['hour_et'] = 0
        df['dow'] = 0
        df['month'] = 1
        df['session'] = 'Unknown'

    if 'hold_bars' in df.columns:
        df['hold_mins'] = df['hold_bars'] * HOLD_MINS_FACTOR
    else:
        df['hold_mins'] = 0.0

    if 'actual_pnl' in df.columns:
        df['is_win'] = (df['actual_pnl'] > 0).astype(int)
    else:
        df['is_win'] = 0
        df['actual_pnl'] = 0.0

    if 'capture_rate' in df.columns:
        df['capture_pct'] = df['capture_rate'].clip(CAPTURE_CLIP_MIN, CAPTURE_CLIP_MAX) * CAPTURE_SCALE
    else:
        df['capture_pct'] = 0.0

    if 'oracle_mfe' in df.columns and 'oracle_label' in df.columns:
        df['signed_oracle'] = df['oracle_mfe'] * df['oracle_label'].apply(lambda x: 1 if x > 0 else -1)
    else:
        df['signed_oracle'] = 0.0

    # Direction correctness
    if 'direction' in df.columns and 'oracle_label' in df.columns:
        df['dir_correct'] = (
            ((df['direction'] == 'LONG') & (df['oracle_label'] > 0)) |
            ((df['direction'] == 'SHORT') & (df['oracle_label'] < 0))
        ).astype(int)
    else:
        df['dir_correct'] = 0

    # Parse workers
    if 'entry_workers' in df.columns:
        df['ew'] = df['entry_workers'].apply(_parse_workers)
        df['w1h_d'] = df['ew'].apply(lambda w: w.get('1h', {}).get('d', NEUTRAL_THRESHOLD))
        df['w1h_c'] = df['ew'].apply(lambda w: w.get('1h', {}).get('c', 0.0))
        df['w5m_d'] = df['ew'].apply(lambda w: w.get('5m', {}).get('d', NEUTRAL_THRESHOLD))
        df['w5m_c'] = df['ew'].apply(lambda w: w.get('5m', {}).get('c', 0.0))

        if 'direction' in df.columns:
            df['w1h_agree'] = (
                ((df['direction'] == 'LONG') & (df['w1h_d'] > NEUTRAL_THRESHOLD)) |
                ((df['direction'] == 'SHORT') & (df['w1h_d'] < NEUTRAL_THRESHOLD))
            ).astype(int)
            df['w5m_agree'] = (
                ((df['direction'] == 'LONG') & (df['w5m_d'] > NEUTRAL_THRESHOLD)) |
                ((df['direction'] == 'SHORT') & (df['w5m_d'] < NEUTRAL_THRESHOLD))
            ).astype(int)
        else:
            df['w1h_agree'] = 0
            df['w5m_agree'] = 0
    else:
        for col in ['w1h_d', 'w1h_c', 'w5m_d', 'w5m_c', 'w1h_agree', 'w5m_agree']:
            df[col] = 0.0

    # Fill missing features with 0
    for col in COMPARE_FEATURES + REGRESSION_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    df.fillna(0, inplace=True)

    # --- Part 2: Good vs Bad Trade Comparison ---
    winners = df[df['is_win'] == 1]
    losers = df[df['is_win'] == 0]

    report_lines.append("")
    report_lines.append("GOOD vs BAD TRADE COMPARISON  (t-test, two-tailed)")
    report_lines.append(f"  {'Feature':<20} {'Winners(n=' + str(len(winners)) + ')':<18} {'Losers(n=' + str(len(losers)) + ')':<18} {'t-stat':>8} {'p-value':>10} {'sig':>5}")
    report_lines.append(f"  {'-'*20} {'-'*18} {'-'*18} {'-'*8} {'-'*10} {'-'*5}")

    for feat in COMPARE_FEATURES:
        if feat not in df.columns: continue
        w_vals = winners[feat]
        l_vals = losers[feat]
        if len(w_vals) < 2 or len(l_vals) < 2:
            continue

        w_mean = w_vals.mean()
        w_std = w_vals.std()
        l_mean = l_vals.mean()
        l_std = l_vals.std()

        t_stat, p_val = stats.ttest_ind(w_vals, l_vals, equal_var=False)

        sig = ""
        if p_val < P_VALUE_SIG_001: sig = "***"
        elif p_val < P_VALUE_SIG_01: sig = "**"
        elif p_val < P_VALUE_SIG_05: sig = "*"

        report_lines.append(f"  {feat:<20} {w_mean:>6.3f} ± {w_std:<8.3f} {l_mean:>6.3f} ± {l_std:<8.3f} {t_stat:>8.2f} {p_val:>10.3f} {sig:>5}")

    # --- Part 3: ANOVA ---
    report_lines.append("")
    report_lines.append("ANOVA: Categorical Variables (Target: actual_pnl)")

    categoricals = ['session', 'dow', 'entry_depth', 'direction', 'exit_reason', 'dir_correct']
    for cat in categoricals:
        if cat not in df.columns: continue
        groups = [group['actual_pnl'].values for name, group in df.groupby(cat) if len(group) > 1]
        if len(groups) < 2: continue

        f_stat, p_val = stats.f_oneway(*groups)
        sig = ""
        if p_val < P_VALUE_SIG_001: sig = "***"
        elif p_val < P_VALUE_SIG_01: sig = "**"
        elif p_val < P_VALUE_SIG_05: sig = "*"

        report_lines.append(f"  {cat}: F={f_stat:.2f} p={p_val:.3f} {sig}")

        # Mini table for means
        report_lines.append(f"    {'Group':<15} {'n':>5} {'Mean PnL':>10} {'Std PnL':>10}")
        stats_df = df.groupby(cat)['actual_pnl'].agg(['count', 'mean', 'std']).reset_index()
        for _, row in stats_df.iterrows():
            grp_name = str(row[cat])
            report_lines.append(f"    {grp_name:<15} {int(row['count']):>5} ${row['mean']:>9.2f} ${row['std']:>9.2f}")

    # --- Part 4: Linear Regression (PnL) ---
    report_lines.append("")
    report_lines.append("LINEAR REGRESSION: actual_pnl ~ features (Standardized)")

    if len(df) > 10:
        X = df[REGRESSION_FEATURES].copy()
        y = df['actual_pnl']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LinearRegression()
        model.fit(X_scaled, y)
        r2 = model.score(X_scaled, y)
        n = len(df)
        # Adj R2
        k = X.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

        report_lines.append(f"  R² = {r2:.3f}  Adj R² = {adj_r2:.3f}  (n={n})")
        report_lines.append(f"  {'Feature':<20} {'Coef':>10}")
        report_lines.append(f"  {'-'*20} {'-'*10}")

        coefs = list(zip(REGRESSION_FEATURES, model.coef_))
        coefs.sort(key=lambda x: abs(x[1]), reverse=True)

        for feat, coef in coefs:
            report_lines.append(f"  {feat:<20} {coef:>10.2f}")
    else:
        report_lines.append("  (Insufficient data for regression)")

    # --- Part 5: Logistic Regression (Win/Loss) ---
    report_lines.append("")
    report_lines.append("LOGISTIC REGRESSION: is_win ~ features")

    if len(df) > 10 and df['is_win'].nunique() > 1:
        X = df[REGRESSION_FEATURES].copy()
        y = df['is_win']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_scaled, y)
        acc = model.score(X_scaled, y)
        try:
            auc = roc_auc_score(y, model.predict_proba(X_scaled)[:, 1])
        except:
            auc = 0.5

        report_lines.append(f"  Accuracy = {acc:.1%}  AUC-ROC = {auc:.2f}")
        report_lines.append(f"  {'Feature':<20} {'Coef':>10} {'Odds Ratio':>12}")
        report_lines.append(f"  {'-'*20} {'-'*10} {'-'*12}")

        coefs = list(zip(REGRESSION_FEATURES, model.coef_[0]))
        coefs.sort(key=lambda x: abs(x[1]), reverse=True)

        for feat, coef in coefs:
            odds = np.exp(coef)
            report_lines.append(f"  {feat:<20} {coef:>10.2f} {odds:>12.2f}")
    else:
        report_lines.append("  (Insufficient data for logistic regression)")

    # --- Part 6: Capture Rate Deep Dive ---
    report_lines.append("")
    report_lines.append("CAPTURE RATE REGRESSION (correct-direction trades)")

    exit_df = df[df['dir_correct'] == 1].copy()
    if len(exit_df) > 10:
        if 'exit_workers' in exit_df.columns:
            exit_df['ew_exit'] = exit_df['exit_workers'].apply(_parse_workers)
            exit_df['w5m_exit_d'] = exit_df['ew_exit'].apply(lambda w: w.get('5m', {}).get('d', NEUTRAL_THRESHOLD))

            # W5m Flipped Logic
            # LONG trade: if exit w5m_d < 0.5 -> flipped
            # SHORT trade: if exit w5m_d > 0.5 -> flipped
            long_flip = (exit_df['direction'] == 'LONG') & (exit_df['w5m_exit_d'] < NEUTRAL_THRESHOLD)
            short_flip = (exit_df['direction'] == 'SHORT') & (exit_df['w5m_exit_d'] > NEUTRAL_THRESHOLD)
            exit_df['w5m_flipped'] = (long_flip | short_flip).astype(int)
        else:
             exit_df['w5m_flipped'] = 0

        # Features available in exit_df
        available_feats = [f for f in CAPTURE_REGRESSION_FEATURES if f in exit_df.columns]
        for f in CAPTURE_REGRESSION_FEATURES:
            if f not in exit_df.columns:
                exit_df[f] = 0.0

        X = exit_df[CAPTURE_REGRESSION_FEATURES].fillna(0)
        y = exit_df['capture_pct']

        model = LinearRegression()
        model.fit(X, y)
        r2 = model.score(X, y)

        report_lines.append(f"  R² = {r2:.3f} (n={len(exit_df)})")
        report_lines.append(f"  {'Feature':<20} {'Coef':>10}")

        coefs = list(zip(CAPTURE_REGRESSION_FEATURES, model.coef_))
        coefs.sort(key=lambda x: abs(x[1]), reverse=True)

        for feat, coef in coefs:
            report_lines.append(f"  {feat:<20} {coef:>10.2f}")
    else:
        report_lines.append("  (Insufficient correct-direction trades)")

    # --- Part 7: Session x Direction Interaction ---
    report_lines.append("")
    report_lines.append("SESSION x DIRECTION (Mean PnL)")

    if 'session' in df.columns and 'direction' in df.columns:
        pivot = df.pivot_table(index='session', columns='direction', values='actual_pnl', aggfunc='mean')
        # Ensure columns exist
        if 'LONG' not in pivot.columns: pivot['LONG'] = np.nan
        if 'SHORT' not in pivot.columns: pivot['SHORT'] = np.nan

        report_lines.append(f"  {'Session':<15} {'LONG':>10} {'SHORT':>10} {'Delta':>10}")
        report_lines.append(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10}")

        # Sort sessions
        ordered_sessions = ['Overnight', 'Europe/PreMkt', 'US_Open', 'US_Mid', 'US_Close']
        for sess in ordered_sessions:
            if sess in pivot.index:
                l_val = pivot.loc[sess, 'LONG']
                s_val = pivot.loc[sess, 'SHORT']
                delta = (l_val if not np.isnan(l_val) else 0) - (s_val if not np.isnan(s_val) else 0)

                l_str = f"${l_val:.0f}" if not np.isnan(l_val) else "N/A"
                s_str = f"${s_val:.0f}" if not np.isnan(s_val) else "N/A"
                d_str = f"${delta:+.0f}"

                report_lines.append(f"  {sess:<15} {l_str:>10} {s_str:>10} {d_str:>10}")

    return "\n".join(report_lines)
