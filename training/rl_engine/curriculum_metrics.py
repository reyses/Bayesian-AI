import os
import subprocess
import numpy as np
import pandas as pd
from scipy import stats

# ─── Build provenance ─────────────────────────────────────────────────────────
# Stamp every persisted diagnostic with the engine build so a reader can tell
# whether the data came from a pre- or post-(schema + leakage) fix engine.
# Cached at import time; a single git call per process.
def _resolve_build_tag() -> str:
    try:
        sha = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL, timeout=2,
        ).decode().strip()
    except Exception:
        sha = 'unknown'
    try:
        dirty_lines = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            stderr=subprocess.DEVNULL, timeout=2,
        ).decode().strip().splitlines()
        if dirty_lines:
            sha = f'{sha}-dirty'
    except Exception:
        pass
    return sha

BUILD_TAG = _resolve_build_tag()
DIAGNOSTIC_SCHEMA_VERSION = 'v2'  # bumped to v2 for full trajectory metadata (entry_bar, exit_bar, direction)
# Where per-segment raw trade parquets land. Gitignored via *.parquet rule.
OOS_DIAGNOSTICS_DIR = os.path.join('reports', 'oos_diagnostics')

def compute_mode_ci(data_array, confidence=0.95, num_bootstraps=1000):
    if len(data_array) < 10:
        return 0.0, (0.0, 0.0)
        
    modes = []
    for _ in range(num_bootstraps):
        sample = np.random.choice(data_array, size=len(data_array), replace=True)
        counts, bins = np.histogram(sample, bins=30)
        max_bin_idx = np.argmax(counts)
        modes.append((bins[max_bin_idx] + bins[max_bin_idx+1]) / 2.0)
        
    modes = np.array(modes)
    lower = np.percentile(modes, (1 - confidence) / 2.0 * 100)
    upper = np.percentile(modes, (1 + confidence) / 2.0 * 100)
    
    counts, bins = np.histogram(data_array, bins=30)
    max_bin_idx = np.argmax(counts)
    actual_mode = (bins[max_bin_idx] + bins[max_bin_idx+1]) / 2.0
    return actual_mode, (lower, upper)

def compute_mean_ci(data_array, confidence=0.95, num_bootstraps=2000):
    if len(data_array) < 10:
        return 0.0, (0.0, 0.0)
    
    means = []
    for _ in range(num_bootstraps):
        sample = np.random.choice(data_array, size=len(data_array), replace=True)
        means.append(np.mean(sample))
        
    means = np.array(means)
    lower = np.percentile(means, (1 - confidence) / 2.0 * 100)
    upper = np.percentile(means, (1 + confidence) / 2.0 * 100)
    return np.mean(data_array), (lower, upper)

def compute_cvar(pnls, alpha=0.05):
    if len(pnls) == 0: return 0.0
    sorted_pnls = np.sort(pnls)
    cutoff_idx = max(1, int(len(pnls) * alpha))
    return float(np.mean(sorted_pnls[:cutoff_idx]))

def compute_worst_loss(pnls):
    if len(pnls) == 0: return 0.0
    return float(np.min(pnls))

def calculate_max_drawdown(trade_pnls):
    if len(trade_pnls) == 0:
        return 0.0
    cumulative = np.cumsum(trade_pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    return np.max(drawdowns)

def evaluate_is_metrics(trade_pnls, trade_durations):
    """
    Computes basic trajectory metrics for In-Sample monitoring.
    AUC is strictly removed from all logic.
    """
    if len(trade_pnls) == 0:
        return {'metric_n': 0.0, 'total_pnl': 0.0, 'pnl_mode_ci': (0.0,0.0), 'dur_mode_ci': (0.0,0.0), 'avg_loss': 0.0}
        
    trade_pnls = np.array(trade_pnls)
    trade_durations = np.array(trade_durations)
    
    wins = trade_pnls[trade_pnls > 0]
    losses = trade_pnls[trade_pnls <= 0]
    gross_profit = np.sum(wins) if len(wins) > 0 else 0.0
    gross_loss = np.sum(np.abs(losses)) if len(losses) > 0 else 1e-9
    metric_n = (gross_profit / gross_loss) - 1.0
    
    _, pnl_ci = compute_mode_ci(trade_pnls)
    _, dur_ci = compute_mode_ci(trade_durations)
    
    return {
        'metric_n': metric_n,
        'total_pnl': np.sum(trade_pnls),
        'pnl_mode_ci': pnl_ci,
        'dur_mode_ci': dur_ci,
        'trade_count': len(trade_pnls)
    }

class OOSDiagnosticsSuite:
    """
    Pooled-source-of-truth OOS diagnostics.

    All metrics are computed from the RAW per-trade arrays. Reconstruction
    from moments is prohibited. Per-segment raw arrays are also persisted
    to disk (parquet) so the chart generator never needs to fabricate.

    Diagnostics here are READ-ONLY: no value returned by this suite is
    consumed by any stopping rule, loss term, gradient, or curriculum
    graduation gate. The training loop reads them only for print().
    """

    def __init__(self):
        self.all_pnls = []
        self.all_durs = []
        self.all_mfe_avail = []
        self.all_mfe_trade = []
        self.all_mae = []
        # Per-segment book-keeping for "% segments with PF>1" + provenance
        self.segment_pfs = []           # one PF per segment (raw)
        self.segment_net_pnls = []      # one net PnL per segment (raw)
        self.segment_trade_counts = []  # one n per segment (raw)
        self._segment_counter = 0       # auto-increment when caller omits id

    def add_segment_data(self, pnls, durs, mfe_avail, mfe_trade, mae, metadata=None,
                         segment_id=None, train_dates=None, eval_dates=None):
        """
        Append a segment's raw per-trade arrays.

        The trainer's existing call sites pass only the first five positional
        args; the new kwargs are optional so the call signature stays
        backwards-compatible. When segment_id is omitted we auto-increment.

        Side effect: writes the raw per-trade arrays for this segment to
        `reports/oos_diagnostics/seg{N}_{build}.parquet` so generate_oos_chart.py
        can plot from real data.
        """
        self.all_pnls.extend(pnls)
        self.all_durs.extend(durs)
        self.all_mfe_avail.extend(mfe_avail)
        self.all_mfe_trade.extend(mfe_trade)
        self.all_mae.extend(mae)

        self._segment_counter += 1
        if segment_id is None:
            segment_id = self._segment_counter

        seg_pnls = np.asarray(pnls, dtype=np.float64)
        seg_durs = np.asarray(durs, dtype=np.float64)
        seg_mfe_avail = np.asarray(mfe_avail, dtype=np.float64)
        seg_mfe_trade = np.asarray(mfe_trade, dtype=np.float64)
        seg_mae = np.asarray(mae, dtype=np.float64)

        if len(seg_pnls) == 0:
            return {}

        # Persist raw per-trade rows for this segment
        self._persist_segment_parquet(
            seg_pnls, seg_durs, seg_mfe_avail, seg_mfe_trade, seg_mae,
            metadata=metadata,
            segment_id=segment_id,
            train_dates=train_dates or '',
            eval_dates=eval_dates or '',
        )

        # Per-segment PF for the "% segments with PF>1" pooled metric
        wins = seg_pnls[seg_pnls > 0]
        losses = seg_pnls[seg_pnls <= 0]
        gp = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gl = float(np.sum(np.abs(losses))) if len(losses) > 0 else 0.0
        seg_pf = (gp / gl) if gl > 0 else (float('inf') if gp > 0 else 0.0)
        self.segment_pfs.append(seg_pf)
        self.segment_net_pnls.append(float(np.sum(seg_pnls)))
        self.segment_trade_counts.append(int(len(seg_pnls)))

        _, pnl_mode_ci = compute_mode_ci(seg_pnls)
        _, dur_mode_ci = compute_mode_ci(seg_durs)

        cap_avail = np.sum(seg_pnls) / (np.sum(seg_mfe_avail) + 1e-9)
        cap_trade = np.sum(seg_pnls) / (np.sum(seg_mfe_trade) + 1e-9)

        return {
            'segment_id': segment_id,
            'build_tag': BUILD_TAG,
            'trade_count': len(pnls),
            'pnl_mode_ci': pnl_mode_ci,
            'dur_mode_ci': dur_mode_ci,
            'max_drawdown': calculate_max_drawdown(seg_pnls),
            'cap_vs_avail': cap_avail,
            'cap_vs_trade': cap_trade,
            'avg_mae': float(np.mean(seg_mae)) if len(seg_mae) > 0 else 0.0,
            'segment_pf': seg_pf,
            'parquet_path': self._segment_parquet_path(segment_id),
        }

    def _segment_parquet_path(self, segment_id) -> str:
        return os.path.join(
            OOS_DIAGNOSTICS_DIR,
            f'seg{int(segment_id):03d}_{BUILD_TAG}.parquet',
        )

    def _persist_segment_parquet(self, seg_pnls, seg_durs, seg_mfe_avail,
                                 seg_mfe_trade, seg_mae, metadata,
                                 segment_id, train_dates, eval_dates):
        """
        Write the raw per-trade arrays for one segment to parquet.

        Schema (v1):
          pnl_net          : float64  — realized PnL per trade
          duration_bars    : float64  — bars held
          mfe_available    : float64  — direction-aware day-bounded MFE oracle
          mfe_in_trade     : float64  — MFE within the trade window
          mae              : float64  — max adverse excursion
          segment_id       : int      — walk-forward segment index
          build_tag        : str      — engine commit hash (-dirty if uncommitted)
          schema_version   : str      — diagnostic schema version
          train_dates      : str      — "YYYY-MM-DD -> YYYY-MM-DD" (or "")
          eval_dates       : str      — "YYYY-MM-DD -> YYYY-MM-DD" (or "")
        DEFERRED COLUMNS (require trainer-side capture):
          entry_bar, exit_bar, exit_reason, direction.
        Their absence is recorded as schema_version=v1; bump to v2 when added.
        """
        os.makedirs(OOS_DIAGNOSTICS_DIR, exist_ok=True)
        df = pd.DataFrame({
            'pnl_net': seg_pnls,
            'duration_bars': seg_durs,
            'mfe_available': seg_mfe_avail,
            'mfe_in_trade': seg_mfe_trade,
            'mae': seg_mae,
        })
        
        if metadata and len(metadata) == len(seg_pnls):
            df['entry_bar'] = [m[0] for m in metadata]
            df['exit_bar'] = [m[1] for m in metadata]
            df['direction'] = [m[2] for m in metadata]
        else:
            # Fallback for empty or mismatched metadata
            df['entry_bar'] = np.nan
            df['exit_bar'] = np.nan
            df['direction'] = np.nan
            
        df['segment_id'] = int(segment_id)
        df['build_tag'] = BUILD_TAG
        df['schema_version'] = DIAGNOSTIC_SCHEMA_VERSION
        df['train_dates'] = str(train_dates)
        df['eval_dates'] = str(eval_dates)
        path = self._segment_parquet_path(segment_id)
        df.to_parquet(path, index=False)
        
    def get_pooled_diagnostics(self):
        """
        Computes the grand pooled aggregate across all OOS trades.
        None of these numbers influence weights, control flow, or penalties.

        All ratios are POOLED sum/sum, never mean-of-ratios.
        Distribution shape (skew, excess kurtosis) is computed from the
        real per-trade array — a near-0 skew + near-0 excess kurtosis
        with mu near 0 is the random-walk / no-edge signature and is
        surfaced explicitly so a reader can recognise it.
        """
        pnls = np.array(self.all_pnls, dtype=np.float64)
        if len(pnls) == 0:
            return {}

        mean_pnl, mean_ci = compute_mean_ci(pnls)

        cap_avail = np.sum(pnls) / (np.sum(self.all_mfe_avail) + 1e-9)
        cap_trade = np.sum(pnls) / (np.sum(self.all_mfe_trade) + 1e-9)

        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gross_loss = float(np.sum(np.abs(losses))) if len(losses) > 0 else 1e-9
        profit_factor = gross_profit / gross_loss

        # Distribution shape on the REAL pooled array
        skew = float(stats.skew(pnls)) if len(pnls) >= 8 else float('nan')
        excess_kurtosis = (
            float(stats.kurtosis(pnls, fisher=True)) if len(pnls) >= 8
            else float('nan')
        )

        # % of segments with PF>1 (count-based across segments)
        n_segs = len(self.segment_pfs)
        if n_segs > 0:
            n_winning_segs = int(np.sum([pf > 1.0 for pf in self.segment_pfs]))
            pct_segs_pf_gt_1 = n_winning_segs / n_segs
        else:
            n_winning_segs = 0
            pct_segs_pf_gt_1 = float('nan')

        total_net = float(np.sum(pnls))
        n_trades = len(pnls)

        # ── Consistency assertion ────────────────────────────────────────
        # The plot's μ must equal Net/n exactly. The reconstruction-from-
        # log-stats bug (the one this fix replaces) failed this check.
        net_per_trade = total_net / n_trades
        assert abs(float(np.mean(pnls)) - net_per_trade) < 1e-9, (
            f'pooled mean PnL {np.mean(pnls):.6f} != Net/n {net_per_trade:.6f} '
            f'— pipeline corruption between raw array and the aggregate'
        )

        return {
            'build_tag': BUILD_TAG,
            'schema_version': DIAGNOSTIC_SCHEMA_VERSION,
            'total_trades': n_trades,
            'total_segments': n_segs,
            'pooled_mean_pnl': float(mean_pnl),
            'pooled_mean_pnl_ci': (float(mean_ci[0]), float(mean_ci[1])),
            'pooled_cvar_5': compute_cvar(pnls, 0.05),
            'pooled_worst_loss': compute_worst_loss(pnls),
            'pooled_max_drawdown': float(calculate_max_drawdown(pnls)),
            'pooled_cap_vs_avail': float(cap_avail),
            'pooled_cap_vs_trade': float(cap_trade),
            'pooled_avg_mae': float(np.mean(self.all_mae)) if len(self.all_mae) > 0 else 0.0,
            'profit_factor': float(profit_factor),
            'total_net_pnl': total_net,
            'skew': skew,
            'excess_kurtosis': excess_kurtosis,
            'n_segments_pf_gt_1': n_winning_segs,
            'pct_segments_pf_gt_1': pct_segs_pf_gt_1,
        }
