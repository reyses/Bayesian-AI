"""
Strategy Brain — Bayesian calibration layer over the Strategy Router NN.

The NN provides the prior (pattern recognition from training data).
The brain accumulates live evidence and adjusts the NN's predictions.

Brain keys: (strategy_id, state_bin)
  strategy_id = (direction, duration) e.g., ('long', 5)
  state_bin   = discretized 79D state (binned key features for lookup)

The brain tracks per key:
  - wins / losses / total (for calibrated P(profit))
  - actual PnL history (for calibrated expected PnL)
  - actual half-life (for calibrated duration)
  - actual drawdown (for calibrated risk)

Spec: docs/Active/NN_SPEC.md
"""
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, Optional


# State binning: discretize key 79D features into a hashable key
# We bin: 1m_z_se, 1m_dmi_diff, 1m_variance_ratio, 1h_z_se, 1h_dmi_diff
# This gives ~5 dimensions of context without being too sparse
STATE_BIN_FEATURES = {
    '1m_z_se': {'edges': [-3, -2, -1, 0, 1, 2, 3]},
    '1m_dmi_diff': {'edges': [-10, -5, 0, 5, 10]},
    '1m_variance_ratio': {'edges': [0.3, 0.5, 0.7, 1.0, 1.5]},
    '1h_z_se': {'edges': [-2, -1, 0, 1, 2]},
    '1h_dmi_diff': {'edges': [-10, -5, 0, 5, 10]},
}


def bin_value(value: float, edges: list) -> int:
    """Bin a continuous value into a discrete bucket."""
    for i, edge in enumerate(edges):
        if value < edge:
            return i
    return len(edges)


def compute_state_bin(features_79d: np.ndarray, feature_names: list) -> tuple:
    """Discretize 79D features into a hashable state bin.

    Args:
        features_79d: 79D feature vector
        feature_names: list of feature names (FEATURE_NAMES_79D)

    Returns:
        Tuple of ints — the binned state key
    """
    bins = []
    for feat_name, config in STATE_BIN_FEATURES.items():
        if feat_name in feature_names:
            idx = feature_names.index(feat_name)
            val = features_79d[idx] if idx < len(features_79d) else 0.0
            bins.append(bin_value(val, config['edges']))
        else:
            bins.append(0)
    return tuple(bins)


class StrategyBrain:
    """Bayesian calibration over NN predictions.

    Usage:
        brain = StrategyBrain()

        # At prediction time:
        nn_output = model.predict(features)
        calibrated = brain.calibrate(nn_output, features_79d)

        # After trade completes:
        brain.update(trade_result)
    """

    def __init__(self, min_observations: int = 5, blend_weight: float = 0.3):
        """
        Args:
            min_observations: minimum trades before brain adjusts NN output
            blend_weight: how much to trust brain vs NN (0 = all NN, 1 = all brain)
                          ramps up with observations: effective_weight = blend * min(1, n_obs/20)
        """
        self.min_obs = min_observations
        self.blend_weight = blend_weight

        # Per strategy_id: aggregate stats (direction + duration level)
        self.strategy_stats = defaultdict(lambda: {
            'wins': 0, 'losses': 0, 'total': 0,
            'pnl_sum': 0.0, 'pnl_list': [],
            'dd_sum': 0.0, 'dd_list': [],
            'actual_dur_sum': 0.0,  # actual hold durations
        })

        # Per (strategy_id, state_bin): fine-grained context-aware stats
        self.context_stats = defaultdict(lambda: {
            'wins': 0, 'losses': 0, 'total': 0,
            'pnl_sum': 0.0,
        })

        # Trade log (for analysis)
        self.trade_log = []

    def calibrate(self, nn_output: dict, features_79d: np.ndarray,
                  feature_names: list) -> dict:
        """Blend NN prediction with brain evidence.

        Args:
            nn_output: dict from StrategyRouterNN.predict()
            features_79d: raw 79D feature vector
            feature_names: FEATURE_NAMES_79D

        Returns:
            Calibrated prediction dict (same keys as nn_output + brain adjustments)
        """
        strategy_id = nn_output['strategy_id']
        state_bin = compute_state_bin(features_79d, feature_names)
        brain_key = (strategy_id, state_bin)

        result = dict(nn_output)  # copy
        result['state_bin'] = state_bin
        result['brain_key'] = brain_key

        # Strategy-level calibration
        s_stats = self.strategy_stats[strategy_id]
        n_obs = s_stats['total']

        if n_obs >= self.min_obs:
            # Ramp up brain influence with more observations
            effective_weight = self.blend_weight * min(1.0, n_obs / 20.0)

            # Calibrate P(profit)
            brain_wr = s_stats['wins'] / max(s_stats['total'], 1)
            nn_p = nn_output['p_profit']
            result['p_profit'] = (1 - effective_weight) * nn_p + effective_weight * brain_wr
            result['brain_wr'] = brain_wr

            # Calibrate expected PnL
            if s_stats['pnl_list']:
                brain_avg_pnl = np.mean(s_stats['pnl_list'][-50:])  # last 50 trades
                nn_pnl = nn_output['expected_pnl']
                result['expected_pnl'] = (1 - effective_weight) * nn_pnl + effective_weight * brain_avg_pnl
                result['brain_avg_pnl'] = brain_avg_pnl

            # Calibrate expected drawdown
            if s_stats['dd_list']:
                brain_avg_dd = np.mean(s_stats['dd_list'][-50:])
                nn_dd = nn_output['expected_drawdown']
                result['expected_drawdown'] = (1 - effective_weight) * nn_dd + effective_weight * brain_avg_dd
                result['brain_avg_dd'] = brain_avg_dd

        result['n_observations'] = n_obs
        result['effective_blend'] = self.blend_weight * min(1.0, n_obs / 20.0) if n_obs >= self.min_obs else 0.0

        # Context-level signal (more specific — this state + this strategy)
        c_stats = self.context_stats[brain_key]
        result['context_n'] = c_stats['total']
        if c_stats['total'] >= 3:
            result['context_wr'] = c_stats['wins'] / max(c_stats['total'], 1)
        else:
            result['context_wr'] = None  # not enough data

        return result

    def update(self, trade_result: dict):
        """Update brain after a trade completes.

        Args:
            trade_result: dict with keys:
              strategy_id: (direction, duration)
              state_bin: tuple from compute_state_bin
              actual_pnl: realized PnL in dollars
              actual_drawdown: realized max drawdown
              actual_duration: actual bars held
              was_profitable: bool
        """
        sid = trade_result['strategy_id']
        brain_key = (sid, trade_result.get('state_bin', ()))

        # Strategy-level update
        s = self.strategy_stats[sid]
        s['total'] += 1
        if trade_result['was_profitable']:
            s['wins'] += 1
        else:
            s['losses'] += 1
        s['pnl_sum'] += trade_result['actual_pnl']
        s['pnl_list'].append(trade_result['actual_pnl'])
        s['dd_sum'] += trade_result.get('actual_drawdown', 0)
        s['dd_list'].append(trade_result.get('actual_drawdown', 0))
        s['actual_dur_sum'] += trade_result.get('actual_duration', 0)

        # Context-level update
        c = self.context_stats[brain_key]
        c['total'] += 1
        if trade_result['was_profitable']:
            c['wins'] += 1
        else:
            c['losses'] += 1
        c['pnl_sum'] += trade_result['actual_pnl']

        # Trade log
        self.trade_log.append(trade_result)

    def get_strategy_summary(self) -> str:
        """Human-readable summary of all tracked strategies."""
        lines = ['Strategy Brain Summary:']
        lines.append(f'  Strategies tracked: {len(self.strategy_stats)}')
        lines.append(f'  Context keys tracked: {len(self.context_stats)}')
        lines.append(f'  Total trades: {sum(s["total"] for s in self.strategy_stats.values())}')
        lines.append('')
        lines.append(f'  {"Strategy":<20} {"N":>5} {"WR":>6} {"Avg PnL":>8} {"Avg DD":>8}')
        lines.append(f'  {"-"*55}')

        for sid, stats in sorted(self.strategy_stats.items(),
                                  key=lambda x: -x[1]['total']):
            if stats['total'] == 0:
                continue
            wr = stats['wins'] / stats['total'] * 100
            avg_pnl = stats['pnl_sum'] / stats['total']
            avg_dd = stats['dd_sum'] / stats['total']
            label = f'{sid[0]}_{sid[1]}' if isinstance(sid, tuple) else str(sid)
            lines.append(f'  {label:<20} {stats["total"]:>5} {wr:>5.1f}% ${avg_pnl:>7.1f} ${avg_dd:>7.1f}')

        return '\n'.join(lines)

    def should_trade(self, calibrated_output: dict, min_p_profit: float = 0.52) -> bool:
        """Should we take this trade based on calibrated probability?

        Conservative: requires P(profit) > threshold.
        The execution layer handles sizing — this is the go/no-go decision.
        """
        if calibrated_output['direction'] == 'skip':
            return False
        if calibrated_output['p_profit'] < min_p_profit:
            return False

        # If brain has seen this context and it's losing, veto
        context_wr = calibrated_output.get('context_wr')
        if context_wr is not None and context_wr < 0.40:
            return False

        return True
