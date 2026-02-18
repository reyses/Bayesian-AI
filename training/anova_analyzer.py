"""
ANOVA Factor Analyzer
Statistical factor analysis on Monte Carlo results.
"""

import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from training.monte_carlo_engine import ComboResult

class ANOVAAnalyzer:
    """
    Statistical factor analysis on Monte Carlo results.
    Identifies which dimensions matter for profitability.
    """

    def _bucket(self, value, thresholds):
        for i, t in enumerate(thresholds):
            if value < t:
                return i
        return len(thresholds)

    def _compute_sharpe(self, trades):
        if len(trades) < 2:
            return 0.0
        pnls = [t.pnl for t in trades]
        std = np.std(pnls)
        if std == 0:
            return 0.0
        return np.mean(pnls) / std

    def analyze(self, results_db: Dict[Tuple[int, str], ComboResult]):
        """
        Run multi-factor ANOVA on results.

        Factors tested:
        - Timeframe (categorical: 1m, 5m, 15m, 1h, 4h)
        - Template cluster (categorical: template_id)
        - Stop width bucket (3 levels: tight/medium/wide)
        - Take profit bucket (3 levels: tight/medium/wide)
        - Hold time bucket (3 levels: short/medium/long)

        Response variable: PnL per trade (or Sharpe ratio)
        """
        if not results_db:
            print("No results to analyze.")
            return {}, []

        # Build DataFrame from all iteration results
        rows = []
        for (tid, tf), combo_result in results_db.items():
            for iteration in combo_result.top_iterations:
                if iteration.num_trades > 0:
                    rows.append({
                        'template_id': tid,
                        'timeframe': tf,
                        'stop_bucket': self._bucket(iteration.params.get('stop_loss_ticks', 15), [12, 18]),
                        'tp_bucket': self._bucket(iteration.params.get('take_profit_ticks', 40), [40, 50]),
                        'hold_bucket': self._bucket(iteration.params.get('max_hold_bars', 50), [30, 100]),
                        'pnl_per_trade': iteration.total_pnl / iteration.num_trades,
                        'win_rate': iteration.win_rate,
                        'sharpe': iteration.sharpe,
                        'total_pnl': iteration.total_pnl,
                        'num_trades': iteration.num_trades,
                        'params': iteration.params
                    })

        df = pd.DataFrame(rows)
        if df.empty:
            print("No valid trades found for ANOVA.")
            return {}, []

        results = {}
        # One-way ANOVA per factor
        for factor in ['timeframe', 'template_id', 'stop_bucket', 'tp_bucket', 'hold_bucket']:
            groups = [group['pnl_per_trade'].values
                      for _, group in df.groupby(factor) if len(group) >= 10]
            if len(groups) >= 2:
                try:
                    f_stat, p_value = f_oneway(*groups)
                    results[factor] = {'f_stat': f_stat, 'p_value': p_value}
                except Exception as e:
                    print(f"ANOVA failed for {factor}: {e}")

        # Report top factors (lowest p-value = most significant)
        print("\nANOVA FACTOR SIGNIFICANCE:")
        print("-" * 50)
        for factor, stats in sorted(results.items(), key=lambda x: x[1]['p_value']):
            sig = "***" if stats['p_value'] < 0.001 else "**" if stats['p_value'] < 0.01 else "*" if stats['p_value'] < 0.05 else "ns"
            print(f"  {factor:20s}  F={stats['f_stat']:8.2f}  p={stats['p_value']:.4f}  {sig}")

        # Best (template, timeframe) combos
        combo_stats = df.groupby(['template_id', 'timeframe']).agg({
            'total_pnl': 'mean',
            'win_rate': 'mean',
            'num_trades': 'mean',
            'sharpe': 'mean'
        }).reset_index()

        # Filter: need meaningful trade count
        viable = combo_stats[combo_stats['num_trades'] >= 5]

        # We need to return list of (tid, tf, params) for top combos.
        # But 'params' are per iteration. Which params do we return?
        # The best params for that combo?
        # Or just the (tid, tf) tuple?
        # Thompson refiner expects "top_combos" to include params?
        # The prompt for ThompsonRefiner says: "List of (template_id, timeframe, base_params)"
        # So we should find the best iteration for each top combo and return its params.

        if viable.empty:
            print("No viable combos found.")
            return results, []

        top_combos_df = viable.nlargest(20, 'sharpe')

        print("\nTOP 20 (TEMPLATE × TIMEFRAME) COMBOS BY SHARPE:")
        print(top_combos_df.to_string(index=False))

        top_combos_list = []
        for _, row in top_combos_df.iterrows():
            tid = row['template_id']
            tf = row['timeframe']
            # Find best params for this combo from original results
            if (tid, tf) in results_db:
                combo = results_db[(tid, tf)]
                if combo.best_params:
                    top_combos_list.append((tid, tf, combo.best_params))
                elif combo.top_iterations:
                    top_combos_list.append((tid, tf, combo.top_iterations[0].params))

        return results, top_combos_list
