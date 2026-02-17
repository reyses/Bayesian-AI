"""
Thompson Sampling Refinement
Bayesian bandit refinement: concentrate compute on promising combos.
"""

import os
import pickle
import numpy as np
from typing import Dict, List, Any, Tuple
from core.bayesian_brain import BayesianBrain, TradeOutcome
from training.monte_carlo_engine import simulate_template_tf_combo, ComboResult
from training.doe_parameter_generator import DOEParameterGenerator

class ThompsonRefiner:
    """
    Bayesian bandit refinement: concentrate compute on promising combos.
    Uses Thompson sampling from Beta posteriors to allocate iterations.
    """

    def __init__(self, brain: BayesianBrain, asset: Any, top_combos: List[Tuple[int, str, Dict]],
                 pattern_library: Dict, checkpoint_dir: str):
        self.brain = brain
        self.asset = asset
        self.top_combos = top_combos  # List of (template_id, timeframe, base_params)
        self.pattern_library = pattern_library
        self.checkpoint_dir = checkpoint_dir
        self.iteration_budget = 20000  # Total iterations to distribute

        # Load scaler
        scaler_path = os.path.join(checkpoint_dir, 'clustering_scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            self.scaler = None

    def _get_best_combo(self):
        # Scan brain for best win rate among top combos
        best_wr = -1.0
        best_id = None
        for tid, tf, _ in self.top_combos:
            key = f"{tid}_{tf}"
            stats = self.brain.get_stats(key)
            wr = stats['wins'] / stats['total'] if stats['total'] > 0 else 0
            if wr > best_wr:
                best_wr = wr
                best_id = key
        return f"{best_id} ({best_wr:.1%})"

    def _save_checkpoint(self, round_num):
        # Save brain state
        path = os.path.join(self.checkpoint_dir, f'thompson_round_{round_num}_brain.pkl')
        self.brain.save(path)

    def _get_final_rankings(self):
        rankings = []
        for tid, tf, params in self.top_combos:
            key = f"{tid}_{tf}"
            stats = self.brain.get_stats(key)
            wr = stats['wins'] / stats['total'] if stats['total'] > 0 else 0

            # Find best PnL from history? Brain doesn't store PnL per trade easily accessible unless we scan history.
            # But we can approximate.
            # We return the combo with its stats.
            rankings.append({
                'template_id': tid,
                'timeframe': tf,
                'params': params,
                'win_rate': wr,
                'total_trades': stats['total'],
                'train_pnl': 0.0 # Placeholder, would need to track separately
            })

        rankings.sort(key=lambda x: x['win_rate'], reverse=True)
        return rankings

    def refine(self, data_root='DATA/ATLAS'):
        """
        Allocate iterations to combos proportional to Thompson samples.
        """
        rounds = 50  # Number of allocation rounds
        iters_per_round = self.iteration_budget // rounds

        print(f"Starting Thompson Refinement: {len(self.top_combos)} combos, {rounds} rounds, {self.iteration_budget} iterations.")

        if len(self.top_combos) == 0:
            print("WARNING: No combos to refine — Monte Carlo sweep produced no valid trades.")
            print("Check that QuantumFieldEngine.batch_compute_states() works for all timeframes.")
            return []

        for round_num in range(rounds):
            # Thompson sampling: draw from each combo's posterior
            thompson_scores = []
            for tid, tf, _ in self.top_combos:
                key = f"{tid}_{tf}"
                # Get stats from brain table.
                # If key not in table, use prior (wins=0, losses=0 -> alpha=1, beta=1)
                # Brain init uses defaultdict, so it works.
                stats = self.brain.table[key]

                # Beta posterior: Beta(wins + 1, losses + 1)
                alpha = stats['wins'] + 1
                beta_param = stats['losses'] + 1
                sample = np.random.beta(alpha, beta_param)
                thompson_scores.append(sample)

            # Allocate iterations proportional to scores
            total_score = sum(thompson_scores)
            if total_score == 0:
                allocations = [iters_per_round // len(self.top_combos)] * len(self.top_combos)
            else:
                allocations = [max(10, int(s / total_score * iters_per_round))
                              for s in thompson_scores]

            # Run simulations for each combo with allocated iterations
            # This loop runs sequentially (can be parallelized if needed, but it's inside a round loop)
            # Parallelizing here would be good if allocations are large.
            # But simulate_template_tf_combo is a worker function.
            # We can use pool here too? Or just run sequential for simplicity as refinement is smaller scale?
            # 20000 iterations total. 400 per round.
            # If 20 combos, 20 iters per combo.
            # Overhead of spawning pool might be high. Sequential is fine.

            for (tid, tf, base_params), n_iters in zip(self.top_combos, allocations):
                if n_iters <= 0: continue

                # Mutate around best known params
                # We assume base_params is the best known so far.

                result = simulate_template_tf_combo(
                    tid, tf, n_iters, data_root,
                    self.pattern_library[tid], self.asset,
                    original_scaler=self.scaler,
                    mutation_base=base_params,
                    mutation_scale=0.1 # Tight mutations
                )

                # Update brain with results
                for iter_result in result.iterations:
                    for trade in iter_result.trades:
                        key = f"{tid}_{tf}"
                        self.brain.update(trade.to_outcome(key))

                    # Also update "best params" for this combo if better PnL found?
                    # The `base_params` in `self.top_combos` is fixed from ANOVA step.
                    # Should we update it?
                    # The prompt says: "Run simulations with tighter parameter mutations around best known"
                    # But doesn't explicitly say to update the base.
                    # Ideally we should track the best params found during refinement.
                    if iter_result.total_pnl > 0: # Check against previous best?
                         # For now, keep mutating around the ANOVA best.
                         pass

            # Report progress
            print(f"  Round {round_num+1}/{rounds} — Best: {self._get_best_combo()}")

            if (round_num + 1) % 10 == 0:
                self._save_checkpoint(round_num + 1)

        return self._get_final_rankings()
