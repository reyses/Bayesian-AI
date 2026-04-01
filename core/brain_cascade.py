"""
Brain Cascade — 4-layer Bayesian calibration chain.

CNN → IS Brain → OOS Brain → Live Brain

Each brain calibrates the probability from the previous layer using
Bayesian updates from observed trade outcomes at that stage.

Usage:
    cascade = BrainCascade(checkpoint_dir='checkpoints/brains')
    cascade.build_is_brain()   # after IS forward pass
    cascade.freeze_is()
    cascade.build_oos_brain()  # after OOS forward pass
    cascade.freeze_oos()
    cascade.init_live()        # copies OOS brain → live brain

    # Per-bar in production:
    p_calibrated = cascade.calibrate(template_id, 'LONG', p_cnn_raw)

    # After each trade:
    cascade.update(template_id, 'LONG', p_at_entry, pnl)

    # Rollback:
    cascade.rollback_live()  # reset live brain to OOS checkpoint
"""
import os
import pickle
import copy
from collections import defaultdict
from typing import Optional, Dict, Any


# Bayesian prior: mildly pessimistic (assumes 45% base rate until data says otherwise)
DEFAULT_PRIOR_WEIGHT = 10.0
DEFAULT_PRIOR_RATE = 0.45


class CalibrationBrain:
    """Single calibration layer: maps (template_id, direction) → observed win rate.

    Applies Bayesian update: blends CNN probability with observed win rate,
    weighted by sample count.
    """

    def __init__(self, prior_weight: float = DEFAULT_PRIOR_WEIGHT,
                 prior_rate: float = DEFAULT_PRIOR_RATE):
        self.prior_weight = prior_weight
        self.prior_rate = prior_rate
        # (template_id, direction) → {wins, losses, total, pnl_sum}
        self.table: Dict[tuple, Dict[str, float]] = defaultdict(
            lambda: {'wins': 0, 'losses': 0, 'total': 0, 'pnl_sum': 0.0}
        )
        self._frozen = False

    def calibrate(self, template_id, direction: str, p_input: float) -> float:
        """Calibrate input probability using observed data for this (template, direction).

        When no data exists, returns p_input unchanged.
        As observations grow, gradually shifts toward observed win rate.

        Returns: calibrated probability in (0, 1)
        """
        key = (template_id, direction)
        data = self.table.get(key)
        if data is None or data['total'] == 0:
            return p_input

        observed_rate = (data['wins'] + 1) / (data['total'] + 2)  # Laplace smoothing
        n = data['total']

        # Bayesian blend: more data = more weight on observed rate
        # At n=0: 100% input, at n=prior_weight: 50/50, at n>>prior_weight: ~100% observed
        weight_observed = n / (n + self.prior_weight)
        weight_input = 1.0 - weight_observed

        return weight_input * p_input + weight_observed * observed_rate

    def update(self, template_id, direction: str, pnl: float):
        """Record trade outcome. No-op if frozen."""
        if self._frozen:
            return

        key = (template_id, direction)
        entry = self.table[key]
        if pnl > 0:
            entry['wins'] += 1
        else:
            entry['losses'] += 1
        entry['total'] += 1
        entry['pnl_sum'] += pnl

    def freeze(self):
        """Stop learning. State becomes read-only."""
        self._frozen = True

    def unfreeze(self):
        """Resume learning."""
        self._frozen = False

    def get_stats(self, template_id, direction: str) -> Optional[Dict]:
        """Return stats for (template, direction) or None."""
        key = (template_id, direction)
        if key not in self.table or self.table[key]['total'] == 0:
            return None
        return dict(self.table[key])

    def save(self, path: str):
        """Persist to disk."""
        data = {
            'table': dict(self.table),
            'prior_weight': self.prior_weight,
            'prior_rate': self.prior_rate,
            'frozen': self._frozen,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str):
        """Load from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.table = defaultdict(
            lambda: {'wins': 0, 'losses': 0, 'total': 0, 'pnl_sum': 0.0},
            data['table']
        )
        self.prior_weight = data.get('prior_weight', DEFAULT_PRIOR_WEIGHT)
        self.prior_rate = data.get('prior_rate', DEFAULT_PRIOR_RATE)
        self._frozen = data.get('frozen', False)

    @property
    def n_entries(self) -> int:
        return sum(1 for v in self.table.values() if v['total'] > 0)

    @property
    def n_trades(self) -> int:
        return sum(v['total'] for v in self.table.values())


class BrainCascade:
    """4-layer calibration chain: CNN → IS → OOS → Live.

    Each layer refines the probability from the previous layer.
    IS and OOS brains are frozen after their respective passes.
    Live brain continues learning in production.
    """

    def __init__(self, checkpoint_dir: str = 'checkpoints/brains',
                 prior_weight: float = DEFAULT_PRIOR_WEIGHT):
        self.checkpoint_dir = checkpoint_dir
        self.prior_weight = prior_weight
        os.makedirs(checkpoint_dir, exist_ok=True)

        # The four brains (None until initialized)
        self.is_brain: Optional[CalibrationBrain] = None
        self.oos_brain: Optional[CalibrationBrain] = None
        self.live_brain: Optional[CalibrationBrain] = None

    # ── Lifecycle ──────────────────────────────────────────────────

    def init_is_brain(self):
        """Create fresh IS brain for IS forward pass."""
        self.is_brain = CalibrationBrain(prior_weight=self.prior_weight)
        return self.is_brain

    def freeze_is(self):
        """Freeze IS brain and save checkpoint."""
        if self.is_brain:
            self.is_brain.freeze()
            self.is_brain.save(os.path.join(self.checkpoint_dir, 'is_brain.pkl'))
            print(f"  [BrainCascade] IS brain frozen: {self.is_brain.n_entries} entries, "
                  f"{self.is_brain.n_trades} trades")

    def init_oos_brain(self):
        """Create OOS brain as copy of IS brain."""
        if self.is_brain is None:
            # Try loading from disk
            is_path = os.path.join(self.checkpoint_dir, 'is_brain.pkl')
            if os.path.exists(is_path):
                self.is_brain = CalibrationBrain(prior_weight=self.prior_weight)
                self.is_brain.load(is_path)
            else:
                raise RuntimeError("IS brain not found — run IS pass first")

        self.oos_brain = copy.deepcopy(self.is_brain)
        self.oos_brain.unfreeze()
        return self.oos_brain

    def freeze_oos(self):
        """Freeze OOS brain and save checkpoint."""
        if self.oos_brain:
            self.oos_brain.freeze()
            self.oos_brain.save(os.path.join(self.checkpoint_dir, 'oos_brain.pkl'))
            print(f"  [BrainCascade] OOS brain frozen: {self.oos_brain.n_entries} entries, "
                  f"{self.oos_brain.n_trades} trades")

    def init_live(self):
        """Create live brain as copy of OOS brain. Ready for production."""
        if self.oos_brain is None:
            oos_path = os.path.join(self.checkpoint_dir, 'oos_brain.pkl')
            if os.path.exists(oos_path):
                self.oos_brain = CalibrationBrain(prior_weight=self.prior_weight)
                self.oos_brain.load(oos_path)
            else:
                raise RuntimeError("OOS brain not found — run OOS pass first")

        self.live_brain = copy.deepcopy(self.oos_brain)
        self.live_brain.unfreeze()
        print(f"  [BrainCascade] Live brain initialized from OOS: "
              f"{self.live_brain.n_entries} entries, {self.live_brain.n_trades} trades")
        return self.live_brain

    def rollback_live(self):
        """Reset live brain to OOS checkpoint."""
        oos_path = os.path.join(self.checkpoint_dir, 'oos_brain.pkl')
        if not os.path.exists(oos_path):
            raise RuntimeError("OOS brain checkpoint not found")
        self.live_brain = CalibrationBrain(prior_weight=self.prior_weight)
        self.live_brain.load(oos_path)
        self.live_brain.unfreeze()
        print(f"  [BrainCascade] Live brain ROLLED BACK to OOS checkpoint")

    def save_live(self):
        """Save live brain state (call after every trade or end of session)."""
        if self.live_brain:
            self.live_brain.save(os.path.join(self.checkpoint_dir, 'live_brain.pkl'))

    # ── Calibration ────────────────────────────────────────────────

    def calibrate(self, template_id, direction: str, p_cnn: float) -> float:
        """Run probability through the full cascade: CNN → IS → OOS → Live.

        Each brain that exists applies its calibration. Missing brains pass through.
        """
        p = p_cnn

        if self.is_brain:
            p = self.is_brain.calibrate(template_id, direction, p)

        if self.oos_brain:
            p = self.oos_brain.calibrate(template_id, direction, p)

        if self.live_brain:
            p = self.live_brain.calibrate(template_id, direction, p)

        return p

    def update(self, template_id, direction: str, pnl: float):
        """Record trade outcome in the active (non-frozen) brain.

        During IS pass: updates IS brain.
        During OOS pass: updates OOS brain.
        During live: updates live brain.
        """
        if self.live_brain and not self.live_brain._frozen:
            self.live_brain.update(template_id, direction, pnl)
        elif self.oos_brain and not self.oos_brain._frozen:
            self.oos_brain.update(template_id, direction, pnl)
        elif self.is_brain and not self.is_brain._frozen:
            self.is_brain.update(template_id, direction, pnl)

    # ── Diagnostics ────────────────────────────────────────────────

    def divergence_report(self) -> str:
        """Compare calibrations across brains for the same inputs."""
        lines = ["Brain Cascade Divergence Report", "=" * 50]

        # Collect all (template, direction) keys across all brains
        all_keys = set()
        for brain in [self.is_brain, self.oos_brain, self.live_brain]:
            if brain:
                all_keys.update(k for k, v in brain.table.items() if v['total'] >= 3)

        if not all_keys:
            return "No data in any brain yet."

        lines.append(f"{'Key':<30} {'IS':>8} {'OOS':>8} {'Live':>8} {'Drift':>8}")
        lines.append("-" * 70)

        for key in sorted(all_keys, key=str):
            p_test = 0.5  # neutral test input
            p_is = self.is_brain.calibrate(*key, p_test) if self.is_brain else p_test
            p_oos = self.oos_brain.calibrate(*key, p_test) if self.oos_brain else p_test
            p_live = self.live_brain.calibrate(*key, p_test) if self.live_brain else p_test
            drift = abs(p_live - p_oos) if self.live_brain and self.oos_brain else 0
            lines.append(f"{str(key):<30} {p_is:>8.3f} {p_oos:>8.3f} {p_live:>8.3f} {drift:>8.3f}")

        return "\n".join(lines)

    def load_all(self):
        """Load all available brain checkpoints from disk."""
        is_path = os.path.join(self.checkpoint_dir, 'is_brain.pkl')
        oos_path = os.path.join(self.checkpoint_dir, 'oos_brain.pkl')
        live_path = os.path.join(self.checkpoint_dir, 'live_brain.pkl')

        if os.path.exists(is_path):
            self.is_brain = CalibrationBrain(prior_weight=self.prior_weight)
            self.is_brain.load(is_path)

        if os.path.exists(oos_path):
            self.oos_brain = CalibrationBrain(prior_weight=self.prior_weight)
            self.oos_brain.load(oos_path)

        if os.path.exists(live_path):
            self.live_brain = CalibrationBrain(prior_weight=self.prior_weight)
            self.live_brain.load(live_path)

        loaded = sum(1 for b in [self.is_brain, self.oos_brain, self.live_brain] if b)
        print(f"  [BrainCascade] Loaded {loaded}/3 brains from {self.checkpoint_dir}")
