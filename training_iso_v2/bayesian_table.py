"""Hierarchical Bayesian table over (regime, tier) cells.

For each (regime_idx, tier_name) cell, maintain posteriors that drive the
adaptive exit-threshold optimizer:

    WR              Beta(α, β)            posterior on win-rate
    EV              Normal-Inverse-Gamma  posterior on $/trade (mean+var)
    TtP_s           Normal posterior      seconds to peak (informs time_stop)
    Capture         Beta-like             actual / peak ratio (informs giveback)
    PeakUSD         Normal posterior      peak $ amount   (informs TP)
    MAE_USD         Normal posterior      max-adverse $   (informs SL)

Hierarchy:
    cell (regime, tier)  →  tier-only  →  universal
A thin cell shrinks toward its tier's pooled posterior, which itself shrinks
toward the universal prior. Standard 2-level Empirical Bayes / hierarchical
Beta-Binomial + Normal-Inverse-Gamma.

Offline-only: built from regret labels; not updated during a run.
"""
from __future__ import annotations

import os
import json
import pickle
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from training_iso_v2.state import REGIME_VOCAB
from training_iso_v2.regret import RegretLabel


# ─── Priors (weakly informative; tuned to be near-zero-effect) ──────────

WR_PRIOR_ALPHA = 5.0      # ~5 prior wins
WR_PRIOR_BETA = 5.0       # ~5 prior losses
PEAK_PRIOR_MU = 50.0      # $ peak prior mean
PEAK_PRIOR_VAR = 50.0**2  # weak
MAE_PRIOR_MU = -25.0
MAE_PRIOR_VAR = 25.0**2
EV_PRIOR_MU = 0.0
EV_PRIOR_VAR = 50.0**2
TTP_PRIOR_MU = 1500.0     # 25 min in seconds
TTP_PRIOR_VAR = 600.0**2
CAP_PRIOR_ALPHA = 1.0     # Beta(1,1) uniform on capture ratio
CAP_PRIOR_BETA = 1.0


# ─── Data containers ────────────────────────────────────────────────────

@dataclass
class CellPosterior:
    """Per-cell summary stats. All means/vars updated by Empirical Bayes."""
    n: int = 0
    # WR (Beta on count-based win rate)
    wr_alpha: float = WR_PRIOR_ALPHA
    wr_beta: float = WR_PRIOR_BETA
    # EV ($/trade) — Normal posterior with known prior var (NIG simplified)
    ev_mu: float = EV_PRIOR_MU
    ev_var: float = EV_PRIOR_VAR
    # Peak $ — Normal
    peak_mu: float = PEAK_PRIOR_MU
    peak_var: float = PEAK_PRIOR_VAR
    # MAE $ — Normal
    mae_mu: float = MAE_PRIOR_MU
    mae_var: float = MAE_PRIOR_VAR
    # Time-to-peak (seconds) — Normal
    ttp_mu: float = TTP_PRIOR_MU
    ttp_var: float = TTP_PRIOR_VAR
    # Capture ratio — Beta on (clipped 0..1)
    cap_alpha: float = CAP_PRIOR_ALPHA
    cap_beta: float = CAP_PRIOR_BETA

    def wr_mean(self) -> float:
        return self.wr_alpha / (self.wr_alpha + self.wr_beta)

    def wr_lower(self, q: float = 0.05) -> float:
        # Beta lower-q quantile via SciPy when available; else Wald approx
        try:
            from scipy.stats import beta
            return float(beta.ppf(q, self.wr_alpha, self.wr_beta))
        except Exception:
            mu = self.wr_mean()
            var = (self.wr_alpha * self.wr_beta /
                       ((self.wr_alpha + self.wr_beta)**2 *
                        (self.wr_alpha + self.wr_beta + 1)))
            return max(0.0, mu - 1.645 * np.sqrt(var))

    def ev_lower(self, q: float = 0.05) -> float:
        # Normal lower bound on mean (assuming known variance)
        return self.ev_mu - 1.645 * np.sqrt(max(self.ev_var, 1e-9))

    def cap_mean(self) -> float:
        return self.cap_alpha / (self.cap_alpha + self.cap_beta)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BayesianTable:
    """Hierarchical container: cells + tier-level pools + universal pool."""
    cells: Dict[Tuple[int, str], CellPosterior] = field(default_factory=dict)
    tier_pools: Dict[str, CellPosterior] = field(default_factory=dict)
    universal: CellPosterior = field(default_factory=CellPosterior)

    def get(self, regime_idx: int, tier: str) -> CellPosterior:
        """Return the cell posterior, creating from prior if absent."""
        key = (int(regime_idx), str(tier))
        if key not in self.cells:
            self.cells[key] = CellPosterior()
        return self.cells[key]

    def get_with_shrinkage(self, regime_idx: int, tier: str,
                                 shrinkage_n: float = 30.0) -> CellPosterior:
        """Return a posterior shrunk toward the tier's pool when n is thin.

        shrinkage_n: target effective sample. Cells with n >> shrinkage_n
        return ~the cell posterior unmodified; cells with n << shrinkage_n
        blend toward the tier pool weighted by n / (n + shrinkage_n).
        """
        cell = self.get(regime_idx, tier)
        pool = self.tier_pools.get(str(tier), self.universal)
        if cell.n == 0:
            return pool
        w = cell.n / (cell.n + shrinkage_n)
        out = CellPosterior(
            n=cell.n,
            wr_alpha=w * cell.wr_alpha + (1 - w) * pool.wr_alpha,
            wr_beta=w * cell.wr_beta + (1 - w) * pool.wr_beta,
            ev_mu=w * cell.ev_mu + (1 - w) * pool.ev_mu,
            ev_var=w * cell.ev_var + (1 - w) * pool.ev_var,
            peak_mu=w * cell.peak_mu + (1 - w) * pool.peak_mu,
            peak_var=w * cell.peak_var + (1 - w) * pool.peak_var,
            mae_mu=w * cell.mae_mu + (1 - w) * pool.mae_mu,
            mae_var=w * cell.mae_var + (1 - w) * pool.mae_var,
            ttp_mu=w * cell.ttp_mu + (1 - w) * pool.ttp_mu,
            ttp_var=w * cell.ttp_var + (1 - w) * pool.ttp_var,
            cap_alpha=w * cell.cap_alpha + (1 - w) * pool.cap_alpha,
            cap_beta=w * cell.cap_beta + (1 - w) * pool.cap_beta,
        )
        return out

    def keys(self) -> List[Tuple[int, str]]:
        return list(self.cells.keys())


# ─── Posterior updaters ─────────────────────────────────────────────────

def _update_normal(post: CellPosterior, actual_attr: str, prior_attr_mu: str,
                          prior_attr_var: str, value: float, noise_var: float):
    """Normal-Normal conjugate update with known noise variance.

    posterior precision = prior precision + 1/noise_var
    posterior mean      = (prior_mu/prior_var + value/noise_var) / posterior precision
    """
    prior_mu = getattr(post, prior_attr_mu)
    prior_var = getattr(post, prior_attr_var)
    prior_prec = 1.0 / max(prior_var, 1e-9)
    obs_prec = 1.0 / max(noise_var, 1e-9)
    new_prec = prior_prec + obs_prec
    new_mu = (prior_mu * prior_prec + value * obs_prec) / new_prec
    new_var = 1.0 / new_prec
    setattr(post, prior_attr_mu, new_mu)
    setattr(post, prior_attr_var, new_var)


def _update_beta(post: CellPosterior, alpha_attr: str, beta_attr: str,
                       success: bool):
    if success:
        setattr(post, alpha_attr, getattr(post, alpha_attr) + 1.0)
    else:
        setattr(post, beta_attr, getattr(post, beta_attr) + 1.0)


def _update_cell_with_label(post: CellPosterior, lbl: RegretLabel,
                                    noise_var_pnl: float = 50.0**2,
                                    noise_var_ttp: float = 600.0**2):
    """Apply one regret label to a CellPosterior (in-place)."""
    post.n += 1
    # WR — count-based
    _update_beta(post, 'wr_alpha', 'wr_beta', lbl.actual_pnl > 0)
    # EV
    _update_normal(post, 'ev_mu', 'ev_mu', 'ev_var',
                          lbl.actual_pnl, noise_var_pnl)
    # Peak
    _update_normal(post, 'peak_mu', 'peak_mu', 'peak_var',
                          lbl.peak_pnl, noise_var_pnl)
    # MAE
    _update_normal(post, 'mae_mu', 'mae_mu', 'mae_var',
                          lbl.mae_pnl, noise_var_pnl)
    # TtP
    _update_normal(post, 'ttp_mu', 'ttp_mu', 'ttp_var',
                          lbl.time_to_peak_s, noise_var_ttp)
    # Capture (clipped to [0, 1])
    cap = lbl.capture_ratio
    if cap == cap:  # not NaN
        cap = float(np.clip(cap, 0.0, 1.0))
        post.cap_alpha += cap
        post.cap_beta += (1.0 - cap)


def build(labels: Iterable[RegretLabel]) -> BayesianTable:
    """Build a hierarchical table from a stream of regret labels."""
    table = BayesianTable()
    labels = list(labels)

    for lbl in labels:
        cell = table.get(lbl.entry_regime_idx, lbl.entry_tier)
        _update_cell_with_label(cell, lbl)

        # Tier pool
        if lbl.entry_tier not in table.tier_pools:
            table.tier_pools[lbl.entry_tier] = CellPosterior()
        _update_cell_with_label(table.tier_pools[lbl.entry_tier], lbl)

        # Universal
        _update_cell_with_label(table.universal, lbl)

    return table


# ─── I/O ────────────────────────────────────────────────────────────────

def save(table: BayesianTable, path: str) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(table, f)


def load(path: str) -> BayesianTable:
    with open(path, 'rb') as f:
        return pickle.load(f)


def to_json(table: BayesianTable) -> dict:
    """Serialize as JSON-friendly dict for inspection."""
    out = {
        'universal': table.universal.to_dict(),
        'tier_pools': {k: v.to_dict() for k, v in table.tier_pools.items()},
        'cells': {f'{r}|{t}': p.to_dict()
                       for (r, t), p in table.cells.items()},
    }
    return out


# ─── CLI ────────────────────────────────────────────────────────────────

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description='Hierarchical Bayesian table builder')
    p.add_argument('--regret', type=str,
                       default='training_iso_v2/output/regret_labels.pkl')
    p.add_argument('--out', type=str,
                       default='training_iso_v2/output/bayesian_table.pkl')
    return p.parse_args()


def main():
    import argparse
    args = _parse_args()
    with open(args.regret, 'rb') as f:
        labels = pickle.load(f)
    print(f'Loaded {len(labels)} regret labels from {args.regret}')

    table = build(labels)
    save(table, args.out)
    print(f'Saved table -> {args.out}')

    json_path = args.out.replace('.pkl', '.json')
    with open(json_path, 'w') as f:
        json.dump(to_json(table), f, indent=2)
    print(f'Saved inspectable JSON -> {json_path}')

    print(f'\nCell summary (n, WR, EV, peak, MAE, TtP_s, capture):')
    keys = sorted(table.keys())
    print(f'  Total cells: {len(keys)}')
    print(f'  {"regime":>2}  {"tier":<18}  {"n":>5}  {"WR":>6}  '
              f'{"EV":>8}  {"peak":>8}  {"MAE":>8}  {"TtP_s":>7}  {"cap":>5}')
    for r, t in keys:
        c = table.cells[(r, t)]
        print(f'  {r:>2}  {t:<18}  {c.n:>5}  {c.wr_mean():>6.1%}  '
                  f'${c.ev_mu:>+7.2f}  ${c.peak_mu:>+7.2f}  ${c.mae_mu:>+7.2f}  '
                  f'{c.ttp_mu:>7.0f}  {c.cap_mean():>5.2f}')


if __name__ == '__main__':
    main()
