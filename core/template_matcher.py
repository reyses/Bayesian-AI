"""
Template Matcher — K-Means clustering with CNN-ready interface.

Clusters 60D grounded features into templates. Each template gets
its own config (SL, TP, direction, hold time). Trained on IS with
full lookahead, validated on OOS bar-by-bar.

Interface (same for K-Means and future CNN):
  - fit(features, outcomes) -> builds templates from IS data
  - match(feature_vector) -> (template_id, distance, confidence)
  - get_config(template_id) -> TemplateConfig
  - save(path) / load(path)

Trade marker logger: every decision logged for inspection.
"""
import json
import os
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict


@dataclass
class TemplateConfig:
    """Per-template trading configuration."""
    template_id: int = 0
    direction: str = ''          # LONG, SHORT, or BOTH
    confidence: float = 0.0      # 0-1 how reliable this template is
    n_samples: int = 0           # how many bars clustered here
    win_rate: float = 0.0        # IS win rate
    avg_pnl_ticks: float = 0.0   # IS average PnL
    sl_ticks: float = 40.0       # stop loss
    tp_ticks: float = 10.0       # take profit (repeating)
    hold_bars: int = 11          # expected hold time
    long_wr: float = 0.0         # IS LONG win rate
    short_wr: float = 0.0        # IS SHORT win rate
    long_pnl: float = 0.0        # IS avg LONG PnL
    short_pnl: float = 0.0       # IS avg SHORT PnL
    active: bool = True          # can be disabled


@dataclass
class MatchResult:
    """Result of matching a feature vector against templates."""
    template_id: int = -1
    distance: float = float('inf')
    confidence: float = 0.0
    direction: str = ''
    config: Optional[TemplateConfig] = None
    reason: str = ''


@dataclass
class TradeMarker:
    """Logged for every trading decision — full audit trail."""
    bar_index: int = 0
    timestamp: float = 0.0
    price: float = 0.0
    template_id: int = -1
    distance: float = 0.0
    direction: str = ''
    action: str = ''            # ENTER, EXIT, HOLD, SKIP
    features: list = field(default_factory=list)  # 60D snapshot
    reason: str = ''
    pnl_ticks: float = 0.0     # filled after trade completes


class TemplateMatcher:
    """K-Means template matcher with CNN-ready interface."""

    def __init__(self, n_templates: int = 400, max_distance: float = 50.0):
        self.n_templates = n_templates
        self.max_distance = max_distance  # reject matches beyond this

        # Fitted state
        self.centroids: Optional[np.ndarray] = None  # (n_templates, 60)
        self.configs: Dict[int, TemplateConfig] = {}
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        self._fitted = False

        # Trade marker log
        self.markers: List[TradeMarker] = []

    def fit(self, features: np.ndarray, outcomes: dict = None):
        """
        Fit K-Means on IS feature data.

        Args:
            features: (N, 60) array of grounded features
            outcomes: optional dict with {bar_index: {direction, pnl_ticks, hold_bars}}
                      for labeling templates with lookahead outcomes
        """
        from sklearn.cluster import MiniBatchKMeans

        N, D = features.shape
        print(f"Fitting {self.n_templates} templates on {N:,} bars x {D}D...")

        # Normalize features (z-score)
        self.scaler_mean = features.mean(axis=0)
        self.scaler_std = features.std(axis=0)
        self.scaler_std[self.scaler_std < 1e-8] = 1.0
        X = (features - self.scaler_mean) / self.scaler_std

        # K-Means clustering
        kmeans = MiniBatchKMeans(
            n_clusters=self.n_templates,
            batch_size=min(10000, N),
            random_state=42,
            n_init=3,
        )
        labels = kmeans.fit_predict(X)
        self.centroids = kmeans.cluster_centers_

        # Build per-template configs
        self.configs = {}
        for tid in range(self.n_templates):
            mask = labels == tid
            n_samples = int(mask.sum())

            cfg = TemplateConfig(
                template_id=tid,
                n_samples=n_samples,
            )

            # Label with outcomes if provided (lookahead training)
            if outcomes:
                tid_outcomes = [outcomes[i] for i in np.where(mask)[0]
                                if i in outcomes]
                if tid_outcomes:
                    pnls = [o['pnl_ticks'] for o in tid_outcomes]
                    dirs = [o['direction'] for o in tid_outcomes]
                    cfg.win_rate = sum(1 for p in pnls if p > 0) / len(pnls)
                    cfg.avg_pnl_ticks = np.mean(pnls)

                    # Direction breakdown
                    long_pnls = [o['pnl_ticks'] for o in tid_outcomes if o['direction'] == 'LONG']
                    short_pnls = [o['pnl_ticks'] for o in tid_outcomes if o['direction'] == 'SHORT']

                    if long_pnls:
                        cfg.long_wr = sum(1 for p in long_pnls if p > 0) / len(long_pnls)
                        cfg.long_pnl = np.mean(long_pnls)
                    if short_pnls:
                        cfg.short_wr = sum(1 for p in short_pnls if p > 0) / len(short_pnls)
                        cfg.short_pnl = np.mean(short_pnls)

                    # Best direction
                    if cfg.long_pnl > cfg.short_pnl and cfg.long_pnl > 0:
                        cfg.direction = 'LONG'
                    elif cfg.short_pnl > cfg.long_pnl and cfg.short_pnl > 0:
                        cfg.direction = 'SHORT'
                    else:
                        cfg.direction = 'BOTH'

                    # Confidence = how much better than random
                    cfg.confidence = min(1.0, max(0.0, (cfg.win_rate - 0.5) * 2))

                    # Optimal hold from outcomes
                    holds = [o.get('hold_bars', 11) for o in tid_outcomes]
                    cfg.hold_bars = int(np.median(holds))

                    # SL/TP from outcome distribution
                    if pnls:
                        losses = [abs(p) for p in pnls if p < 0]
                        wins = [p for p in pnls if p > 0]
                        if losses:
                            cfg.sl_ticks = float(np.percentile(losses, 75))
                        if wins:
                            cfg.tp_ticks = float(min(10, np.percentile(wins, 25)))

            self.configs[tid] = cfg

        self._fitted = True

        # Stats
        active = sum(1 for c in self.configs.values() if c.n_samples > 10)
        profitable = sum(1 for c in self.configs.values() if c.avg_pnl_ticks > 0)
        print(f"Templates: {self.n_templates} total, {active} with >10 samples, "
              f"{profitable} profitable")

    def match(self, feature_vector: np.ndarray, bar_index: int = 0,
              timestamp: float = 0.0, price: float = 0.0) -> MatchResult:
        """
        Match a 60D feature vector against templates.

        Returns MatchResult with template_id, distance, direction.
        """
        if not self._fitted or self.centroids is None:
            return MatchResult(reason='not fitted')

        # Normalize with fitted scaler
        x = (feature_vector - self.scaler_mean) / self.scaler_std
        x = x.reshape(1, -1)

        # Compute distances to all centroids
        dists = np.linalg.norm(self.centroids - x, axis=1)
        best_tid = int(np.argmin(dists))
        best_dist = float(dists[best_tid])

        # Reject if too far
        if best_dist > self.max_distance:
            result = MatchResult(
                template_id=best_tid,
                distance=best_dist,
                reason=f'distance {best_dist:.1f} > {self.max_distance}',
            )
            self._log_marker(bar_index, timestamp, price, result, feature_vector, 'SKIP')
            return result

        cfg = self.configs.get(best_tid, TemplateConfig())

        # Check if template is active and has enough samples
        if not cfg.active or cfg.n_samples < 10:
            result = MatchResult(
                template_id=best_tid,
                distance=best_dist,
                config=cfg,
                reason=f'inactive or thin (n={cfg.n_samples})',
            )
            self._log_marker(bar_index, timestamp, price, result, feature_vector, 'SKIP')
            return result

        result = MatchResult(
            template_id=best_tid,
            distance=best_dist,
            confidence=cfg.confidence,
            direction=cfg.direction,
            config=cfg,
            reason=f'T{best_tid} d={best_dist:.1f} dir={cfg.direction} wr={cfg.win_rate:.0%}',
        )
        self._log_marker(bar_index, timestamp, price, result, feature_vector, 'MATCH')
        return result

    def _log_marker(self, bar_index, timestamp, price, result, features, action):
        """Log a trade marker for inspection."""
        self.markers.append(TradeMarker(
            bar_index=bar_index,
            timestamp=timestamp,
            price=price,
            template_id=result.template_id,
            distance=result.distance,
            direction=result.direction,
            action=action,
            features=features.tolist() if hasattr(features, 'tolist') else list(features),
            reason=result.reason,
        ))

    def save(self, path: str):
        """Save fitted model to directory."""
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, 'centroids.npy'), self.centroids)
        np.save(os.path.join(path, 'scaler_mean.npy'), self.scaler_mean)
        np.save(os.path.join(path, 'scaler_std.npy'), self.scaler_std)

        configs = {str(k): asdict(v) for k, v in self.configs.items()}
        with open(os.path.join(path, 'template_configs.json'), 'w') as f:
            json.dump(configs, f, indent=2)
        print(f"Saved {len(self.configs)} templates to {path}")

    def load(self, path: str):
        """Load fitted model from directory."""
        self.centroids = np.load(os.path.join(path, 'centroids.npy'))
        self.scaler_mean = np.load(os.path.join(path, 'scaler_mean.npy'))
        self.scaler_std = np.load(os.path.join(path, 'scaler_std.npy'))

        with open(os.path.join(path, 'template_configs.json')) as f:
            raw = json.load(f)
        self.configs = {int(k): TemplateConfig(**v) for k, v in raw.items()}
        self.n_templates = len(self.configs)
        self._fitted = True
        print(f"Loaded {self.n_templates} templates from {path}")

    def save_markers(self, path: str):
        """Save trade markers to CSV for inspection."""
        if not self.markers:
            return
        import csv
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'bar_index', 'timestamp', 'price', 'template_id',
                'distance', 'direction', 'action', 'reason', 'pnl_ticks',
            ])
            writer.writeheader()
            for m in self.markers:
                writer.writerow({
                    'bar_index': m.bar_index,
                    'timestamp': m.timestamp,
                    'price': m.price,
                    'template_id': m.template_id,
                    'distance': f'{m.distance:.2f}',
                    'direction': m.direction,
                    'action': m.action,
                    'reason': m.reason,
                    'pnl_ticks': f'{m.pnl_ticks:.1f}',
                })
        print(f"Saved {len(self.markers)} markers to {path}")
