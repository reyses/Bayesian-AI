"""
PnL Surface Predictor — predicts what happens at every duration in both directions.

Input:  79D state (10 features x 6 TFs + helpers + time_of_day)
Output: 18 values — the full PnL surface:
  - pnl_long at 8 durations:  [1, 3, 5, 10, 15, 20, 30, 60 bars]
  - pnl_short at 8 durations: [1, 3, 5, 10, 15, 20, 30, 60 bars]
  - max_dd_long:  worst drawdown on the long path (30 bars)
  - max_dd_short: worst drawdown on the short path (30 bars)

The NN doesn't pick a strategy. It predicts the outcome of EVERY option.
The trading decision is trivial: pick the cell with best risk-adjusted PnL.

Post-training: the 64D backbone embedding is clustered to discover strategies.
Each cluster = a strategy with known direction, duration, and exit profile.

Architecture:
  79D -> 256 -> 128 -> 64 (embedding) -> 18 outputs
  ~30K parameters. Trains fast. Runs at 1ms inference.
"""
import torch
import torch.nn as nn
import numpy as np

# Durations we predict PnL for (must match training labels)
DURATIONS = [1, 3, 5, 10, 15, 20, 30, 60]
N_DURATIONS = len(DURATIONS)

# Output layout: [long_pnl x 8, short_pnl x 8, dd_long, dd_short]
N_OUTPUTS = N_DURATIONS * 2 + 2  # 18

# Input dimension (from features_79d)
N_INPUT = 79


class PnLSurfaceNet(nn.Module):
    """Predicts the full PnL surface from a 79D state.

    The backbone produces a 64D embedding that captures the "type" of market state.
    The head maps that to 18 PnL predictions.
    The embedding is extracted post-training for strategy clustering.
    """

    def __init__(self, input_dim=N_INPUT, embed_dim=64, dropout=0.2):
        super().__init__()
        self.embed_dim = embed_dim

        # Backbone: 79D -> 256 -> 128 -> 64
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )

        # PnL head: 64 -> 18
        self.pnl_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, N_OUTPUTS),
        )

    def forward(self, x):
        """Forward pass. Returns dict with pnl_surface and embedding."""
        embedding = self.backbone(x)
        pnl_surface = self.pnl_head(embedding)
        return {
            'pnl_surface': pnl_surface,   # (batch, 18)
            'embedding': embedding,         # (batch, 64) — for clustering
        }

    def predict(self, x):
        """Single-sample prediction with full trade recommendation.

        Args:
            x: (1, 79) tensor

        Returns:
            dict with: direction, duration, expected_pnl, expected_dd,
                       pnl_surface (all 18), embedding (64D),
                       long_pnls, short_pnls, confidence
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(x)
            surface = out['pnl_surface'][0].cpu().numpy()
            embedding = out['embedding'][0].cpu().numpy()

        # Parse surface
        long_pnls = surface[:N_DURATIONS]
        short_pnls = surface[N_DURATIONS:N_DURATIONS * 2]
        dd_long = surface[-2]
        dd_short = surface[-1]

        # Find best option (risk-adjusted)
        best_pnl = -999
        best_dir = 'skip'
        best_dur_idx = 0
        best_dd = 0

        for i, dur in enumerate(DURATIONS):
            # Long option
            risk_adj_long = long_pnls[i] - 0.5 * max(0, -dd_long)
            if risk_adj_long > best_pnl and long_pnls[i] > 0.50:  # must exceed cost
                best_pnl = risk_adj_long
                best_dir = 'long'
                best_dur_idx = i
                best_dd = dd_long

            # Short option
            risk_adj_short = short_pnls[i] - 0.5 * max(0, -dd_short)
            if risk_adj_short > best_pnl and short_pnls[i] > 0.50:
                best_pnl = risk_adj_short
                best_dir = 'short'
                best_dur_idx = i
                best_dd = dd_short

        best_dur = DURATIONS[best_dur_idx]
        expected_pnl = (long_pnls[best_dur_idx] if best_dir == 'long'
                       else short_pnls[best_dur_idx] if best_dir == 'short'
                       else 0.0)

        # Confidence: how much better is the best option vs the second best?
        all_options = list(long_pnls) + list(short_pnls)
        all_options.sort(reverse=True)
        if len(all_options) >= 2 and all_options[0] > 0:
            confidence = 1.0 - (all_options[1] / max(all_options[0], 0.01))
            confidence = max(0.0, min(1.0, confidence))
        else:
            confidence = 0.0

        return {
            'direction': best_dir,
            'duration': best_dur,
            'expected_pnl': float(expected_pnl),
            'expected_dd': float(abs(best_dd)),
            'confidence': confidence,
            'pnl_surface': surface,
            'long_pnls': long_pnls,
            'short_pnls': short_pnls,
            'dd_long': float(dd_long),
            'dd_short': float(dd_short),
            'embedding': embedding,
        }

    def get_embedding(self, x):
        """Extract just the 64D embedding (for clustering)."""
        self.eval()
        with torch.no_grad():
            return self.backbone(x)
