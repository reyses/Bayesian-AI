"""
RetreatPredictor — Layer 3 of the Three-Layer CNN architecture.

Runs every bar while in trade. Decides: should we retreat NOW?

Input: trade context + current market physics (~12D)
Output: P(retreat) — probability we should exit this bar

Trained on IS trade trajectories. Label = 1 if holding from this point
leads to a worse outcome than exiting now. The model learns the physics
of "this trade is done" from actual price action, not fixed thresholds.

~600 params. Runs per-bar so must be lightweight.
"""
import torch
import torch.nn as nn


class RetreatPredictor(nn.Module):
    """Predicts P(retreat) for each bar while in trade.

    Input (12D):
      [0]  unrealized_pnl    — current PnL in ticks (signed)
      [1]  peak_pnl          — best PnL reached so far (ticks)
      [2]  drawdown          — peak_pnl - current_pnl (how much given back)
      [3]  bars_held         — normalized (bars / predicted_hold)
      [4]  pred_dmi_now      — L1's current prediction (does it still agree?)
      [5]  pred_confidence   — abs(pred_dmi) — how sure is L1?
      [6]  side_agreement    — +1 if L1 still agrees with trade dir, -1 if flipped
      [7]  velocity_now      — current bar velocity from features
      [8]  z_se_now          — current z_se from features
      [9]  vol_rel_now       — current relative volume
      [10] bar_range_now     — current bar range (volatility proxy)
      [11] dmi_gap_now       — current dmi gap (trend strength)
    """

    def __init__(self, input_dim=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Linear(12, 1),
            nn.Sigmoid(),
        )

        _total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[RetreatPredictor] {_total:,} trainable params | "
              f"input: {input_dim}D | output: P(retreat)")

    def forward(self, x):
        return self.net(x).squeeze(-1)
