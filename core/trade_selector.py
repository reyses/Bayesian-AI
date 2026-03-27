"""
DurationPredictor — Layer 2 of the Two-Layer CNN architecture.

Takes Layer 1's prediction vector + regime/session context and outputs:
  1. P(take) — should we take this trade? (binary)
  2. hold_bars — how many bars to hold (regression)

~800 params. Trained AFTER Layer 1, on Layer 1's saved signals + actual outcomes.
The model learns to skip low-quality entries and commit to hold durations,
eliminating trade fragmentation.
"""
import torch
import torch.nn as nn


class DurationPredictor(nn.Module):
    """Predicts take/skip and hold duration for Layer 1 signals.

    Input: Layer 1 prediction (7D) + regime snapshot (4D) + session context (4D) = 15D
    Output: P(take), hold_bars

    For single-horizon models (horizons=[10]), L1 prediction is 7D not 21D.
    """

    def __init__(self, input_dim=15):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.take_head = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self.hold_head = nn.Sequential(
            nn.Linear(16, 1),
            nn.ReLU(),  # hold is always positive
        )

        _total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[DurationPredictor] {_total:,} trainable params | "
              f"input: {input_dim}D | output: P(take) + hold_bars")

    def forward(self, x):
        h = self.shared(x)
        p_take = self.take_head(h).squeeze(-1)
        hold_bars = self.hold_head(h).squeeze(-1)
        return p_take, hold_bars
