"""
DualHeadPredictor — Predicts both future state AND direction.

Two heads from one backbone:
  Head 1 (state): 7D feature prediction (MSE) — grounded in physics
  Head 2 (direction): P(long) binary (BCE) — grounded in outcome

The state head forces the model to understand the physics.
The direction head forces it to get the binary answer right.
They reinforce each other through the shared backbone.

Each TF gets its own model. The latent vector is saved for merging.
"""
import torch
import torch.nn as nn


class CNNBackbone(nn.Module):
    """Conv1d stack -> latent vector."""

    def __init__(self, n_features=13, latent_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        x = self.relu(self.fc(x))
        return x


class DualHeadPredictor(nn.Module):
    """Shared backbone, two heads: state prediction + direction classification.

    Input:  (batch, lookback, n_features)
    Output: state_7d (batch, 7), p_long (batch,)

    Loss = MSE(state) + lambda * BCE(direction)
    Both heads must agree — predicted state implies a direction,
    and that direction must match the classification head.
    """

    def __init__(self, n_features=13, latent_dim=64, n_state=7):
        super().__init__()
        self.backbone = CNNBackbone(n_features=n_features, latent_dim=latent_dim)
        self.latent_dim = latent_dim

        # Head 1: predict 7D future state (physics)
        self.state_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, n_state),
        )

        # Head 2: predict direction (binary)
        self.dir_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        _total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[DualHeadPredictor] {_total:,} params | "
              f"input: ({n_features}D) | output: 7D state + P(long)")

    def forward(self, x):
        """Returns (state_7d, p_long)."""
        latent = self.backbone(x)
        state = self.state_head(latent)
        p_long = self.dir_head(latent).squeeze(-1)
        return state, p_long

    def forward_with_latent(self, x):
        """Returns (state_7d, p_long, latent) for merging."""
        latent = self.backbone(x)
        state = self.state_head(latent)
        p_long = self.dir_head(latent).squeeze(-1)
        return state, p_long, latent

    def predict(self, x):
        """Returns direction, confidence, state prediction."""
        with torch.no_grad():
            state, p_long = self.forward(x)
        p = p_long.item()
        confidence = abs(p - 0.5) * 2
        direction = 'long' if p > 0.5 else 'short'
        # Cross-check: does state head agree with direction head?
        state_dir = 'long' if state[0, 0].item() > 0 else 'short'  # dmi_diff sign
        agreement = state_dir == direction
        return direction, confidence, p, state.cpu().numpy()[0], agreement
