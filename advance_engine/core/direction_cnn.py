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
        state_dir = 'long' if state[0, 0].item() > 0 else 'short'
        agreement = state_dir == direction
        return direction, confidence, p, state.cpu().numpy()[0], agreement


class TrajectoryPredictor(nn.Module):
    """Shared backbone, state head + multiple P(D) horizon heads.

    Predicts the trajectory decay curve: P(D) at n+1, n+2, ..., n+K.
    Each horizon gets its own direction head. State head grounds the
    prediction in physics.

    Input:  (batch, lookback, n_features)
    Output: state_7d (batch, 7), p_long (batch, K) where K = number of horizons

    The trajectory shape drives all trading decisions:
      - Flat curve = strong trend, hold
      - Steep decay = approaching inflection, prepare to exit
      - Crossing 50% = direction flipping, exit/enter
    """

    def __init__(self, n_features=13, latent_dim=64, n_state=7, horizons=None):
        super().__init__()
        self.horizons = horizons or [1, 2, 3, 4]
        self.n_horizons = len(self.horizons)
        self.backbone = CNNBackbone(n_features=n_features, latent_dim=latent_dim)
        self.latent_dim = latent_dim

        # State head: 7D physics prediction at first horizon
        self.state_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, n_state),
        )

        # One P(D) head per horizon — shared hidden layer, separate outputs
        self.dir_shared = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
        )
        self.dir_outputs = nn.ModuleList([
            nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
            for _ in self.horizons
        ])

        _total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[TrajectoryPredictor] {_total:,} params | "
              f"input: ({n_features}D) | output: 7D state + P(D)x{self.n_horizons} "
              f"horizons={self.horizons}")

    def forward(self, x):
        """Returns (state_7d, p_long_trajectory).

        state_7d: (batch, 7) predicted features at first horizon
        p_long_trajectory: (batch, K) P(long) at each horizon
        """
        latent = self.backbone(x)
        state = self.state_head(latent)

        dir_hidden = self.dir_shared(latent)
        p_longs = torch.cat([head(dir_hidden) for head in self.dir_outputs], dim=-1)

        return state, p_longs

    def forward_with_latent(self, x):
        """Returns (state_7d, p_long_trajectory, latent)."""
        latent = self.backbone(x)
        state = self.state_head(latent)
        dir_hidden = self.dir_shared(latent)
        p_longs = torch.cat([head(dir_hidden) for head in self.dir_outputs], dim=-1)
        return state, p_longs, latent

    def predict_trajectory(self, x):
        """Returns trajectory dict for one sample."""
        with torch.no_grad():
            state, p_longs = self.forward(x)
        p_arr = p_longs.cpu().numpy()[0]
        state_arr = state.cpu().numpy()[0]

        trajectory = {}
        for i, h in enumerate(self.horizons):
            p = float(p_arr[i])
            trajectory[f'n+{h}'] = {
                'p_long': p,
                'confidence': abs(p - 0.5) * 2,
                'direction': 'long' if p > 0.5 else 'short',
            }

        # Sight distance: how many horizons before P(D) drops below chop
        sight = 0
        for i in range(self.n_horizons):
            if abs(p_arr[i] - 0.5) > 0.1:  # outside chop zone
                sight = i + 1
            else:
                break

        return {
            'trajectory': trajectory,
            'state': state_arr,
            'sight_distance': sight,
            'curve': p_arr,
        }
