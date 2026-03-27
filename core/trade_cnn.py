"""
StatePredictor — CNN that predicts future feature states.

Model A (single-TF): Conv1d backbone -> state prediction head
Input: (batch, lookback=10, 13 features)
Output: (batch, 21) = 7 directional features × 3 horizons (t+1, t+5, t+10)

The model predicts WHAT the market state will look like, not WHAT to do.
Trading logic interprets the predicted state into entry/exit decisions.

~15K trainable parameters.
"""
import torch
import torch.nn as nn


class CNNBackbone(nn.Module):
    """Shared feature extractor: Conv1d stack -> latent vector."""

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
        # x: (batch, lookback, n_features) -> (batch, n_features, lookback)
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)  # (batch, 64)
        x = self.relu(self.fc(x))     # (batch, latent_dim)
        return x


class StatePredictor(nn.Module):
    """Predicts future 7D feature states at multiple horizons.

    Architecture: CNNBackbone -> merge -> head(latent_dim -> n_labels)
    """

    def __init__(self, n_features=13, latent_dim=64, n_labels=21):
        super().__init__()
        self.backbone = CNNBackbone(n_features=n_features, latent_dim=latent_dim)
        self.head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, n_labels),
        )

        # Count params
        _total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[StatePredictor] {_total:,} trainable params | "
              f"input: ({n_features}D, lookback) | output: {n_labels}D")

    def forward(self, x):
        latent = self.backbone(x)
        return self.head(latent)

    def predict_state(self, x):
        """Convenience: returns dict of predicted features per horizon."""
        with torch.no_grad():
            pred = self.forward(x)
        # Split into horizons
        result = {}
        from training.train_trade_cnn import HORIZONS, FEATURE_NAMES_7D
        for hi, h in enumerate(HORIZONS):
            start = hi * 7
            result[f't{h}'] = {
                fname: pred[0, start + fi].item()
                for fi, fname in enumerate(FEATURE_NAMES_7D)
            }
        return result
