"""
Shared NN direction-filter module.

Used by:
  - tools/train_pivot_direction_nn.py (training)
  - tools/apply_pivot_nn_filter.py (post-hoc)
  - training_RM_physics/rm_physics_engine.py (live forward-pass filtering)

Exports:
  - feat_to_grid(f91) — reshape 91D to 6×15 grid
  - PivotCNN — the classifier model
  - load_nn_filter(path) — load checkpoint, return (model, mean, std) ready to infer

The checkpoint is a dict with keys: state_dict, mean, std.
"""
import numpy as np
import torch
import torch.nn as nn


# Grid layout: 6 TFs × (12 core + 3 helper) = 6 × 15
N_CORE_PER_TF = 12
N_HELPER_PER_TF = 3
N_TFS = 6
GRID_H = N_TFS
GRID_W = N_CORE_PER_TF + N_HELPER_PER_TF   # 15


def feat_to_grid(f91):
    """Reshape 91D feature vector to 6 × 15 grid. 91st element dropped."""
    f = np.asarray(f91[:90], dtype=np.float32)
    grid = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    core_block = f[:N_CORE_PER_TF * N_TFS]
    helper_block = f[N_CORE_PER_TF * N_TFS:]
    for tf in range(N_TFS):
        grid[tf, :N_CORE_PER_TF] = core_block[tf * N_CORE_PER_TF:(tf + 1) * N_CORE_PER_TF]
        grid[tf, N_CORE_PER_TF:] = helper_block[tf * N_HELPER_PER_TF:(tf + 1) * N_HELPER_PER_TF]
    return grid


class PivotCNN(nn.Module):
    """Small CNN on 6 × 15 grid → P(win logit)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 3 * 7, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.flatten(1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x).squeeze(-1)
        return x


_model_cache = {}    # path → (model, mean, std, device)


def load_nn_filter(model_path: str, device: str = None):
    """Load a trained NN direction model. Cached to avoid re-loading per day.

    Returns (model_in_eval_mode, mean_array, std_array, device_str).
    """
    global _model_cache
    key = (os.path.abspath(model_path), device)
    if key in _model_cache:
        return _model_cache[key]

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = PivotCNN().to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    mean = ckpt['mean']
    std = ckpt['std']
    _model_cache[key] = (model, mean, std, device)
    return model, mean, std, device


def predict_pwin(features_91d, model, mean, std, device):
    """Compute P(win) for a single 91D feature vector. Returns float in [0, 1]."""
    grid = feat_to_grid(features_91d)
    grid_norm = (grid - mean) / std  # shape (1, 6, 15)
    tensor = torch.from_numpy(np.asarray(grid_norm, dtype=np.float32)) \
        .reshape(1, 1, GRID_H, GRID_W).to(device)
    with torch.no_grad():
        logit = model(tensor).item()
    prob = 1.0 / (1.0 + float(np.exp(-logit)))
    return prob


import os  # used in load_nn_filter
