"""CNN inference helpers — wraps a trained V2DirectionCNN as engine hooks.

Two roles:
  - CNNFilter : plugged into Engine.cnn_filter. When a deterministic strategy
                 fires direction D, take iff CNN's predicted direction matches
                 (or softmax_D > take_threshold).
  - CNNEntry  : plugged into Engine.cnn_entry. When NO strategy fires, the
                 CNN may emit its own entry if max-class prob > entry_threshold
                 AND argmax is LONG or SHORT (not FLAT).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch

from training_iso_v2.cnn.model import V2DirectionCNN, v2_to_grid, GRID_H, GRID_W
from training_iso_v2.strategies.base import EntrySignal


# Class indices — must match training labels in dataset.py
CLS_SHORT = 0
CLS_FLAT = 1
CLS_LONG = 2


def load_cnn(model_path: str, device: str = 'cpu') -> dict:
    """Load saved checkpoint. Returns dict with model + metadata."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = V2DirectionCNN()
    state = ckpt.get('model_state', ckpt.get('state_dict', None))
    if state is None:
        raise RuntimeError(f'No model_state in {model_path}')
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return {
        'model': model,
        'device': device,
        'meta': {k: v for k, v in ckpt.items()
                     if k not in ('model_state', 'state_dict')},
    }


def _predict(model: V2DirectionCNN, device: str, v2_vector,
                regime_idx: int) -> np.ndarray:
    """Single-bar inference. Returns softmax probs (3,)."""
    grid, tod = v2_to_grid(v2_vector)
    g = torch.from_numpy(grid).unsqueeze(0).unsqueeze(0).to(device)
    t = torch.tensor([[tod]], dtype=torch.float32, device=device)
    r = torch.tensor([int(regime_idx)], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(g, t, r)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs


class CNNFilter:
    """Filter: gate deterministic strategies. Returns True (take) / False (skip).

    Take iff:
      - CNN's argmax matches the strategy's direction, AND
      - softmax for that direction >= take_threshold.

    Lower take_threshold = more permissive. Default 0.5 = "more likely than not".
    """

    def __init__(self, model_path: str, take_threshold: float = 0.5,
                 device: str = 'cpu'):
        ckpt = load_cnn(model_path, device=device)
        self.model = ckpt['model']
        self.device = device
        self.take_threshold = take_threshold

    def __call__(self, state, signal) -> bool:
        probs = _predict(self.model, self.device, state.v2_vector,
                              state.regime_idx)
        cls = CLS_LONG if signal.direction == 'long' else CLS_SHORT
        return bool(probs[cls] >= self.take_threshold and
                       probs.argmax() == cls)


class CNNEntry:
    """Spawn a CNN-originated entry when no deterministic rule fires.

    Only fires when:
      - CNN's argmax is LONG or SHORT (not FLAT), AND
      - softmax for that class >= entry_threshold.

    entry_threshold should be HIGHER than CNNFilter.take_threshold — we want
    the CNN to be highly confident before generating an entry on its own,
    because there's no deterministic-rule support behind it.
    """

    def __init__(self, model_path: str, entry_threshold: float = 0.65,
                 fire_on: str = '5m', device: str = 'cpu'):
        ckpt = load_cnn(model_path, device=device)
        self.model = ckpt['model']
        self.device = device
        self.entry_threshold = entry_threshold
        self.fire_on = fire_on  # only spawn entries on these bar closes

    def __call__(self, state, ledger) -> Optional[EntrySignal]:
        ready = {
            '1m': state.is_1m_close,
            '5m': state.is_5m_close,
            '15m': state.is_15m_close,
        }.get(self.fire_on, False)
        if not ready:
            return None
        probs = _predict(self.model, self.device, state.v2_vector,
                              state.regime_idx)
        cls = int(probs.argmax())
        if cls == CLS_FLAT:
            return None
        if probs[cls] < self.entry_threshold:
            return None
        direction = 'long' if cls == CLS_LONG else 'short'
        return EntrySignal(direction=direction, tier='CNN_ENTRY',
                              extras={'p_short': float(probs[CLS_SHORT]),
                                          'p_flat': float(probs[CLS_FLAT]),
                                          'p_long': float(probs[CLS_LONG])})
