"""
Canonical 16D Feature Vector  -- Single Source of Truth
=====================================================
Both FractalClusteringEngine.extract_features() and
TimeframeBeliefNetwork.state_to_features() delegate here.

Vector layout (16D):
  [0]  abs(z_score)
  [1]  log1p(|velocity|)
  [2]  log1p(|momentum|)
  [3]  entropy_normalized
  [4]  log2(tf_seconds)          -- timeframe scale
  [5]  depth                     -- fractal depth
  [6]  parent_is_band_reversal   -- 1.0 if parent type is BAND_REVERSAL
  [7]  adx / 100                -- self regime ADX
  [8]  hurst_exponent            -- self regime Hurst
  [9]  dmi_diff / 100            -- self regime (DMI+ - DMI-)
  [10] parent_z                  -- immediate parent |z|
  [11] parent_dmi_diff           -- immediate parent DMI diff
  [12] root_is_roche             -- 1.0 if root ancestor is BAND_REVERSAL (historical name kept for checkpoint compat)
  [13] tf_alignment              -- sign(self_dmi) * sign(root_dmi)
  [14] term_pid                  -- PID control force
  [15] oscillation_entropy_normalized  -- oscillation coherence
"""

import numpy as np


def extract_feature_vector(
    z_score: float, velocity: float, momentum: float,
    entropy_normalized: float, tf_seconds: int, depth: float,
    parent_is_band_reversal: float,
    adx: float, hurst: float, dmi_diff: float,
    parent_z: float, parent_dmi_diff: float,
    root_is_roche: float, tf_alignment: float,
    pid: float, osc_coherence: float,
) -> list:
    """Canonical 16D feature vector. Single source of truth.

    All inputs are raw values  -- compression (log1p, log2, /100) is applied here.
    Callers must NOT pre-compress.
    """
    v_feat = np.log1p(abs(velocity))
    m_feat = np.log1p(abs(momentum))
    tf_scale = np.log2(max(1, tf_seconds))

    return [
        abs(z_score), v_feat, m_feat, entropy_normalized,
        tf_scale, depth, parent_is_band_reversal,
        adx, hurst, dmi_diff,
        parent_z, parent_dmi_diff, root_is_roche, tf_alignment,
        pid, osc_coherence,
    ]
