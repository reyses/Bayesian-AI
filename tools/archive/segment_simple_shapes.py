"""Strict simple-shapes-only segmenter — recursive split at internal turns.

Pipeline (per user spec 2026-05-10):
    1. Walk day at 15m CRM, segment into monotonic spans via inflection
    2. For each span, classify against SIMPLE shapes only (no V/U/oscillator)
    3. If best simple-shape fit r < threshold, FIND THE TURN inside the span
       and recursively split. Repeat until every leaf segment is a simple shape.
    4. Same procedure at 5m CRM within each 15m simple-shape segment.

Pure 2D shape (price-time only), no chord features.

Compound shapes (V, U, oscillators, etc.) are NEVER emitted as labels —
they decompose into their constituent monotonic simple-shape legs.

USAGE
    python tools/segment_simple_shapes.py --day 2026_02_12
    python tools/segment_simple_shapes.py --day 2026_02_12 --threshold 0.80
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.research.seeds import SeedPrimitiveLibrary
from tools.segment_day_motif_melody import (
    _load_5s, _load_tf_ohlcv, _ffill_to_5s, TF_WINDOW, PERIOD_S,
    _rolling_r2_adjusted,
)


# Extended TF anchor windows for the 5-level drill-down (15m -> 5m -> 1m -> 15s -> 5s)
# These override the segment_day_motif_melody.TF_WINDOW for THIS tool only.
#   15m: 12 bars (3hr rolling mean) — strategic phrase
#   5m:  9 bars (45min rolling mean) — motif
#   1m:  15 bars (15min rolling mean) — sub_motif
#   15s: 12 bars (3min rolling mean) — measure
#   5s:  6 bars (30s rolling mean)   — note (leaf 5s shape)
TF_WINDOW_EXT = {**TF_WINDOW, '15s': 12, '5s': 6}
PERIOD_S_EXT = {**PERIOD_S, '15s': 15, '5s': 5}


# Hierarchy spec: one entry per LEVEL.
# (level_name, child_attr, tf, min_bars_5s, slope_lookback_5s)
LEVELS = [
    ('phrase',     'phrases',     '15m', 60, 720),
    ('motif',      'motifs',      '5m',  12, 360),
    ('sub_motif',  'sub_motifs',  '1m',   6,  90),
    ('measure',    'measures',    '15s',  4,  30),
    ('note',       'notes',       '5s',   2,  12),
]


SIMPLE_SHAPES = {
    # Straight lines (with steep/gentle modifier)
    'STEEP_LINEAR_UP',  'GENTLE_LINEAR_UP',
    'STEEP_LINEAR_DOWN','GENTLE_LINEAR_DOWN',
    # Curves: 4 concavity-direction combos with steep/gentle modifier
    # Math convention:
    #   CONVEX  = curves like cup (second derivative > 0)
    #   CONCAVE = curves like cap (second derivative < 0)
    'STEEP_CONVEX_UP',   'GENTLE_CONVEX_UP',     # rising, accelerating  (exp-like)
    'STEEP_CONCAVE_UP',  'GENTLE_CONCAVE_UP',    # rising, decelerating  (log-like)
    'STEEP_CONCAVE_DOWN','GENTLE_CONCAVE_DOWN',  # falling, accelerating (-exp-like)
    'STEEP_CONVEX_DOWN', 'GENTLE_CONVEX_DOWN',   # falling, decelerating (-log-like)
    'FLATLINE',
    # NOTE: skew (FRONT_SKEWED/BACK_SKEWED/SYMMETRIC) is a SECONDARY TAG,
    # not a primary label. Stored on each segment alongside the primary shape.
}

# Steep/gentle threshold (absolute slope, pts/min)
# At 15m level, steep MNQ moves are typically >= 0.5 pts/min average over the
# segment. Below that = gentle. Tuned for MNQ futures.
STEEP_THRESHOLD_PTS_PER_MIN = 0.5


class SimpleShapeLibrary:
    """13-primitive library: lines + curves with steep/gentle modifier + FLATLINE.

    Skew (BACK/FRONT_SKEWED) is a secondary tag, not a primary label.
    STEEP/GENTLE prefix is determined by absolute slope (pts/min) AFTER
    the base shape is matched.

    classify() returns:
        (primary_shape: str, pearson_r: float, skew_tag: str)
    """
    BASE_SHAPES = ['LINEAR_UP', 'LINEAR_DOWN',
                    'CONVEX_UP', 'CONCAVE_UP',
                    'CONCAVE_DOWN', 'CONVEX_DOWN']

    def __init__(self, N: int = 16, threshold: float = 0.85,
                  steep_threshold_pts_per_min: float = STEEP_THRESHOLD_PTS_PER_MIN):
        self.N = N
        self.threshold = threshold
        self.steep_threshold = steep_threshold_pts_per_min
        self.templates = self._build_templates(N)

    def _build_templates(self, N: int) -> dict:
        x = np.linspace(0, 1, N)
        templates = {
            'LINEAR_UP':    x.copy(),
            'LINEAR_DOWN':  1.0 - x,
            # Convex (cup) rising: accelerating up, exp-like
            'CONVEX_UP':    self._norm01(np.exp(4 * x) - 1),
            # Concave (cap) rising: decelerating up, log-like
            'CONCAVE_UP':   self._norm01(np.log(1 + 10 * x)),
            # Concave (cap) falling: accelerating down, -exp-like
            'CONCAVE_DOWN': 1.0 - self._norm01(np.exp(4 * x) - 1),
            # Convex (cup) falling: decelerating down, -log-like
            'CONVEX_DOWN':  1.0 - self._norm01(np.log(1 + 10 * x)),
        }
        return templates

    @staticmethod
    def _norm01(arr: np.ndarray) -> np.ndarray:
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    def _detect_skew(self, resampled: np.ndarray) -> str:
        """Locate the bar with maximum |second-derivative|.
        Front-third → FRONT_SKEWED, back-third → BACK_SKEWED, else SYMMETRIC."""
        if len(resampled) < 5:
            return 'SYMMETRIC'
        d2 = np.abs(np.diff(np.diff(resampled)))
        if len(d2) == 0 or d2.max() < 1e-9:
            return 'SYMMETRIC'
        peak_idx = int(np.argmax(d2))
        third = max(1, len(d2) // 3)
        if peak_idx < third:
            return 'FRONT_SKEWED'
        if peak_idx >= 2 * third:
            return 'BACK_SKEWED'
        return 'SYMMETRIC'

    def _apply_steep_modifier(self, base_shape: str,
                                slope_pts_per_min: float) -> str:
        """Prefix STEEP_ or GENTLE_ to the base shape based on |slope|."""
        if base_shape == 'FLATLINE' or base_shape == 'NOISE':
            return base_shape
        prefix = ('STEEP_' if abs(slope_pts_per_min) >= self.steep_threshold
                   else 'GENTLE_')
        return prefix + base_shape

    def classify(self, segment: np.ndarray,
                  slope_pts_per_min: float = None) -> tuple[str, float, str]:
        """Returns (final_shape_with_modifier, pearson_r, skew_tag).
        skew_tag is 'SYMMETRIC' / 'FRONT_SKEWED' / 'BACK_SKEWED' / 'NONE' (lines).
        """
        n = len(segment)
        if n < 4:
            return 'NOISE', 0.0, 'NONE'
        src_x = np.linspace(0, 1, n)
        tgt_x = np.linspace(0, 1, self.N)
        resampled = np.interp(tgt_x, src_x, segment)
        if resampled.std() < 1e-9:
            return 'FLATLINE', 0.0, 'NONE'

        best, best_r = 'NOISE', -2.0
        for name, tpl in self.templates.items():
            if tpl.std() < 1e-9:
                continue
            r = float(np.corrcoef(resampled, tpl)[0, 1])
            if r > best_r:
                best, best_r = name, r

        if best_r < self.threshold:
            return 'NOISE', best_r, 'NONE'

        # Skew detection (only for curves, not lines)
        if best in ('LINEAR_UP', 'LINEAR_DOWN'):
            skew = 'NONE'
        else:
            skew = self._detect_skew(resampled)

        # Apply STEEP/GENTLE modifier
        if slope_pts_per_min is not None and np.isfinite(slope_pts_per_min):
            final = self._apply_steep_modifier(best, slope_pts_per_min)
        else:
            final = best  # no slope info; emit base shape unmodified

        return final, best_r, skew


def _find_turn(span: np.ndarray) -> int:
    """Find the V/U turn within span (TF M_close is a step-function at 5s
    sampling — diffs between bars are mostly 0, so we walk the unique step
    values).

    Returns local index (0-based within span)."""
    n = len(span)
    if n < 3:
        return n // 2
    # Indices where the step-function value changes
    change_mask = np.concatenate([[False], span[1:] != span[:-1]])
    change_idx = np.where(change_mask)[0]
    if len(change_idx) < 2:
        return n // 2  # no internal step change; split at midpoint
    step_vals = span[change_idx]
    step_diffs = np.diff(step_vals)
    # First sign flip between consecutive steps
    for i in range(1, len(step_diffs)):
        if step_diffs[i] * step_diffs[i - 1] < 0:
            return int(change_idx[i])
    # No sign flip in step values; split at extremum
    if span.max() - span[0] > span[0] - span.min():
        return int(np.argmax(span))
    else:
        return int(np.argmin(span))


def _magnitude_gate(span: np.ndarray, sigma: float, k: float = 1.0,
                     r2_threshold: float = 0.50) -> bool:
    """True if span is FLAT — small magnitude AND no linear trend.

    Dual criterion to avoid mis-labeling small-but-directional moves as FLATLINE:
        range < k * sigma                 (small magnitude relative to noise)
        AND r2adj_of_linear_fit < r2_threshold  (no clean linear trend)

    A span with small range BUT high R² is a small LINEAR move, not FLATLINE.
    A span with small range AND low R² is genuinely flat noise.
    """
    if not np.isfinite(sigma) or sigma <= 0:
        return False
    finite = span[np.isfinite(span)]
    if len(finite) < 2:
        return True
    seg_range = float(finite.max() - finite.min())
    if seg_range >= k * sigma:
        return False
    # Small magnitude — check R² of linear fit
    r2 = _rolling_r2_adjusted(finite)
    if not np.isfinite(r2):
        return True
    return r2 < r2_threshold


def _split_into_monotonic_runs(M_5s: np.ndarray, start_i: int, end_i: int,
                                min_bars: int,
                                slope_lookback_5s_bars: int = 720) -> list[tuple[int, int]]:
    """Split [start_i, end_i] at sign-flips of the LONGER-WINDOW slope.

    Bar-to-bar sign flips of the 15m M_close are too noisy — a gradual
    upward rise with tiny dips would get fragmented into many pieces.
    Using a longer lookback (default 1h = 720 5s bars) smooths over those
    micro wiggles and only splits where the OVERALL direction reverses.

    Returns list of (start_i, end_i) for each monotonic-by-lookback-slope run.
    """
    n_total = end_i - start_i + 1
    if n_total < min_bars:
        return []
    if n_total <= slope_lookback_5s_bars:
        # Span shorter than lookback — emit as one run (no internal slope to compute)
        return [(start_i, end_i)]

    span = M_5s[start_i:end_i + 1]
    # Lookback slope: span[i] - span[i - W]
    W = slope_lookback_5s_bars
    slope = np.zeros(len(span))
    slope[W:] = span[W:] - span[:-W]
    sign = np.sign(slope)

    # Sign flips of the longer-window slope
    flip_local = []
    for i in range(W + 1, len(span)):
        if sign[i] != 0 and sign[i - 1] != 0 and sign[i] != sign[i - 1]:
            flip_local.append(i)

    boundaries = [0] + flip_local + [len(span) - 1]
    runs = []
    for i in range(len(boundaries) - 1):
        s = start_i + boundaries[i]
        e = start_i + boundaries[i + 1]
        if (e - s) >= min_bars:
            runs.append((s, e))
    return runs


def _segment_simple(M_5s: np.ndarray, S_5s: np.ndarray,
                    start_i: int, end_i: int,
                    lib: SimpleShapeLibrary, min_bars: int = 12,
                    sigma_gate: float = 1.0,
                    slope_lookback_5s_bars: int = 720) -> list[dict]:
    """Split into monotonic runs (mandatory) then classify each as a
    simple shape. Runs that don't match a simple shape above threshold
    are emitted as NOISE — they're monotonic but not a clean curve fit.
    """
    runs = _split_into_monotonic_runs(M_5s, start_i, end_i, min_bars,
                                       slope_lookback_5s_bars=slope_lookback_5s_bars)
    if not runs:
        return []

    out = []
    for r_start, r_end in runs:
        span = M_5s[r_start:r_end + 1]
        seg_S = S_5s[r_start:r_end + 1]
        mean_sigma = float(np.nanmean(seg_S)) if seg_S.size else float('nan')

        # Compute slope for the STEEP/GENTLE modifier
        length_min = max((r_end - r_start) * 5 / 60.0, 1e-6)
        slope = (span[-1] - span[0]) / length_min if length_min > 0 else 0.0

        # Classify shape FIRST. If a simple shape matches strongly, use it
        # (including GENTLE_ prefix for small-magnitude). Only fall back to
        # FLATLINE if no shape matched AND magnitude is genuinely tiny —
        # otherwise emit NOISE (monotonic but unclassifiable).
        label, r_val, skew = lib.classify(span, slope_pts_per_min=slope)

        if label == 'NOISE':
            # Fallback: small magnitude + low R^2 = FLATLINE; else stays NOISE
            if _magnitude_gate(span, mean_sigma, k=sigma_gate):
                label = 'FLATLINE'
                skew = 'NONE'
                r_val = 0.0

        out.append({'start_i': r_start, 'end_i': r_end,
                    'shape': label, 'r': r_val,
                    'skew': skew,
                    'mean_sigma': mean_sigma})
    return out


LINE_SHAPES_SET = {
    'STEEP_LINEAR_UP', 'GENTLE_LINEAR_UP',
    'STEEP_LINEAR_DOWN', 'GENTLE_LINEAR_DOWN',
    'FLATLINE',
}


def _sandwich_merge_lines(segments: list[dict], M_5s: np.ndarray,
                           S_5s: np.ndarray, lib,
                           max_sandwich_5s_bars: int = 180) -> list[dict]:
    """For LINES only: if a short non-line phrase sits between two same-line
    phrases, absorb the middle into the extended line.

    Lines are simpler structurally — a brief NOISE/transition between two
    GENTLE_LINEAR_UP phrases is more naturally one extended GENTLE_LINEAR_UP
    than three separate phrases.

    max_sandwich_5s_bars=180 = 15 minutes (one 15m bar of sandwich tolerance).
    """
    if len(segments) < 3:
        return segments
    out = []
    i = 0
    while i < len(segments):
        cur = segments[i].copy()
        # Greedy extend: absorb same-shape (direct) or short-NON-line (sandwich) + next-same
        while True:
            extended = False
            if cur['shape'] not in LINE_SHAPES_SET:
                break
            j = len(out) + (i + 1) if False else i + 1
            # 1) Direct adjacency: merge with next if same shape
            if i + 1 < len(segments) and segments[i + 1]['shape'] == cur['shape']:
                cur['end_i'] = segments[i + 1]['end_i']
                i += 1
                extended = True
                continue
            # 2) Sandwich: skip short non-line if next-next is same line
            if (i + 2 < len(segments)
                    and segments[i + 1]['shape'] not in LINE_SHAPES_SET
                    and segments[i + 2]['shape'] == cur['shape']):
                middle = segments[i + 1]
                middle_bars = middle['end_i'] - middle['start_i']
                if middle_bars <= max_sandwich_5s_bars:
                    cur['end_i'] = segments[i + 2]['end_i']
                    i += 2
                    extended = True
                    continue
            break
        # Re-classify cur on its (possibly extended) span
        new_span = M_5s[cur['start_i']:cur['end_i'] + 1]
        new_S = S_5s[cur['start_i']:cur['end_i'] + 1]
        cur['mean_sigma'] = float(np.nanmean(new_S)) if new_S.size else float('nan')
        length_min = max((cur['end_i'] - cur['start_i']) * 5 / 60.0, 1e-6)
        slope = (new_span[-1] - new_span[0]) / length_min if length_min > 0 else 0.0
        label, r_val, skew = lib.classify(new_span, slope_pts_per_min=slope)
        if label == cur['shape']:
            cur['r'] = r_val; cur['skew'] = skew
        else:
            cur['r'] = r_val; cur['skew'] = skew  # keep original shape but update r
        out.append(cur)
        i += 1
    return out


def _merge_adjacent_same_shape(segments: list[dict], M_5s: np.ndarray,
                                 S_5s: np.ndarray,
                                 lib: SimpleShapeLibrary) -> list[dict]:
    """Collapse runs of adjacent segments that share the same shape_class.

    A series of contiguous FLATLINE-FLATLINE-FLATLINE phrases is structurally
    one continuous FLATLINE — the 1h-slope sign flips between them are micro
    wiggles, not real direction changes.

    After merging, re-fit the shape on the merged span. If the merged span no
    longer fits any simple shape above threshold, keep the original shape
    (the merge is still valid; the fit just got noisier).
    """
    if len(segments) <= 1:
        return segments

    merged = [segments[0].copy()]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg['shape'] == prev['shape']:
            new_start = prev['start_i']
            new_end = seg['end_i']
            new_span = M_5s[new_start:new_end + 1]
            new_S = S_5s[new_start:new_end + 1]
            new_mean_sigma = float(np.nanmean(new_S)) if new_S.size else float('nan')
            length_min = max((new_end - new_start) * 5 / 60.0, 1e-6)
            slope = (new_span[-1] - new_span[0]) / length_min
            label, r_val, skew = lib.classify(new_span, slope_pts_per_min=slope)
            final_shape = label if label == prev['shape'] else prev['shape']
            prev['start_i'] = new_start
            prev['end_i'] = new_end
            prev['shape'] = final_shape
            prev['r'] = r_val
            prev['skew'] = skew
            prev['mean_sigma'] = new_mean_sigma
        else:
            merged.append(seg.copy())
    return merged


def _prepare_tf_anchor(tf: str, day: str, ts_5s: np.ndarray):
    oh = _load_tf_ohlcv(tf, day)
    if oh.empty:
        return None
    N = TF_WINDOW_EXT[tf]
    oh['M'] = oh['close'].rolling(N, min_periods=2).mean()
    oh['S'] = oh['close'].rolling(N, min_periods=2).std()
    tf_ts = oh['timestamp'].values.astype(np.int64)
    M = _ffill_to_5s(oh['M'].values, tf_ts, ts_5s, PERIOD_S_EXT[tf])
    S = _ffill_to_5s(oh['S'].values, tf_ts, ts_5s, PERIOD_S_EXT[tf])
    return M, S


def _split_flatlines_by_5m_substructure(phrases: list[dict], M_5m: np.ndarray,
                                         S_5m: np.ndarray, ts_5s: np.ndarray,
                                         lib, min_bars_inner: int = 60,
                                         flat_5m_lookback_5s: int = 360) -> list[dict]:
    """For each FLATLINE phrase, check if 5m M_close has INTERNAL direction
    reversals within the span. If yes, split at those reversals — the 15m
    line is too smoothed to see sub-1h V/U shapes that the 5m line reveals.

    flat_5m_lookback_5s = 360 = 30min slope lookback on 5m line
        (shorter than the 1h lookback used at 15m level — captures 5m structure)
    """
    out = []
    for p in phrases:
        if p['shape'] != 'FLATLINE':
            out.append(p)
            continue
        # Look at 5m line within this span
        sub_runs = _split_into_monotonic_runs(
            M_5m, p['start_i'], p['end_i'], min_bars_inner,
            slope_lookback_5s_bars=flat_5m_lookback_5s)
        if len(sub_runs) <= 1:
            out.append(p)
            continue
        # 5m line has internal direction reversals; reclassify each sub-span
        new_phrases = []
        for rs, re in sub_runs:
            sub_span = M_5m[rs:re + 1]
            sub_S = S_5m[rs:re + 1]
            sub_mean_sigma = float(np.nanmean(sub_S)) if sub_S.size else float('nan')
            length_min = max((re - rs) * 5 / 60.0, 1e-6)
            slope = (sub_span[-1] - sub_span[0]) / length_min if length_min > 0 else 0.0
            label, r_val, skew = lib.classify(sub_span, slope_pts_per_min=slope)
            if label == 'NOISE':
                if _magnitude_gate(sub_span, sub_mean_sigma, k=1.0):
                    label = 'FLATLINE'; skew = 'NONE'; r_val = 0.0
            new_phrases.append({
                'start_i': rs, 'end_i': re,
                'shape': label, 'r': r_val,
                'skew': skew, 'mean_sigma': sub_mean_sigma,
            })
        out.extend(new_phrases)
    return out


def _stamp_segment(seg: dict, ts_5s: np.ndarray) -> None:
    seg['start_ts'] = int(ts_5s[seg['start_i']])
    seg['end_ts'] = int(ts_5s[seg['end_i']])
    seg['length_min'] = round((seg['end_ts'] - seg['start_ts']) / 60.0, 2)


def _populate_sub_levels(parents: list[dict], level_idx: int,
                          anchors: dict, lib: SimpleShapeLibrary,
                          ts_5s: np.ndarray, levels: list = None) -> None:
    """Recursively segment each parent into the next-deeper level.
    `levels` defaults to module LEVELS but can be capped by max_depth."""
    levels = levels if levels is not None else LEVELS
    if level_idx >= len(levels):
        return
    _, child_attr, tf, min_bars, lookback = levels[level_idx]
    M_child, S_child = anchors[tf]
    for p in parents:
        children = _segment_simple(M_child, S_child,
                                    p['start_i'], p['end_i'], lib,
                                    min_bars=min_bars,
                                    slope_lookback_5s_bars=lookback)
        children = _merge_adjacent_same_shape(children, M_child, S_child, lib)
        for c in children:
            _stamp_segment(c, ts_5s)
        p[child_attr] = children
        _populate_sub_levels(children, level_idx + 1, anchors, lib, ts_5s, levels)


def segment_day_simple(day: str, threshold: float = 0.85,
                        max_depth: int = 5) -> dict:
    """Segment a day into a 5-level hierarchy:
        phrase (15m) -> motif (5m) -> sub_motif (1m) -> measure (15s) -> note (5s)
    Each level is anchored on its own TF rolling-mean and emits simple-shape
    primitives only. Adjacent same-shape segments are collapsed.
    `max_depth` lets you cap the drill-down (1 = phrases only, 5 = full).
    """
    df_5s = _load_5s(day)
    if df_5s.empty:
        return {}
    ts_5s = df_5s['timestamp'].values.astype(np.int64)

    # Build all anchors up front (one ffill-to-5s per TF)
    anchors = {}
    levels_used = LEVELS[:max_depth]
    for _, _, tf, _, _ in levels_used:
        a = _prepare_tf_anchor(tf, day, ts_5s)
        if a is None:
            return {}
        anchors[tf] = a

    lib = SimpleShapeLibrary(threshold=threshold)
    n = len(ts_5s)

    # Top-level (phrase) over whole day
    _, _, top_tf, top_min_bars, top_lookback = levels_used[0]
    M_top, S_top = anchors[top_tf]
    phrases = _segment_simple(M_top, S_top, 0, n - 1, lib,
                               min_bars=top_min_bars,
                               slope_lookback_5s_bars=top_lookback)
    phrases = _merge_adjacent_same_shape(phrases, M_top, S_top, lib)
    for p in phrases:
        _stamp_segment(p, ts_5s)

    # Drill into all sub-levels (only those within max_depth)
    if max_depth > 1:
        _populate_sub_levels(phrases, 1, anchors, lib, ts_5s, levels=levels_used)

    # Counts at each depth
    def _count(parents: list, attr_chain: list[str]) -> int:
        if not attr_chain:
            return len(parents)
        attr = attr_chain[0]
        return sum(_count(p.get(attr, []), attr_chain[1:]) for p in parents)

    counts = {'n_phrases': len(phrases)}
    chain = []
    for level_idx in range(1, len(levels_used)):
        chain.append(levels_used[level_idx][1])
        plural = f'n_{levels_used[level_idx][0]}s'
        counts[plural] = _count(phrases, chain)

    return {
        'day': day,
        'date_iso': datetime.fromtimestamp(int(ts_5s[0]),
                                            tz=timezone.utc).strftime('%Y-%m-%d'),
        'threshold': threshold,
        'max_depth': max_depth,
        **counts,
        # legacy alias
        'n_motifs': counts.get('n_motifs', 0),
        'phrases': phrases,
    }


def render_chart(day: str, hierarchy: dict, out_path: str):
    df_5s = _load_5s(day)
    if df_5s.empty or not hierarchy:
        return
    ts_5s = df_5s['timestamp'].values.astype(np.int64)
    dt_5s = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in ts_5s]
    close_5s = df_5s['close'].values

    M_15m, _ = _prepare_tf_anchor('15m', day, ts_5s)
    M_5m, _ = _prepare_tf_anchor('5m', day, ts_5s)

    fig, ax = plt.subplots(1, 1, figsize=(22, 9))
    ax.plot(dt_5s, close_5s, color='black', lw=0.5, alpha=0.85, label='5s close')
    if M_15m is not None:
        ax.plot(dt_5s, M_15m, color='#1E88E5', lw=1.6, alpha=0.85,
                label='15m M_close')
    if M_5m is not None:
        ax.plot(dt_5s, M_5m, color='#FB8C00', lw=1.0, alpha=0.85,
                label='5m M_close')

    SHAPE_COLORS = {
        'LINEAR_UP': '#43A047', 'LINEAR_DOWN': '#E53935',
        'EXPONENTIAL_UP': '#7CB342', 'EXPONENTIAL_DOWN': '#EF5350',
        'LOGARITHMIC_UP': '#26A69A', 'LOGARITHMIC_DOWN': '#D81B60',
        'STEP_UP': '#3949AB', 'STEP_DOWN': '#5E35B1',
        'BACK_SKEWED_UP': '#8D6E63', 'BACK_SKEWED_DOWN': '#6D4C41',
        'FRONT_SKEWED_UP': '#7986CB', 'FRONT_SKEWED_DOWN': '#A1887F',
        'FLATLINE': '#FFB300', 'NOISE': '#9C27B0',
    }
    for p_idx, p in enumerate(hierarchy['phrases']):
        s_dt = datetime.fromtimestamp(p['start_ts'], tz=timezone.utc)
        e_dt = datetime.fromtimestamp(p['end_ts'], tz=timezone.utc)
        col = SHAPE_COLORS.get(p['shape'], '#999')
        ax.axvspan(s_dt, e_dt, color=col, alpha=0.10, zorder=0)
        ax.axvline(s_dt, color='#1E88E5', lw=1.4, alpha=0.7)
        ymax = ax.get_ylim()[1]
        tag = f'P{p_idx}: {p["shape"]} r={p["r"]:.2f}\n{p["length_min"]:.0f}m'
        ax.text(s_dt, ymax, tag, fontsize=7, color='#0D47A1',
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          alpha=0.85, edgecolor='#1E88E5', linewidth=0.5))
        # 5m motif boundaries
        for m in p['motifs']:
            ms_dt = datetime.fromtimestamp(m['start_ts'], tz=timezone.utc)
            ax.axvline(ms_dt, color='#FB8C00', lw=0.5, alpha=0.55, linestyle=':')

    if hierarchy['phrases']:
        last_end = datetime.fromtimestamp(hierarchy['phrases'][-1]['end_ts'],
                                            tz=timezone.utc)
        ax.axvline(last_end, color='#1E88E5', lw=1.4, alpha=0.7)

    n_compound_emitted = sum(1 for p in hierarchy['phrases']
                              if p['shape'] == 'NOISE')
    ax.set_title(
        f'{day}  SIMPLE-SHAPES-ONLY segmenter  threshold r={hierarchy["threshold"]}\n'
        f'15m phrases: {hierarchy["n_phrases"]}  '
        f'(of which NOISE residual: {n_compound_emitted})  '
        f'total 5m motifs: {hierarchy["n_motifs"]}\n'
        f'BLUE = phrase boundaries (15m simple-shape segments)  '
        f'ORANGE dotted = motif boundaries (5m simple-shape sub-segments)',
        fontsize=11)
    ax.set_ylabel('price'); ax.set_xlabel('time (UTC)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.20)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', required=True)
    ap.add_argument('--threshold', type=float, default=0.85)
    ap.add_argument('--out-json', default=None)
    ap.add_argument('--out-chart', default=None)
    args = ap.parse_args()

    h = segment_day_simple(args.day, threshold=args.threshold)
    if not h:
        print(f'No data for {args.day}'); sys.exit(1)

    print(f'Day {args.day}  simple-shape segmentation @ r >= {args.threshold}')
    print(f'  {h["n_phrases"]} phrases (15m), {h["n_motifs"]} motifs (5m)')

    # Shape histogram per level
    from collections import Counter
    phrase_shapes = Counter([p['shape'] for p in h['phrases']])
    motif_shapes = Counter()
    for p in h['phrases']:
        motif_shapes.update([m['shape'] for m in p['motifs']])

    print('\n  Phrase (15m) shapes:')
    for sh, n in phrase_shapes.most_common():
        print(f'    {sh:<22s} {n}')
    print('\n  Motif (5m) shapes:')
    for sh, n in motif_shapes.most_common():
        print(f'    {sh:<22s} {n}')

    out_json = args.out_json or f'reports/findings/segments/simple/{args.day}.json'
    out_chart = args.out_chart or f'chart/segments/simple/{args.day}.png'
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(h, f, indent=2)
    print(f'\nJSON  -> {out_json}')
    render_chart(args.day, h, out_chart)
    print(f'Chart -> {out_chart}')


if __name__ == '__main__':
    main()
