"""
Seed Loader — Convert seed JSON into List[PatternEvent] for the training pipeline.

Supports two formats:
  1. Manual seeds: single-day, flat {seeds: [...]} from visual marking
  2. Auto-swing seeds: multi-day, nested {days: {date: {seeds: [...]}}} from ZigZag detection

Both formats produce PatternEvents for Phase 2 clustering:
  - Loads ATLAS data at the execution TF (15s)
  - Computes MarketState via StatisticalFieldEngine
  - Classifies oracle markers from MFE/MAE
  - Returns List[PatternEvent] that plugs directly into clustering

Usage:
    from training.seed_loader import load_seeds_as_manifest, load_auto_swing_as_manifest
    # Manual seeds (single day, ~50 seeds):
    manifest = load_seeds_as_manifest("DATA/regime_seeds/seeds.json", "DATA/ATLAS")
    # Auto-swing seeds (multi-day, ~37K seeds):
    manifest = load_auto_swing_as_manifest("DATA/regime_seeds/auto_swing/auto_seeds_all.json", "DATA/ATLAS")
"""

import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from training.fractal_discovery_agent import PatternEvent, TIMEFRAME_SECONDS
from core.statistical_field_engine import StatisticalFieldEngine
from config.oracle_config import (
    ORACLE_MIN_MOVE_TICKS, ORACLE_HOME_RUN_RATIO, ORACLE_SCALP_RATIO,
    MARKER_MEGA_LONG, MARKER_SCALP_LONG, MARKER_NOISE,
    MARKER_SCALP_SHORT, MARKER_MEGA_SHORT,
)


def _classify_oracle_marker(direction: str, mfe_ticks: float, mae_ticks: float) -> int:
    """Classify a seed into an oracle marker using the standard oracle thresholds."""
    if mfe_ticks < ORACLE_MIN_MOVE_TICKS:
        return MARKER_NOISE

    ratio = mfe_ticks / max(mae_ticks, 0.25)  # avoid div/0

    is_long = direction.upper() == 'LONG'

    if ratio >= ORACLE_HOME_RUN_RATIO:
        return MARKER_MEGA_LONG if is_long else MARKER_MEGA_SHORT
    elif ratio >= ORACLE_SCALP_RATIO:
        return MARKER_SCALP_LONG if is_long else MARKER_SCALP_SHORT
    else:
        return MARKER_NOISE


def _load_atlas_tf(atlas_dir: str, tf: str, month_str: str) -> pd.DataFrame:
    """Load ATLAS data for a specific timeframe and month. month_str like '2025_07'."""
    tf_dir = os.path.join(atlas_dir, tf)
    if not os.path.isdir(tf_dir):
        raise FileNotFoundError(f"No {tf} directory in ATLAS: {tf_dir}")

    parquet_path = os.path.join(tf_dir, f"{month_str}.parquet")
    if not os.path.isfile(parquet_path):
        # Try globbing for any file matching the month
        import glob
        candidates = glob.glob(os.path.join(tf_dir, f"*{month_str}*"))
        if candidates:
            parquet_path = candidates[0]
        else:
            raise FileNotFoundError(f"No parquet for month {month_str} in {tf_dir}")

    df = pd.read_parquet(parquet_path)
    return df


def _detect_month(seeds: list) -> str:
    """Get ATLAS month string from seed timestamps."""
    ts = seeds[0]['ts_start']
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return f"{dt.year}_{dt.month:02d}"


def _detect_date(seeds: list) -> str:
    """Get trading date from seed timestamps."""
    ts = seeds[0]['ts_start']
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime('%Y-%m-%d')


def load_seeds_as_manifest(
    seed_path: str,
    atlas_dir: str,
    tag_filter: Optional[str] = None,
    timeframe: str = '15s',
    depth: int = 8,
) -> List[PatternEvent]:
    """
    Load seed JSON and convert each seed into a PatternEvent.

    Args:
        seed_path: Path to seed JSON file
        atlas_dir: Path to ATLAS root (e.g., DATA/ATLAS)
        tag_filter: Optional — only load seeds with this tag (e.g., 'Swing', 'Scalp')
        timeframe: Timeframe to assign to seed patterns (default '15s' = execution TF,
                   must match forward pass TF for centroid matching to work)
        depth: Fractal depth to assign (default 8 = standard)

    Returns:
        List[PatternEvent] ready for Phase 2 clustering
    """
    # 1. Load seeds (support flat and nested formats)
    with open(seed_path) as f:
        data = json.load(f)

    if 'seeds' in data:
        seeds = data['seeds']
    elif 'days' in data:
        seeds = []
        for day_data in data['days'].values():
            seeds.extend(day_data.get('seeds', []))
    else:
        print(f"[SeedLoader] Unknown seed format -- keys: {list(data.keys())}")
        return []
    if tag_filter:
        seeds = [s for s in seeds if s.get('tag', '') == tag_filter]

    if not seeds:
        print(f"[SeedLoader] No seeds found in {seed_path}"
              f"{f' with tag={tag_filter}' if tag_filter else ''}")
        return []

    print(f"\n{'='*60}")
    print(f"SEED LOADER: {len(seeds)} seeds from {os.path.basename(seed_path)}")
    if tag_filter:
        print(f"  Tag filter: {tag_filter}")
    print(f"{'='*60}")

    # 2. Load ATLAS data at the EXECUTION timeframe (15s) — must match forward pass
    #    Seeds were marked on 1m charts, but we compute MarketState from 15s
    #    so the feature vector (z, vel, mom, regression bands) matches what
    #    the forward pass produces at each 15s bar.
    month_str = _detect_month(seeds)
    date_str = _detect_date(seeds)
    print(f"  Date: {date_str}, Month: {month_str}")
    print(f"  Loading {timeframe} data (execution TF -- matches forward pass)")

    df = _load_atlas_tf(atlas_dir, timeframe, month_str)

    # Filter to the seed date
    if 'timestamp' in df.columns:
        ts_col = 'timestamp'
    elif 'time' in df.columns:
        ts_col = 'time'
    else:
        raise KeyError(f"No timestamp column in {timeframe} data. Columns: {list(df.columns)}")

    # Convert to epoch if needed
    if pd.api.types.is_datetime64_any_dtype(df[ts_col]):
        df['_ts'] = df[ts_col].astype(np.int64) // 10**9
    else:
        df['_ts'] = df[ts_col].astype(float)

    # Filter to the seed day (same calendar date)
    seed_ts_start = min(s['ts_start'] for s in seeds)
    seed_ts_end = max(s['ts_end'] for s in seeds)
    # Add margin: 30 min before first seed, 30 min after last
    day_start = seed_ts_start - 1800
    day_end = seed_ts_end + 1800

    day_mask = (df['_ts'] >= day_start) & (df['_ts'] <= day_end)
    day_data = df[day_mask].copy().reset_index(drop=True)

    if len(day_data) < 30:
        print(f"  WARNING: Only {len(day_data)} bars for seed date -- need >=30 for regression")
        return []

    print(f"  Loaded {len(day_data)} {timeframe} bars for {date_str}")

    # 3. Compute MarketState for all bars
    engine = StatisticalFieldEngine(regression_period=21)
    raw_states = engine.batch_compute_states(day_data)

    # batch_compute_states returns [{'bar_idx': int, 'state': MarketState, ...}]
    # Build index: bar_idx → MarketState
    state_map = {}
    for entry in raw_states:
        if isinstance(entry, dict):
            state_map[entry['bar_idx']] = entry['state']
        else:
            # Fallback if format changes
            state_map[len(state_map)] = entry

    timestamps = day_data['_ts'].values
    prices = day_data['close'].values if 'close' in day_data.columns else day_data['price'].values

    print(f"  Computed {len(state_map)} MarketStates")

    # 4. Match each seed to the nearest bar and build PatternEvent
    manifest = []
    tf_seconds = TIMEFRAME_SECONDS.get(timeframe, 60)
    matched = 0
    skipped = 0

    for i, seed in enumerate(seeds):
        # Handle both formats: trade_id (tagged seeds) and regime_id (raw seeds)
        seed_id = seed.get('trade_id', seed.get('regime_id', i))
        entry_ts = seed['ts_start']

        # Find nearest bar within +-60s
        diffs = np.abs(timestamps - entry_ts)
        best_idx = int(np.argmin(diffs))

        if diffs[best_idx] > 60:
            print(f"  SKIP seed T{seed_id}: no bar within 60s "
                  f"(nearest is {diffs[best_idx]:.0f}s away)")
            skipped += 1
            continue

        state = state_map.get(best_idx, None)
        if state is None:
            print(f"  SKIP seed T{seed_id}: no MarketState (warmup period)")
            skipped += 1
            continue

        # Oracle classification
        marker = _classify_oracle_marker(
            seed['direction'],
            seed.get('mfe_ticks', 0),
            seed.get('mae_ticks', 0.25),
        )

        oracle_meta = {
            'mfe': seed.get('mfe_ticks', 0) * 0.25,  # ticks -> points (MNQ tick = 0.25)
            'mae': seed.get('mae_ticks', 0) * 0.25,
            'mfe_bar': seed.get('n_bars', 0),
            'duration_mins': seed.get('duration_mins', 0),
            'seed_id': seed_id,
            'tag': seed.get('tag', ''),
            'direction': seed['direction'],
        }

        # Determine pattern type from z_score
        z = getattr(state, 'z_score', 0.0)
        pattern_type = 'BAND_REVERSAL' if abs(z) > 1.5 else 'MOMENTUM_BREAK'

        pe = PatternEvent(
            pattern_type=pattern_type,
            timestamp=float(timestamps[best_idx]),
            price=float(prices[best_idx]),
            z_score=getattr(state, 'z_score', 0.0),
            velocity=getattr(state, 'velocity', 0.0),
            momentum=getattr(state, 'momentum_strength', 0.0),
            entropy_normalized=getattr(state, 'entropy_normalized', 0.0),
            file_source=f"seed:{os.path.basename(seed_path)}",
            idx=best_idx,
            state=state,
            timeframe=timeframe,
            depth=depth,
            parent_type='',
            parent_tf='',
            window_data=None,
            parent_chain=[],
            oracle_marker=marker,
            oracle_meta=oracle_meta,
        )
        manifest.append(pe)
        matched += 1

    print(f"\n  Result: {matched} PatternEvents created, {skipped} skipped")

    # Summary table
    if manifest:
        markers = [p.oracle_marker for p in manifest]
        print(f"  Oracle breakdown:")
        for label, val in [('MEGA_LONG', 2), ('SCALP_LONG', 1), ('NOISE', 0),
                           ('SCALP_SHORT', -1), ('MEGA_SHORT', -2)]:
            count = sum(1 for m in markers if m == val)
            if count > 0:
                print(f"    {label:>12s}: {count}")

        types = [p.pattern_type for p in manifest]
        print(f"  Pattern types: BAND_REVERSAL={sum(1 for t in types if t=='BAND_REVERSAL')}, "
              f"MOMENTUM_BREAK={sum(1 for t in types if t=='MOMENTUM_BREAK')}")

    return manifest


def inspect_templates(templates, output_path: Optional[str] = None):
    """
    Generate a detailed template inspection report + editable JSON feedback file.

    For each template shows:
      - Stats (WR, MFE, MAE, direction bias)
      - Every member seed with its direction, PnL, duration
      - Editable fields: action (KEEP/DROP), direction_override (LONG/SHORT/AUTO), notes

    Args:
        templates: List[PatternTemplate] from clustering
        output_path: Optional base path (e.g. 'reports/template_inspection')
                     Produces .txt (human readable) and .json (editable feedback)
    """
    from datetime import datetime as _dt

    # ── Build structured data for each template ──
    template_data = []

    for t in sorted(templates, key=lambda x: x.template_id):
        members = []
        long_count = 0
        short_count = 0

        for p in (t.patterns or []):
            meta = getattr(p, 'oracle_meta', {}) or {}
            direction = meta.get('direction', '?')
            if direction == 'LONG':
                long_count += 1
            elif direction == 'SHORT':
                short_count += 1

            members.append({
                'seed_id': f"T{meta.get('seed_id', '?')}",
                'direction': direction,
                'marker': getattr(p, 'oracle_marker', 0),
                'mfe_ticks': meta.get('mfe', 0) / 0.25 if meta.get('mfe') else 0,
                'mae_ticks': meta.get('mae', 0) / 0.25 if meta.get('mae') else 0,
                'duration_mins': meta.get('duration_mins', 0),
                'tag': meta.get('tag', ''),
                'z_score': round(getattr(p, 'z_score', 0.0), 3),
                'price': round(getattr(p, 'price', 0.0), 2),
            })

        total = long_count + short_count
        if total > 0:
            dominant_dir = 'LONG' if long_count > short_count else 'SHORT'
            dir_pct = max(long_count, short_count) / total
        else:
            dominant_dir = 'UNKNOWN'
            dir_pct = 0.0

        td = {
            'template_id': t.template_id,
            'name': t.semantic_name,
            'member_count': t.member_count,
            'win_rate': round(t.stats_win_rate, 3),
            'mean_mfe_ticks': round(t.mean_mfe_ticks, 1),
            'mean_mae_ticks': round(t.mean_mae_ticks, 1),
            'dominant_direction': dominant_dir,
            'direction_confidence': round(dir_pct, 2),
            'long_count': long_count,
            'short_count': short_count,
            'members': members,
            # ── EDITABLE FIELDS (user fills these in) ──
            'action': 'KEEP',               # KEEP | DROP
            'direction_override': 'AUTO',   # LONG | SHORT | AUTO (use dominant)
            'notes': '',
        }
        template_data.append(td)

    # ── Print human-readable report ──
    lines = []
    lines.append("=" * 95)
    lines.append("TEMPLATE INSPECTION -- Edit the .json file to correct directions & flag templates")
    lines.append("=" * 95)

    for td in template_data:
        lines.append("")
        lines.append(f"  TEMPLATE {td['template_id']}  |  {td['name']}")
        lines.append(f"  {'-' * 70}")
        lines.append(f"  Members: {td['member_count']}  |  WR: {td['win_rate']:.1%}  |  "
                     f"MFE: {td['mean_mfe_ticks']:.1f} ticks  |  MAE: {td['mean_mae_ticks']:.1f} ticks")
        lines.append(f"  Direction: {td['dominant_direction']} ({td['direction_confidence']:.0%})  "
                     f"[{td['long_count']}L / {td['short_count']}S]")
        lines.append(f"  Action: {td['action']}  |  Override: {td['direction_override']}")
        lines.append("")
        lines.append(f"    {'Seed':<6s} {'Dir':<6s} {'Marker':>7s} {'MFE':>7s} {'MAE':>7s} "
                     f"{'Dur':>6s} {'Tag':<7s} {'Z':>7s}  {'Price':>10s}")
        lines.append(f"    {'-' * 72}")

        for m in td['members']:
            marker_name = {2: 'MEGA_L', 1: 'SCALP_L', 0: 'NOISE',
                          -1: 'SCALP_S', -2: 'MEGA_S'}.get(m['marker'], '?')
            lines.append(f"    {m['seed_id']:<6s} {m['direction']:<6s} {marker_name:>7s} "
                         f"{m['mfe_ticks']:>7.0f} {m['mae_ticks']:>7.0f} "
                         f"{m['duration_mins']:>5.1f}m {m['tag']:<7s} "
                         f"{m['z_score']:>7.3f}  {m['price']:>10.2f}")

    lines.append("")
    lines.append("=" * 95)
    lines.append(f"Total: {len(template_data)} templates, "
                 f"{sum(td['member_count'] for td in template_data)} seeds")
    lines.append("")
    lines.append("HOW TO USE:")
    lines.append("  1. Open the .json file alongside this report")
    lines.append("  2. For each template, set:")
    lines.append("     - 'action': 'KEEP' or 'DROP'")
    lines.append("     - 'direction_override': 'LONG', 'SHORT', or 'AUTO'")
    lines.append("     - 'notes': any observations")
    lines.append("  3. Re-run with --seeds --template-feedback reports/template_feedback.json")

    report = "\n".join(lines)
    print(report)

    # ── Save files ──
    if output_path:
        base = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
        txt_path = base + '.txt'
        json_path = base + '_feedback.json'

        os.makedirs(os.path.dirname(txt_path) or '.', exist_ok=True)

        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n  Saved report:   {txt_path}")

        # JSON feedback (the editable file)
        feedback = {
            'created': _dt.now().strftime('%Y-%m-%d %H:%M:%S'),
            'n_templates': len(template_data),
            'templates': template_data,
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(feedback, f, indent=2)
        print(f"  Saved feedback: {json_path}")
        print(f"\n  Edit {json_path} -> change 'action'/'direction_override'/'notes' -> re-run")

    return template_data


def compute_seed_thresholds(
    seed_path: str,
    tag_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract quality thresholds from seed data for filtering discovery manifests.

    Uses P25 (25th percentile) of seed stats as the minimum quality bar.
    Patterns from normal discovery that don't meet these minimums get filtered out.

    Returns:
        Dict with min_duration_mins, min_mfe_ticks, min_range_ticks, n_seeds
        Empty dict if too few seeds.
    """
    with open(seed_path) as f:
        data = json.load(f)

    # Support both flat format {seeds: [...]} and nested format {days: {date: {seeds: [...]}}}
    if 'seeds' in data:
        seeds = data['seeds']
    elif 'days' in data:
        seeds = []
        for day_data in data['days'].values():
            seeds.extend(day_data.get('seeds', []))
    else:
        print(f"[SeedThresholds] Unknown seed format -- keys: {list(data.keys())}")
        return {}
    if tag_filter:
        seeds = [s for s in seeds if s.get('tag', '') == tag_filter]

    if len(seeds) < 3:
        print(f"[SeedThresholds] Only {len(seeds)} seeds -- too few for thresholds")
        return {}

    durations = [s.get('duration_mins', 0) for s in seeds]
    mfe_ticks = [s.get('mfe_ticks', 0) for s in seeds]
    ranges = [abs(s.get('change_ticks', 0)) for s in seeds]

    thresholds = {
        'min_duration_mins': float(np.percentile(durations, 25)),
        'min_mfe_ticks': float(np.percentile(mfe_ticks, 25)),
        'min_range_ticks': float(np.percentile(ranges, 25)),
        'n_seeds': len(seeds),
        'stats': {
            'duration': {'p25': float(np.percentile(durations, 25)),
                         'median': float(np.median(durations)),
                         'p75': float(np.percentile(durations, 75))},
            'mfe_ticks': {'p25': float(np.percentile(mfe_ticks, 25)),
                          'median': float(np.median(mfe_ticks)),
                          'p75': float(np.percentile(mfe_ticks, 75))},
            'range_ticks': {'p25': float(np.percentile(ranges, 25)),
                            'median': float(np.median(ranges)),
                            'p75': float(np.percentile(ranges, 75))},
        },
    }

    print(f"\n{'='*60}")
    print(f"SEED QUALITY THRESHOLDS (from {len(seeds)} seeds)")
    print(f"{'='*60}")
    print(f"  Duration (mins):  P25={thresholds['min_duration_mins']:.1f}  "
          f"med={thresholds['stats']['duration']['median']:.1f}  "
          f"P75={thresholds['stats']['duration']['p75']:.1f}")
    print(f"  MFE (ticks):      P25={thresholds['min_mfe_ticks']:.0f}  "
          f"med={thresholds['stats']['mfe_ticks']['median']:.0f}  "
          f"P75={thresholds['stats']['mfe_ticks']['p75']:.0f}")
    print(f"  Range (ticks):    P25={thresholds['min_range_ticks']:.0f}  "
          f"med={thresholds['stats']['range_ticks']['median']:.0f}  "
          f"P75={thresholds['stats']['range_ticks']['p75']:.0f}")
    print(f"{'='*60}")

    return thresholds


def filter_manifest_by_thresholds(
    manifest: List[PatternEvent],
    thresholds: Dict[str, Any],
    tick_size: float = 0.25,
) -> List[PatternEvent]:
    """
    Filter a discovery manifest using seed-derived quality thresholds.

    For each pattern, checks:
      1. Duration (mfe_bar * tf_seconds / 60) >= min_duration_mins
      2. MFE (oracle_meta['mfe'] / tick_size) >= min_mfe_ticks

    Patterns without oracle_meta are dropped (no quality signal).

    Args:
        manifest: Full discovery manifest (List[PatternEvent])
        thresholds: From compute_seed_thresholds()
        tick_size: MNQ tick size (0.25)

    Returns:
        Filtered List[PatternEvent]
    """
    if not thresholds:
        return manifest

    min_dur = thresholds.get('min_duration_mins', 0)
    min_mfe = thresholds.get('min_mfe_ticks', 0)

    filtered = []
    reasons = {'no_meta': 0, 'short_duration': 0, 'low_mfe': 0}

    for p in manifest:
        meta = getattr(p, 'oracle_meta', None) or {}
        if not meta:
            reasons['no_meta'] += 1
            continue

        # Duration proxy: mfe_bar tells us how long the move took
        tf_secs = TIMEFRAME_SECONDS.get(p.timeframe, 60)
        mfe_bar = meta.get('mfe_bar', 0)
        duration_mins = (mfe_bar * tf_secs) / 60.0

        # MFE in ticks: oracle stores in price points, convert
        mfe_points = meta.get('mfe', 0)
        mfe_ticks = mfe_points / tick_size if tick_size > 0 else 0

        if duration_mins < min_dur:
            reasons['short_duration'] += 1
            continue
        if mfe_ticks < min_mfe:
            reasons['low_mfe'] += 1
            continue

        filtered.append(p)

    kept_pct = len(filtered) / len(manifest) * 100 if manifest else 0
    print(f"\n  [Seed Filter] {len(manifest)} -> {len(filtered)} patterns ({kept_pct:.1f}% kept)")
    print(f"  Thresholds: duration >= {min_dur:.1f}m, MFE >= {min_mfe:.0f} ticks")
    print(f"  Filtered out: {reasons['no_meta']} no oracle, "
          f"{reasons['short_duration']} too short, {reasons['low_mfe']} low MFE")

    return filtered


def load_template_feedback(feedback_path: str) -> Dict[int, Dict]:
    """
    Load user-edited template feedback JSON.

    Returns:
        Dict mapping template_id -> {'action': str, 'direction_override': str, 'notes': str}
    """
    with open(feedback_path) as f:
        data = json.load(f)

    feedback = {}
    for td in data.get('templates', []):
        tid = td['template_id']
        feedback[tid] = {
            'action': td.get('action', 'KEEP').upper(),
            'direction_override': td.get('direction_override', 'AUTO').upper(),
            'notes': td.get('notes', ''),
        }

    kept = sum(1 for v in feedback.values() if v['action'] == 'KEEP')
    dropped = sum(1 for v in feedback.values() if v['action'] == 'DROP')
    overrides = sum(1 for v in feedback.values() if v['direction_override'] != 'AUTO')
    print(f"\n[Feedback] Loaded: {kept} KEEP, {dropped} DROP, {overrides} direction overrides")
    return feedback


def is_auto_swing_format(seed_path: str) -> bool:
    """Detect if a seed file is auto-swing format (multi-day, nested by date)."""
    return 'auto_seeds' in os.path.basename(seed_path)


def load_auto_swing_as_manifest(
    seed_path: str,
    atlas_dir: str,
    timeframe: str = '15s',
    depth: int = 8,
) -> List[PatternEvent]:
    """
    Load auto-swing seeds (multi-day ZigZag format) and convert to PatternEvents.

    Auto-swing files have nested structure:
        {days: {"2025-01-02": {seeds: [...]}, "2025-01-03": {seeds: [...]}, ...}}

    Groups seeds by ATLAS month for efficient parquet loading.
    Computes MarketState from 15s data (matches forward pass TF).

    Args:
        seed_path: Path to auto-swing seed JSON
        atlas_dir: Path to ATLAS root (e.g., DATA/ATLAS)
        timeframe: Execution TF for MarketState (default '15s')
        depth: Fractal depth to assign (default 8)

    Returns:
        List[PatternEvent] ready for Phase 2 clustering
    """
    from tqdm import tqdm
    from collections import defaultdict

    # 1. Load and flatten all seeds with globally unique IDs
    with open(seed_path, encoding='utf-8') as f:
        data = json.load(f)

    if 'days' not in data:
        raise ValueError(f"Not an auto-swing file (no 'days' key): {seed_path}")

    all_seeds = []
    global_id = 0
    for date_str, day_data in sorted(data['days'].items()):
        for seed in day_data.get('seeds', []):
            seed['_date'] = date_str
            seed['_global_id'] = global_id
            global_id += 1
            all_seeds.append(seed)

    if not all_seeds:
        print(f"[AutoSwingLoader] No seeds found in {seed_path}")
        return []

    n_days = len(data['days'])
    print(f"\n{'='*60}")
    print(f"AUTO-SWING SEED LOADER")
    print(f"  Seeds: {len(all_seeds):,} across {n_days} days")
    print(f"  Source: {os.path.basename(seed_path)}")
    print(f"  MarketState TF: {timeframe} (matches forward pass)")
    print(f"{'='*60}")

    # 2. Group seeds by ATLAS month for efficient parquet loading
    month_seeds: Dict[str, List] = defaultdict(list)
    for seed in all_seeds:
        dt = datetime.strptime(seed['_date'], '%Y-%m-%d')
        month_key = f"{dt.year}_{dt.month:02d}"
        month_seeds[month_key].append(seed)

    print(f"  Months: {', '.join(f'{m}({len(s):,})' for m, s in sorted(month_seeds.items()))}")

    # 3. Process each month
    manifest = []
    skipped = 0
    engine = StatisticalFieldEngine(regression_period=21)

    for month_key in tqdm(sorted(month_seeds.keys()), desc="Loading months", unit="mo"):
        seeds_this_month = month_seeds[month_key]

        # Load ATLAS 15s parquet for this month (one load per month)
        try:
            df_month = _load_atlas_tf(atlas_dir, timeframe, month_key)
        except FileNotFoundError:
            print(f"\n  SKIP month {month_key}: no {timeframe} data")
            skipped += len(seeds_this_month)
            continue

        # Prepare timestamp column
        if 'timestamp' in df_month.columns:
            ts_col = 'timestamp'
        elif 'time' in df_month.columns:
            ts_col = 'time'
        else:
            print(f"\n  SKIP month {month_key}: no timestamp column")
            skipped += len(seeds_this_month)
            continue

        if pd.api.types.is_datetime64_any_dtype(df_month[ts_col]):
            df_month['_ts'] = df_month[ts_col].astype(np.int64) // 10**9
        else:
            df_month['_ts'] = df_month[ts_col].astype(float)

        month_timestamps = df_month['_ts'].values

        # Group seeds by date within this month
        date_seeds: Dict[str, List] = defaultdict(list)
        for seed in seeds_this_month:
            date_seeds[seed['_date']].append(seed)

        for date_str in sorted(date_seeds.keys()):
            day_seeds = date_seeds[date_str]

            # Filter ATLAS to this day (+/- 30 min margin)
            ts_min = min(s['ts_start'] for s in day_seeds) - 1800
            ts_max = max(s['ts_end'] for s in day_seeds) + 1800
            day_mask = (month_timestamps >= ts_min) & (month_timestamps <= ts_max)
            day_data = df_month[day_mask].copy().reset_index(drop=True)

            if len(day_data) < 30:
                skipped += len(day_seeds)
                continue

            # Compute MarketState for all bars this day (single CUDA batch)
            raw_states = engine.batch_compute_states(day_data)
            state_map = {}
            for entry in raw_states:
                if isinstance(entry, dict):
                    state_map[entry['bar_idx']] = entry['state']
                else:
                    state_map[len(state_map)] = entry

            day_timestamps = day_data['_ts'].values
            day_prices = (day_data['close'].values if 'close' in day_data.columns
                          else day_data['price'].values)

            # Match each seed to nearest 15s bar
            for seed in day_seeds:
                entry_ts = seed['ts_start']
                diffs = np.abs(day_timestamps - entry_ts)
                best_idx = int(np.argmin(diffs))

                if diffs[best_idx] > 60:
                    skipped += 1
                    continue

                state = state_map.get(best_idx)
                if state is None:
                    skipped += 1
                    continue

                marker = _classify_oracle_marker(
                    seed['direction'],
                    seed.get('mfe_ticks', 0),
                    seed.get('mae_ticks', 0.25),
                )

                oracle_meta = {
                    'mfe': seed.get('mfe_ticks', 0) * 0.25,
                    'mae': seed.get('mae_ticks', 0) * 0.25,
                    'mfe_bar': seed.get('n_bars', 0),
                    'duration_mins': seed.get('duration_mins', 0),
                    'seed_id': seed['_global_id'],
                    'direction': seed['direction'],
                    'source': 'auto_swing',
                }

                z = getattr(state, 'z_score', 0.0)
                pattern_type = 'BAND_REVERSAL' if abs(z) > 1.5 else 'MOMENTUM_BREAK'

                pe = PatternEvent(
                    pattern_type=pattern_type,
                    timestamp=float(day_timestamps[best_idx]),
                    price=float(day_prices[best_idx]),
                    z_score=getattr(state, 'z_score', 0.0),
                    velocity=getattr(state, 'velocity', 0.0),
                    momentum=getattr(state, 'momentum_strength', 0.0),
                    entropy_normalized=getattr(state, 'entropy_normalized', 0.0),
                    file_source=f"auto_swing:{date_str}:T{seed.get('trade_id', seed['_global_id'])}",
                    idx=best_idx,
                    state=state,
                    timeframe=timeframe,
                    depth=depth,
                    parent_type='',
                    parent_tf='',
                    window_data=None,
                    parent_chain=[],
                    oracle_marker=marker,
                    oracle_meta=oracle_meta,
                )
                manifest.append(pe)

        # Free GPU memory between months
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    matched = len(manifest)
    print(f"\n  Result: {matched:,} PatternEvents created, {skipped:,} skipped")

    if manifest:
        markers = [p.oracle_marker for p in manifest]
        print(f"  Oracle breakdown:")
        for label, val in [('MEGA_LONG', 2), ('SCALP_LONG', 1), ('NOISE', 0),
                           ('SCALP_SHORT', -1), ('MEGA_SHORT', -2)]:
            count = sum(1 for m in markers if m == val)
            if count > 0:
                print(f"    {label:>12s}: {count:,}")

        types = [p.pattern_type for p in manifest]
        print(f"  Pattern types: BAND_REVERSAL={sum(1 for t in types if t=='BAND_REVERSAL'):,}, "
              f"MOMENTUM_BREAK={sum(1 for t in types if t=='MOMENTUM_BREAK'):,}")

        dirs = [p.oracle_meta.get('direction', '?') for p in manifest]
        print(f"  Direction: LONG={sum(1 for d in dirs if d=='LONG'):,}, "
              f"SHORT={sum(1 for d in dirs if d=='SHORT'):,}")

    return manifest
