"""
Phase 4: Terminology Refactor Script
Renames quantum physics metaphors to standard statistical terminology.
Run from project root: python scripts/terminology_refactor.py [--dry-run]
"""
import os
import sys
import re

DRY_RUN = '--dry-run' in sys.argv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Skip archive/ — those are frozen historical files
SKIP_DIRS = {'archive', '__pycache__', '.git', 'node_modules', 'backups', '.claude'}
SKIP_FILES = {'terminology_refactor.py'}  # Don't rename ourselves

# ── RENAME MAP (order matters: longer/more-specific first) ────────────────
# Each tuple: (old, new)
RENAMES = [
    # Classes (most specific first)
    ('ThreeBodyQuantumState', 'MarketState'),
    ('QuantumFieldEngine', 'StatisticalFieldEngine'),
    ('QuantumRiskEngine', 'MonteCarloRiskEngine'),

    # Multi-word attributes (before shorter substrings)
    ('resonance_coherence', 'alignment_score'),
    ('fractal_alignment_count', 'multi_tf_alignment_count'),
    ('particle_position', 'price'),
    ('particle_velocity', 'velocity'),
    ('center_position', 'regression_center'),
    ('event_horizon_upper', 'upper_band_3sigma'),
    ('event_horizon_lower', 'lower_band_3sigma'),
    ('upper_singularity', 'upper_band_2sigma'),
    ('lower_singularity', 'lower_band_2sigma'),
    ('tunnel_probability', 'reversion_probability'),
    ('escape_probability', 'breakout_probability'),
    ('sigma_fractal', 'regression_sigma'),
    ('lagrange_zone', 'band_zone'),
    ('spin_inverted', 'reversal_confirmed'),
    ('time_at_roche', 'time_at_band_extreme'),
    ('F_reversion', 'mean_reversion_force'),

    # Enum values
    ('STRUCTURAL_DRIVE', 'MOMENTUM_BREAK'),
    ('ROCHE_SNAP', 'BAND_REVERSAL'),
    ('L1_STABLE', 'INNER'),
    ('L2_ROCHE', 'UPPER_EXTREME'),
    ('L3_ROCHE', 'LOWER_EXTREME'),

    # F_net needs word boundary care (don't match _F_net_something)
    # Handled specially below

    # coherence → entropy_normalized (handled specially — skip cluster quality usage)
]

# These need word-boundary-aware replacement
WORD_BOUNDARY_RENAMES = [
    ('F_net', 'net_force'),
]


def find_py_files(root):
    """Find all .py files, skipping SKIP_DIRS."""
    result = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skipped directories
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in filenames:
            if f.endswith('.py') and f not in SKIP_FILES:
                result.append(os.path.join(dirpath, f))
    return sorted(result)


def apply_coherence_rename(content, filepath):
    """
    Rename 'coherence' → 'entropy_normalized' ONLY for state field usage.
    Skip cluster quality metric usage in waveform_standalone.py.
    """
    # In waveform_standalone.py, there are two different 'coherence' concepts:
    # 1. Feature vector position [3] = state.coherence → rename
    # 2. Cluster quality metric (local var) → do NOT rename
    #
    # Strategy: rename attribute access patterns and dataclass field declarations.
    # Leave bare variable names in waveform clustering code alone.

    rel = os.path.relpath(filepath, PROJECT_ROOT)

    # Files where ALL 'coherence' refs are the state field → safe global replace
    safe_global = [
        'core/three_body_state.py',
        'core/quantum_field_engine.py',
        'live/live_engine.py',
        'training/timeframe_belief_network.py',
        'training/fractal_discovery_agent.py',
        'tests/test_pid_analyzer.py',
        'tests/test_clustering_integration.py',
        'scripts/verify_clustering.py',
    ]

    rel_normalized = rel.replace('\\', '/')

    if rel_normalized in safe_global:
        content = content.replace('coherence', 'entropy_normalized')
        return content

    # For other files (especially waveform_standalone.py):
    # Only rename attribute access and specific patterns
    # .coherence → .entropy_normalized
    content = re.sub(r'\.coherence\b', '.entropy_normalized', content)
    # coherence= in function calls/dataclass fields
    content = re.sub(r'\bcoherence\s*=', 'entropy_normalized=', content)
    # 'coherence' as string key in feature vectors referencing the state field
    content = content.replace("'coherence'", "'entropy_normalized'")
    # "coherence" double-quoted
    content = content.replace('"coherence"', '"entropy_normalized"')

    return content


def apply_renames(content, filepath):
    """Apply all renames to file content."""
    original = content

    # 1. Standard string replacements (order matters)
    for old, new in RENAMES:
        content = content.replace(old, new)

    # 2. Word-boundary renames
    for old, new in WORD_BOUNDARY_RENAMES:
        content = re.sub(r'\b' + re.escape(old) + r'\b', new, content)

    # 3. Coherence (special handling)
    if 'coherence' in content:
        content = apply_coherence_rename(content, filepath)

    return content, content != original


def main():
    files = find_py_files(PROJECT_ROOT)
    changed_files = []
    total_files = 0

    for fpath in files:
        try:
            with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
        except Exception as e:
            print(f"  SKIP (read error): {fpath} — {e}")
            continue

        new_content, changed = apply_renames(content, fpath)

        if changed:
            rel = os.path.relpath(fpath, PROJECT_ROOT)
            changed_files.append(rel)
            total_files += 1
            if DRY_RUN:
                print(f"  WOULD CHANGE: {rel}")
            else:
                with open(fpath, 'w', encoding='utf-8', newline='') as f:
                    f.write(new_content)
                print(f"  CHANGED: {rel}")

    print(f"\n{'DRY RUN — ' if DRY_RUN else ''}Total files {'would be ' if DRY_RUN else ''}changed: {total_files}")

    if not DRY_RUN and changed_files:
        print("\nChanged files:")
        for f in changed_files:
            print(f"  {f}")


if __name__ == '__main__':
    main()
