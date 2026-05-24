"""Pre-session SIM deploy preflight check.

Run before launching `python -m live.engine_v2 --engine-mode l5` to
verify all preconditions for the L5 stack:

  1. ATLAS_NT8 data current to yesterday
  2. cross_day_features.parquet has today's row (B10 needs it)
  3. B7/B9/B10 model pickles load
  4. V2 streaming parity passes on yesterday
  5. LiveFeatureEngineV2 + L5Decider import cleanly
  6. NT8 BayesianBridge port 5199 reachable (optional, requires NT8 running)
  7. Latest live checkpoint timestamp not stale (warns if > 1 day old)

Output: green/red per check + exit code 0 (all pass) or 1 (any fail).

Usage:
    python tools/preflight_check.py            # all checks
    python tools/preflight_check.py --skip-nt8 # skip NT8 connection probe
"""
from __future__ import annotations
import argparse
import json
import os
import socket
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'


def ok(msg):
    print(f'{GREEN}[ OK ]{RESET}  {msg}')


def fail(msg):
    print(f'{RED}[FAIL]{RESET}  {msg}')


def warn(msg):
    print(f'{YELLOW}[WARN]{RESET}  {msg}')


def check_atlas_nt8_current() -> bool:
    """ATLAS_NT8 must have data within last 3 calendar days."""
    p = Path('DATA/ATLAS_NT8/5s')
    if not p.exists():
        fail(f'Missing DATA/ATLAS_NT8/5s directory')
        return False
    files = sorted(p.glob('*.parquet'))
    if not files:
        fail(f'No 5s parquets under DATA/ATLAS_NT8/5s/')
        return False
    latest = files[-1].stem  # 'YYYY_MM_DD'
    try:
        latest_dt = datetime.strptime(latest, '%Y_%m_%d')
    except ValueError:
        fail(f'Cannot parse latest day: {latest}')
        return False
    today = datetime.utcnow()
    age_days = (today - latest_dt).days
    if age_days > 3:
        fail(f'ATLAS_NT8 latest day {latest} is {age_days}d old. '
              f'Run NT8 BayesianHistoryDumper to refresh.')
        return False
    if age_days > 1:
        warn(f'ATLAS_NT8 latest day {latest} is {age_days}d old (acceptable).')
    else:
        ok(f'ATLAS_NT8 current to {latest} ({age_days}d old)')
    return True


def check_cross_day_features() -> bool:
    """cross_day_features.parquet must have today's row."""
    p = Path('DATA/CROSS_DAY/cross_day_features.parquet')
    if not p.exists():
        fail(f'Missing {p}. Run tools/sourcing/build_cross_day_features.py')
        return False
    import pandas as pd
    try:
        df = pd.read_parquet(p)
    except Exception as e:
        fail(f'Cannot read {p}: {e}')
        return False
    today_label = datetime.utcnow().strftime('%Y_%m_%d')
    has_today = (df['date_label'] == today_label).any()
    if has_today:
        ok(f'cross_day_features has today ({today_label})')
        return True

    # Check yesterday as fallback (session-day cutoffs vary)
    yesterday_label = (datetime.utcnow() - timedelta(days=1)).strftime('%Y_%m_%d')
    has_yesterday = (df['date_label'] == yesterday_label).any()
    if has_yesterday:
        warn(f'cross_day_features has yesterday ({yesterday_label}) but not '
              f'today ({today_label}). B10 will default to normal mode.')
        return True

    fail(f'cross_day_features missing today AND yesterday. '
          f'Run tools/sourcing/build_cross_day_features.py.')
    return False


def check_l5_models() -> bool:
    """B7/B9/B10 pickles must load cleanly."""
    import pickle
    paths = {
        'B7':      'reports/findings/regret_oracle/b7_leg_sizer.pkl',
        'B9 K=5':  'reports/findings/regret_oracle/b9_remaining_amplitude_K5.pkl',
        'B10 high':'reports/findings/regret_oracle/b10_vol_regime_high.pkl',
        'B10 low': 'reports/findings/regret_oracle/b10_vol_regime_low.pkl',
    }
    all_ok = True
    for name, path in paths.items():
        if not os.path.exists(path):
            fail(f'{name} pickle missing: {path}')
            all_ok = False
            continue
        try:
            with open(path, 'rb') as f:
                m = pickle.load(f)
            n = (len(m.get('v2_cols', [])) or len(m.get('feat_cols', []))
                  or '?')
            ok(f'{name} loaded ({n} features)')
        except Exception as e:
            fail(f'{name} pickle load failed: {e}')
            all_ok = False
    return all_ok


def check_v2_parity() -> bool:
    """Run V2 streaming-vs-batch parity on yesterday (quick: 50 anchors)."""
    import subprocess
    p = Path('DATA/ATLAS_NT8/FEATURES_5s_v2/L0')
    if not p.exists():
        fail('No FEATURES_5s_v2 batch parquets to validate against')
        return False
    files = sorted(p.glob('*.parquet'))
    if not files:
        fail('Empty FEATURES_5s_v2/L0/')
        return False
    test_day = files[-2].stem if len(files) >= 2 else files[-1].stem
    print(f'  Running parity test on {test_day} (50 anchors)...')
    try:
        result = subprocess.run(
            [sys.executable, 'tools/test_live_v2_parity.py',
              '--day', test_day, '--max-anchors', '50',
              '--first-skip', '500'],
            capture_output=True, text=True, timeout=180,
        )
        last_lines = result.stdout.strip().split('\n')[-5:]
        if 'PASS' in result.stdout:
            ok(f'V2 parity on {test_day}: PASS')
            return True
        fail(f'V2 parity FAILED. Last output:')
        for ln in last_lines:
            print(f'    {ln}')
        return False
    except subprocess.TimeoutExpired:
        fail('V2 parity test timed out (>180s)')
        return False
    except Exception as e:
        fail(f'V2 parity test crashed: {e}')
        return False


def check_l5_imports() -> bool:
    """LiveFeatureEngineV2 + L5Decider + engine_v2 must import."""
    try:
        from training.live_feature_engine_v2 import LiveFeatureEngineV2
        from live.l5_decider import L5Decider, L5Context
        from live.engine_v2 import LiveEngineV2
        ok('L5 stack imports clean')
        return True
    except Exception as e:
        fail(f'L5 stack import failed: {e}')
        return False


def check_nt8_port(host: str = '127.0.0.1', port: int = 5199,
                     timeout: float = 2.0) -> bool:
    """Probe NT8 BayesianBridge TCP port. Warning only (NT8 may be off pre-session)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            r = s.connect_ex((host, port))
            if r == 0:
                ok(f'NT8 BayesianBridge reachable on {host}:{port}')
                return True
            warn(f'NT8 port {host}:{port} not reachable (rc={r}). '
                  f'Start NT8 + load BayesianBridge.cs before live launch.')
            return True  # warning, not failure
    except Exception as e:
        warn(f'NT8 probe failed: {e}')
        return True


def check_checkpoint_age() -> bool:
    """Live checkpoint age. Warn if > 1 day."""
    paths = ['live/state/checkpoint.json', 'DATA/ATLAS_NT8/checkpoint.json']
    found = False
    for p in paths:
        if not os.path.exists(p):
            continue
        found = True
        try:
            with open(p) as f:
                cp = json.load(f)
            last_ts = cp.get('last_ts', 0)
            if last_ts == 0:
                warn(f'{p} has no last_ts')
                continue
            age_h = (datetime.utcnow().timestamp() - last_ts) / 3600
            if age_h > 48:
                warn(f'{p} is {age_h:.0f}h old')
            else:
                ok(f'{p} age {age_h:.0f}h')
        except Exception as e:
            warn(f'Cannot read {p}: {e}')
    if not found:
        warn('No checkpoint found -- engine will cold-start')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-nt8', action='store_true',
                    help='Skip NT8 port probe (e.g. NT8 not running yet).')
    ap.add_argument('--skip-parity', action='store_true',
                    help='Skip V2 parity test (saves ~60s).')
    args = ap.parse_args()

    print('=' * 70)
    print(f'L5 SIM DEPLOY PREFLIGHT  {datetime.utcnow().isoformat()}Z')
    print('=' * 70)

    checks = []
    checks.append(('ATLAS_NT8 current',    check_atlas_nt8_current()))
    checks.append(('cross_day_features',   check_cross_day_features()))
    checks.append(('L5 models load',       check_l5_models()))
    checks.append(('L5 imports',           check_l5_imports()))
    if not args.skip_parity:
        checks.append(('V2 streaming parity', check_v2_parity()))
    if not args.skip_nt8:
        checks.append(('NT8 port 5199',     check_nt8_port()))
    checks.append(('Checkpoint age',       check_checkpoint_age()))

    print()
    print('=' * 70)
    print(f'SUMMARY: {sum(1 for _, p in checks if p)}/{len(checks)} pass')
    print('=' * 70)
    failures = [n for n, p in checks if not p]
    if failures:
        print(f'{RED}FAILED:{RESET}  {", ".join(failures)}')
        return 1
    print(f'{GREEN}ALL PREFLIGHT CHECKS PASSED{RESET}')
    print('Launch with:  python -m live.engine_v2 --engine-mode l5 --mock')
    print('(remove --mock when ready to connect real NT8)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
