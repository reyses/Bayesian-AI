"""DRS Path A - Step 2: build IS pivot-probability cloud.

Thin wrapper that calls the deliverable's `pivot_probability_cloud.py` with
the IS prediction caches produced by step 1. Does NOT modify the deliverable.

Run:
  python tools/sourcing/drs_a_step2_build_cloud_is.py

Expected runtime: ~30s. Output: DATA/CROSS_DAY/predictions_IS/pivot_probability_cloud_IS.parquet
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

DELIVER = Path('deliverables/composite_zigzag_pipeline')
SCRIPT  = DELIVER / 'tools' / 'pivot_probability_cloud.py'

OUT_PARQ = Path('DATA/CROSS_DAY/predictions_IS/pivot_probability_cloud_IS.parquet')
OUT_RPT  = Path('DATA/CROSS_DAY/predictions_IS/pivot_probability_cloud_IS.txt')

ARGS = [
    sys.executable, str(SCRIPT),
    '--b1-cache', 'DATA/CROSS_DAY/predictions_IS/b1_proba_IS.parquet',
    '--truth',    str(DELIVER / 'caches' / 'zigzag_pivot_dataset_IS_atr4.parquet'),
    '--b4-cache', 'DATA/CROSS_DAY/predictions_IS/b4_proba_IS.parquet',
    '--b5-cache', 'DATA/CROSS_DAY/predictions_IS/b5_proba_IS.parquet',
    '--out-parquet', str(OUT_PARQ),
    '--out-report',  str(OUT_RPT),
]

if __name__ == '__main__':
    OUT_PARQ.parent.mkdir(parents=True, exist_ok=True)
    print('Calling:', ' '.join(ARGS))
    rc = subprocess.run(ARGS).returncode
    sys.exit(rc)
