import os
import shutil
import glob

# Ensure dirs exist
dirs = [
    "training/datasets", "training/models/cnn", "training/models/gbm",
    "training/strategies", "training/pipelines", "training/calibration",
    "training/analysis", "training/regret", "training/utils", "training/archive"
]
for d in dirs:
    os.makedirs(d, exist_ok=True)

with open('training/__init__.py', 'w') as f: pass
with open('training/README.md', 'w') as f: pass

# 1. B-stack trainers: tools/train_b*.py -> training/models/gbm/b*.py
for f in glob.glob('tools/train_b*.py'):
    base = os.path.basename(f).replace('train_b', 'b')
    shutil.copy(f, f"training/models/gbm/{base}")

# 2. Calibration scripts: tools/calibrate_*.py -> training/calibration/
for f in glob.glob('tools/calibrate_*.py'):
    shutil.copy(f, f"training/calibration/{os.path.basename(f)}")

# 3. tools/velocity_regime.py, seed_per_regime.py -> training/calibration/
for f in ['tools/velocity_regime.py', 'tools/seed_per_regime.py']:
    if os.path.exists(f):
        shutil.copy(f, f"training/calibration/{os.path.basename(f)}")

# 4. core_v2/exits.py
if os.path.exists('training_iso_v2/exits.py'):
    shutil.copy('training_iso_v2/exits.py', 'core_v2/exits.py')

# 5. Move files from training_iso_v2 (preferred) and training_v2
srcs = ['training_iso_v2', 'training_v2']

mapping = {
    'threshold_optimizer.py': 'training/calibration',
    'threshold_bayesian.py': 'training/calibration',
    'threshold_mode_tuned.py': 'training/calibration',
    'learn_zband_thresholds.py': 'training/calibration',
    'tier_discovery.py': 'training/calibration',
    'cell_filters.py': 'training/calibration',
    
    'full_feature_eda.py': 'training/analysis/feature_eda.py',
    'within_cell_eda.py': 'training/analysis',
    'flip_rule_validation.py': 'training/analysis',
    'loser_autopsy.py': 'training/analysis',
    'bleed_cause_analysis.py': 'training/analysis',
    'vol_adaptive_test.py': 'training/analysis',
    
    'bayesian_table.py': 'training/regret',
    'regret.py': 'training/regret',
    'regret_full.py': 'training/regret',
    'regret_by_regime.py': 'training/regret',
    
    'run.py': 'training/pipelines/v2_native.py',
    'pipeline.py': 'training/pipelines/iso.py',
    'run_iso.py': 'training/pipelines/run_iso.py',
    'iso_orchestrator.py': 'training/pipelines/iso_orchestrator.py',
    
    'ticker.py': 'training/utils/ticker.py',
    'state.py': 'training/utils/state.py',
    'v2_cols.py': 'training/utils/v2_cols.py',
    'regime_router.py': 'training/utils/regime_router.py',
    
    'cnn': 'training/models/cnn',
    'strategies': 'training/strategies',
}

for src_dir in srcs:
    if not os.path.exists(src_dir): continue
    for src_name, dest in mapping.items():
        src_path = os.path.join(src_dir, src_name)
        if os.path.exists(src_path):
            if os.path.isdir(src_path):
                dest_path = dest
                if not os.path.exists(dest_path):
                    shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
            else:
                dest_path = dest
                if os.path.isdir(dest):
                    dest_path = os.path.join(dest, os.path.basename(src_path))
                if not os.path.exists(dest_path):
                    shutil.copy(src_path, dest_path)

# 6. training_zigzag
if os.path.exists('training_zigzag/forward_zigzag.py'):
    shutil.copy('training_zigzag/forward_zigzag.py', 'training/archive/forward_zigzag.py')
if os.path.exists('training_zigzag/v2_ticker.py'):
    shutil.copy('training_zigzag/v2_ticker.py', 'training/archive/v2_ticker.py')

# 7. training_iso
if os.path.exists('training_iso/nightmare_iso.py'):
    shutil.copy('training_iso/nightmare_iso.py', 'training/archive/nightmare_iso.py')

print("Phase 4 migration completed.")
