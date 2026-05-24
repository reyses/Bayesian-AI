import os
import glob
import re

mapping = {
    'threshold_optimizer': 'training.calibration.threshold_optimizer',
    'threshold_bayesian': 'training.calibration.threshold_bayesian',
    'threshold_mode_tuned': 'training.calibration.threshold_mode_tuned',
    'learn_zband_thresholds': 'training.calibration.learn_zband_thresholds',
    'tier_discovery': 'training.calibration.tier_discovery',
    'cell_filters': 'training.calibration.cell_filters',
    
    'full_feature_eda': 'training.analysis.feature_eda',
    'within_cell_eda': 'training.analysis.within_cell_eda',
    'flip_rule_validation': 'training.analysis.flip_rule_validation',
    'loser_autopsy': 'training.analysis.loser_autopsy',
    'bleed_cause_analysis': 'training.analysis.bleed_cause_analysis',
    'vol_adaptive_test': 'training.analysis.vol_adaptive_test',
    
    'bayesian_table': 'training.regret.bayesian_table',
    'regret': 'training.regret.regret',
    'regret_full': 'training.regret.regret_full',
    'regret_by_regime': 'training.regret.regret_by_regime',
    
    'run': 'training.pipelines.v2_native',
    'pipeline': 'training.pipelines.iso',
    'run_iso': 'training.pipelines.run_iso',
    'iso_orchestrator': 'training.pipelines.iso_orchestrator',
    
    'ticker': 'training.utils.ticker',
    'state': 'training.utils.state',
    'v2_cols': 'training.utils.v2_cols',
    'regime_router': 'training.utils.regime_router',
    
    'engine': 'core_v2.strategy_engine',
    'exits': 'core_v2.exits',
    'exits_tick_exact': 'core_v2.exits_tick_exact',
    
    'cnn.': 'training.models.cnn.',
    'strategies.': 'training.strategies.',
}

prefixes = ['training_v2', 'training_iso_v2', 'training_iso', 'training_zigzag']

files = glob.glob('**/*.py', recursive=True)
for f in files:
    # skip old folders
    if any(f.startswith(p) for p in prefixes):
        continue
    
    try:
        with open(f, 'r', encoding='utf-8') as file:
            content = file.read()
    except UnicodeDecodeError:
        try:
            with open(f, 'r', encoding='utf-16') as file:
                content = file.read()
        except:
            continue
            
    orig = content
    for p in prefixes:
        for k, v in mapping.items():
            content = content.replace(f'from {p}.{k}', f'from {v}')
            content = content.replace(f'import {p}.{k}', f'import {v}')
            content = content.replace(f'{p}.{k}', f'{v}')
        
        content = content.replace(f'from {p} import', f'from training import')
        content = content.replace(f'import {p}', f'import training')
        content = content.replace(f'{p}.', f'training.')
        
    if content != orig:
        with open(f, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"Updated {f}")

print("Import rewrites completed.")
