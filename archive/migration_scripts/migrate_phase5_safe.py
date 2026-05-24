import os
import shutil
import glob

tools_dir = 'tools'

for d in os.listdir(tools_dir):
    full_path = os.path.join(tools_dir, d)
    if os.path.isdir(full_path) and d.startswith('_'):
        new_name = d[1:]
        new_full_path = os.path.join(tools_dir, new_name)
        if os.path.exists(new_full_path):
            for item in os.listdir(full_path):
                shutil.move(os.path.join(full_path, item), new_full_path)
            os.rmdir(full_path)
        else:
            os.rename(full_path, new_full_path)

new_categories = ['entries', 'exits', 'forward_pass', 'zigzag', 'regime', 'risk', 'suites', 'archive', 'features']
for cat in new_categories:
    os.makedirs(os.path.join(tools_dir, cat), exist_ok=True)

flat_files = [f for f in os.listdir(tools_dir) if os.path.isfile(os.path.join(tools_dir, f)) and f.endswith('.py')]
for f in flat_files:
    if f == '__init__.py': continue
    src = os.path.join(tools_dir, f)
    dst_folder = 'archive'
    if 'regime' in f: dst_folder = 'regime'
    elif 'zigzag' in f: dst_folder = 'zigzag'
    elif 'risk' in f or 'blowout' in f or 'drawdown' in f or 'saturation' in f: dst_folder = 'risk'
    elif 'forward' in f or 'sim_' in f or 'fps' in f: dst_folder = 'forward_pass'
    elif 'exit' in f or 'giveback' in f: dst_folder = 'exits'
    elif 'entry' in f or 'filter' in f: dst_folder = 'entries'
    elif 'feature' in f or 'v2_' in f: dst_folder = 'features'
    elif 'eda' in f: dst_folder = 'eda'
    elif 'level' in f or 'peak' in f: dst_folder = 'levels'
    elif 'tier' in f: dst_folder = 'tier'
    elif 'regret' in f: dst_folder = 'regret'
    elif 'pivot' in f: dst_folder = 'pivot'
    elif 'data' in f or 'validate' in f or 'build_' in f or 'extract_' in f: dst_folder = 'data'
    elif 'chart' in f or 'plot' in f: dst_folder = 'charts'
    elif 'test' in f: dst_folder = 'util'
    if 'suite' in f: dst_folder = 'suites'
    
    shutil.move(src, os.path.join(tools_dir, dst_folder, f))

if os.path.exists(os.path.join(tools_dir, 'trade_outcome_suite')):
    shutil.move(os.path.join(tools_dir, 'trade_outcome_suite'), os.path.join(tools_dir, 'suites', 'trade_outcome_suite'))

if os.path.exists('research/TOOLS_INDEX.md'):
    shutil.move('research/TOOLS_INDEX.md', 'tools/TOOLS_INDEX.md')

print("Phase 5 folder migration completed.")
