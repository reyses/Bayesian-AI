import os
import glob

def patch_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # 1. replace resolution_idx
    old_res = "'resolution_idx': _exit_idx if '_exit_idx' in locals() else -1"
    new_res = "'resolution_idx': (_exit_idx + (event_idx if 'event_idx' in locals() else (e_idx if 'e_idx' in locals() else i)) + 1) if ('_exit_idx' in locals() and _exit_idx != -1) else -1"
    content = content.replace(old_res, new_res)

    # 2. replace depth with duration_bars AND depth oneliner
    old_depth = "'depth': (_exit_idx if '_exit_idx' in locals() else -1) - (event_idx if 'event_idx' in locals() else (e_idx if 'e_idx' in locals() else 0))"
    
    # We use a helper function call or inline generator
    oneliner = "'duration_bars': _exit_idx if '_exit_idx' in locals() else -1,\n"
    oneliner += "                        'depth': (lambda l: next((abs(float(l[k])) for k in ['magnitude', 'div', 'adx_val', 'z', 'z_val', 'z_score', 'distance', 'gap'] if k in l and l[k] is not None), abs(l.get('p0',0) - l.get('open_price',0)) if 'p0' in l and 'open_price' in l else 0.0))(locals())"
    
    content = content.replace(old_depth, oneliner)
    
    # 3. Bar-level ORDERFLOW Filtering
    if "14_orderflow" in filepath:
        old_filter = "if magnitude < 5 or magnitude > 200:"
        new_filter = '''
            # [FIX] Bar-level spike-and-revert filter
            closes = df_day['close'].values
            bad_mask = np.zeros(len(df_day), dtype=bool)
            for j in range(1, len(df_day) - 1):
                if abs(closes[j] - closes[j-1]) > 50 and abs(closes[j] - closes[j+1]) > 50:
                    bad_mask[j] = True
            if bad_mask.sum() > 0:
                print(f"[Spike Filter] Dropped {bad_mask.sum()} corrupt bars on {day_str}")
            df_day = df_day[~bad_mask].copy()
            df_day.reset_index(drop=True, inplace=True)
            
            if False:'''
        content = content.replace(old_filter, new_filter)

    with open(filepath, 'w') as f:
        f.write(content)

if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tests'))
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.startswith("ag_deepdive_") and file.endswith(".py"):
                patch_file(os.path.join(root, file))
    print("One-liner patch complete.")
