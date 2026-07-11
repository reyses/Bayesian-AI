import os
import glob
import re

def process_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # 1. Fix resolution_idx
    # Old: 'resolution_idx': _exit_idx if '_exit_idx' in locals() else -1,
    old_res = "'resolution_idx': _exit_idx if '_exit_idx' in locals() else -1,"
    new_res = "'resolution_idx': (_exit_idx + (event_idx if 'event_idx' in locals() else (e_idx if 'e_idx' in locals() else i)) + 1) if ('_exit_idx' in locals() and _exit_idx != -1) else -1,"
    
    if old_res in content:
        content = content.replace(old_res, new_res)
    else:
        # Might be slightly different
        pass
        
    # 2. Fix depth
    # Old: 'depth': (_exit_idx if '_exit_idx' in locals() else -1) - (event_idx if 'event_idx' in locals() else (e_idx if 'e_idx' in locals() else 0)),
    old_depth = "'depth': (_exit_idx if '_exit_idx' in locals() else -1) - (event_idx if 'event_idx' in locals() else (e_idx if 'e_idx' in locals() else 0)),"
    
    # We will inject _trigger_depth right before results.append
    injector = """
        # --- ROUND 2 DEPTH FIX ---
        _trigger_depth = 0.0
        if 'div' in locals() and div is not None: _trigger_depth = abs(div)
        elif 'adx_val' in locals() and adx_val is not None: _trigger_depth = float(adx_val)
        elif 'z' in locals() and z is not None: _trigger_depth = abs(z)
        elif 'z_val' in locals() and z_val is not None: _trigger_depth = abs(z_val)
        elif 'z_score' in locals() and z_score is not None: _trigger_depth = abs(z_score)
        elif 'distance' in locals() and distance is not None: _trigger_depth = abs(distance)
        elif 'gap' in locals() and gap is not None: _trigger_depth = abs(gap)
        elif 'p0' in locals() and 'open_price' in locals(): _trigger_depth = abs(p0 - open_price)
        """
        
    if "results.append({" in content:
        content = content.replace("results.append({", injector + "\n        results.append({")
    elif "events.append({" in content and "resolution_idx" in content:
        # For scripts that append directly to events
        content = content.replace("events.append({", injector + "\n            events.append({")

    new_depth = "'duration_bars': _exit_idx if '_exit_idx' in locals() else -1,\n        'depth': _trigger_depth,"
    
    if old_depth in content:
        content = content.replace(old_depth, new_depth)

    # 3. Orderflow specific
    if "14_orderflow" in filepath:
        # Remove the old skip filter
        old_skip = "if abs(magnitude) > 100.0:\n                print(f\"[Skip Filter] Dropped {magnitude:.2f} pts anomaly at idx {i} on {day_str}\")\n                continue"
        if old_skip in content:
            content = content.replace(old_skip, "")
            
        # Add the bar level filter
        bar_filter = """
    # [FIX] Bar-level spike-and-revert filter
    closes = df_day['close'].values
    bad_mask = np.zeros(len(df_day), dtype=bool)
    for j in range(1, len(df_day) - 1):
        if abs(closes[j] - closes[j-1]) > 50 and abs(closes[j] - closes[j+1]) > 50:
            bad_mask[j] = True
    corrupt_count = bad_mask.sum()
    if corrupt_count > 0:
        print(f"[Spike Filter] Dropped {corrupt_count} corrupt bars on {day_str}")
    df_day = df_day[~bad_mask].copy()
"""
        if "df_day = df_day.sort_values('dt').copy()" in content and "bad_mask" not in content:
            content = content.replace("df_day = df_day.sort_values('dt').copy()", "df_day = df_day.sort_values('dt').copy()\n" + bar_filter)

    with open(filepath, 'w') as f:
        f.write(content)
        
if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tests'))
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.startswith("ag_deepdive_") and file.endswith(".py"):
                process_file(os.path.join(root, file))
    print("Patching complete.")
