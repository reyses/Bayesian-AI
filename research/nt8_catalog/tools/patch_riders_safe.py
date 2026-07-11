import os
import glob

def patch_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # 1. replace resolution_idx
        if "'resolution_idx': _exit_idx if '_exit_idx' in locals() else -1" in line:
            line = line.replace("'resolution_idx': _exit_idx if '_exit_idx' in locals() else -1", 
                "'resolution_idx': (_exit_idx + (event_idx if 'event_idx' in locals() else (e_idx if 'e_idx' in locals() else i)) + 1) if ('_exit_idx' in locals() and _exit_idx != -1) else -1")
                
        # 2. replace depth
        if "'depth': (_exit_idx if '_exit_idx' in locals() else -1) - (event_idx if 'event_idx' in locals() else (e_idx if 'e_idx' in locals() else 0))" in line:
            line = line.replace(
                "'depth': (_exit_idx if '_exit_idx' in locals() else -1) - (event_idx if 'event_idx' in locals() else (e_idx if 'e_idx' in locals() else 0))",
                "'duration_bars': _exit_idx if '_exit_idx' in locals() else -1,\n" + 
                (len(line) - len(line.lstrip())) * " " + "'depth': _trigger_depth"
            )
            
        # 3. inject _trigger_depth before events.append({ or results.append({
        if "events.append({" in line or "results.append({" in line:
            # We ONLY want to inject if it's the append that creates the dictionary with 'resolution_idx'
            # Let's just check if the next lines contain 'resolution_idx'
            # Lookahead up to 15 lines
            found = False
            for j in range(i, min(i+15, len(lines))):
                if "'resolution_idx'" in lines[j]:
                    found = True
                    break
            
            if found:
                indent = len(line) - len(line.lstrip())
                ind_s = " " * indent
                injector = [
                    f"{ind_s}# --- ROUND 2 DEPTH FIX ---\n",
                    f"{ind_s}_trigger_depth = 0.0\n",
                    f"{ind_s}if 'div' in locals() and div is not None: _trigger_depth = abs(div)\n",
                    f"{ind_s}elif 'adx_val' in locals() and adx_val is not None: _trigger_depth = float(adx_val)\n",
                    f"{ind_s}elif 'z' in locals() and z is not None: _trigger_depth = abs(z)\n",
                    f"{ind_s}elif 'z_val' in locals() and z_val is not None: _trigger_depth = abs(z_val)\n",
                    f"{ind_s}elif 'z_score' in locals() and z_score is not None: _trigger_depth = abs(z_score)\n",
                    f"{ind_s}elif 'distance' in locals() and distance is not None: _trigger_depth = abs(distance)\n",
                    f"{ind_s}elif 'gap' in locals() and gap is not None: _trigger_depth = abs(gap)\n",
                    f"{ind_s}elif 'p0' in locals() and 'open_price' in locals(): _trigger_depth = abs(p0 - open_price)\n"
                ]
                out.extend(injector)
                
        out.append(line)
        i += 1
        
    with open(filepath, 'w') as f:
        f.writelines(out)

if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tests'))
    files_to_patch = [
        "ADX-08_Trend_Gate/ag_deepdive_08_adx.py",
        "ATR-09_Statistical_Fade/ag_deepdive_09_atr.py",
        "FIB-17_Confluence/ag_deepdive_17_fib.py",
        "OHLC-01_Prior_Day/ag_deepdive_01_ohlc.py",
        "ORDERFLOW-14/ag_deepdive_14_orderflow.py",
        "PIVOT-16_Floor_Levels/ag_deepdive_16_pivots.py"
    ]
    for rel_f in files_to_patch:
        f = os.path.join(base_dir, rel_f)
        if os.path.exists(f):
            patch_file(f)
    print("Safely patched.")
