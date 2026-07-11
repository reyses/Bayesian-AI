import os
import glob
import re

def generate_sweep_summary():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    tests_dir = os.path.join(base_dir, 'tests')
    reports_dir = os.path.join(base_dir, 'reports')
    
    os.makedirs(reports_dir, exist_ok=True)
    
    md_files = glob.glob(os.path.join(tests_dir, '**', 'DOC_*.md'), recursive=True)
    md_files = [f for f in md_files if 'archive' not in f.lower()]
    
    summary_lines = []
    summary_lines.append("# Document ID: AG-CAT-00-SWEEP-SUMMARY")
    summary_lines.append("**Title:** Master Catalog Sweep Summary (Phase 4 Unit-Standardized)")
    summary_lines.append("**Status:** Audit Completed (AUDIT-ACC-01 §3)")
    summary_lines.append("")
    summary_lines.append("> [!WARNING] AUDIT MANDATE")
    summary_lines.append("> **NO UNCONDITIONALLY STABLE POSITIVE EDGES WERE FOUND** across the 18 base hypotheses over the 2024–2025 dataset.")
    summary_lines.append("> All positive EV results observed in the initial raw sweeps failed to survive strict causality firewalls, lookahead corrections, and standardisation to the ±2.05σ symmetric barrier metric.")
    summary_lines.append("")
    summary_lines.append("## Consolidated Unit-Standardized Sweep")
    summary_lines.append("")
    summary_lines.append("| Year | Dossier | Setup | N | WR% | EV (Mean σ) | EV 95% CI | Sig? |")
    summary_lines.append("|---|---|---|---|---|---|---|---|")
    
    all_data = []
    
    for md_file in md_files:
        dossier_name = os.path.basename(os.path.dirname(md_file))
        dossier_id = dossier_name.split('_')[0] if '_' in dossier_name else dossier_name
        
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        in_table = False
        current_year = "All"
        for line in content.split('\n'):
            if line.startswith('### Results for '):
                current_year = line.replace('### Results for ', '').strip()
                
            if line.startswith('|') and '---' in line:
                in_table = True
                continue
            
            if line.startswith('|') and in_table:
                cols = [c.strip() for c in line.split('|')[1:-1]]
                if len(cols) >= 5 and cols[0].isdigit() or (len(cols) > 0 and '%' in cols[0]):
                    if len(cols) == 8: 
                        setup, desc, n, wr, mag, ev, ci, sig = cols
                        all_data.append(f"| {current_year} | {dossier_id} | {setup} ({desc}) | {n} | {wr} | {ev} | {ci} | {sig} |")
                    elif len(cols) >= 6:
                        setup = cols[0]
                        desc = cols[1]
                        n = cols[2]
                        sig = cols[-1]
                        ev = cols[-3]
                        wr = cols[3]
                        ci = cols[-2]
                        all_data.append(f"| {current_year} | {dossier_id} | {setup} ({desc}) | {n} | {wr} | {ev} | {ci} | {sig} |")
            elif not line.startswith('|'):
                in_table = False
                
    summary_lines.extend(sorted(all_data))
    
    summary_lines.append("")
    summary_lines.append("## Next Steps")
    summary_lines.append("Proceeding to **Phase 4: Multi-Dimensional Conditioning Sweep** to identify conditional interaction spaces (Hour-of-day, Regime, Volatility state, Event depth).")
    
    out_path = os.path.join(reports_dir, 'AG_cat_00_SWEEP_SUMMARY.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(summary_lines))
        
    print(f"Sweep Summary written to {out_path}")

if __name__ == '__main__':
    generate_sweep_summary()
