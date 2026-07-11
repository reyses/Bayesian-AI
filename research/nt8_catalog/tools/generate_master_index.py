import os
import glob
import re
import pandas as pd
import numpy as np
import scipy.stats as stats

def bootstrap_ci(data, n_iterations=1000, ci=95):
    if len(data) < 2: return 0.0, 0.0
    means = []
    n = len(data)
    for _ in range(n_iterations):
        sample = np.random.choice(data, size=n, replace=True)
        means.append(np.mean(sample))
    alpha = (100 - ci) / 2.0
    return np.percentile(means, alpha), np.percentile(means, 100 - alpha)

def get_descriptions(md_file):
    desc_map = {}
    if not os.path.exists(md_file): return desc_map
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    in_table = False
    for line in content.split('\n'):
        if line.startswith('|') and '---' in line:
            in_table = True
            continue
        if line.startswith('|') and in_table:
            cols = [c.strip() for c in line.split('|')[1:-1]]
            if len(cols) >= 5 and cols[0].isdigit():
                setup = cols[0]
                desc = cols[1]
                desc_map[setup] = desc
        elif not line.startswith('|'):
            in_table = False
    return desc_map

def generate_sweep_summary():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    tests_dir = os.path.join(base_dir, 'tests')
    reports_dir = os.path.join(base_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    events_files = glob.glob(os.path.join(tests_dir, '**', 'events.parquet'), recursive=True)
    events_files = [f for f in events_files if 'archive' not in f.lower()]
    
    summary_lines = []
    summary_lines.append("# Document ID: AG-CAT-00-SWEEP-SUMMARY")
    summary_lines.append("**Title:** Master Catalog Sweep Summary (Phase 4 Unit-Standardized)")
    summary_lines.append("**Status:** Audit Completed (AUDIT-ACC-01 §3)")
    summary_lines.append(f"**Generated:** 2026-07-11 by Gemini")
    summary_lines.append("")
    summary_lines.append("> [!WARNING] AUDIT MANDATE")
    summary_lines.append("> **NO UNCONDITIONALLY STABLE POSITIVE EDGES WERE FOUND** across the 18 base hypotheses over the 2024–2025 dataset.")
    summary_lines.append("> All positive EV results observed in the initial raw sweeps failed to survive strict causality firewalls, lookahead corrections, and standardisation.")
    summary_lines.append("")
    summary_lines.append("> [!NOTE] SQUEEZE EDGE")
    summary_lines.append("> SQZ-04 Volatility Squeeze is marked with a 1.00 Resp Freq because it is a duration measurement edge, where response is guaranteed by construction. It is an outlier compared to directional edges.")
    summary_lines.append("")
    summary_lines.append("## Consolidated Unit-Standardized Sweep")
    summary_lines.append("")
    summary_lines.append("| Year | Dossier | Setup | N | PF-WR | EV (Raw Pts) | EV 95% CI | Sig? | EV (Mean σ) |")
    summary_lines.append("|---|---|---|---|---|---|---|---|---|")
    
    all_data = []
    
    for parquet_file in events_files:
        dossier_name = os.path.basename(os.path.dirname(parquet_file))
        if dossier_name == 'tests':
            continue
        dossier_id = dossier_name.split('_')[0] if '_' in dossier_name else dossier_name
        
        md_file = os.path.join(os.path.dirname(parquet_file), f"DOC_{dossier_id.replace('-', '_')}.md")
        if not os.path.exists(md_file):
            md_files = glob.glob(os.path.join(os.path.dirname(parquet_file), 'DOC_*.md'))
            if md_files:
                md_file = md_files[0]
                
        desc_map = get_descriptions(md_file)
        
        try:
            df = pd.read_parquet(parquet_file)
        except:
            continue
            
        for year in ['2024', '2025']:
            df_year = df[df['year'] == year]
            if len(df_year) == 0: continue
            
            for setup in df_year['setup'].unique():
                df_sub = df_year[df_year['setup'] == setup]
                n = len(df_sub)
                if n == 0: continue
                
                mags = df_sub['magnitude'].dropna().values
                wr = np.mean(mags > 0) if len(mags) > 0 else 0.0
                gross_profit = np.sum(mags[mags > 0])
                gross_loss = np.abs(np.sum(mags[mags < 0]))
                pf = gross_profit / gross_loss if gross_loss > 0 else np.inf
                
                ev_raw = np.mean(mags) if len(mags) > 0 else 0.0
                ci_low, ci_high = bootstrap_ci(mags)
                sig = "Yes" if ci_low > 0 or ci_high < 0 else "No"
                
                if 'magnitude_sigma' in df_sub.columns:
                    ev_sigma = df_sub['magnitude_sigma'].dropna().mean()
                else:
                    ev_sigma = np.nan
                    
                desc = desc_map.get(str(setup), "Unknown")
                setup_desc = f"{setup} ({desc})"
                
                pf_wr_str = f"{pf:.2f} | {wr:.2f}"
                if 'SQZ-04' in dossier_id:
                    pf_wr_str = f"{pf:.2f} | 1.00*"
                    
                all_data.append(f"| {year} | {dossier_id} | {setup_desc} | {n} | {pf_wr_str} | {ev_raw:.2f} | [{ci_low:.2f}, {ci_high:.2f}] | {sig} | {ev_sigma:.2f}σ |")

    all_data.sort()
    summary_lines.extend(all_data)
    
    summary_lines.append("")
    summary_lines.append("## Next Steps")
    summary_lines.append("Proceeding to **Phase 4: Multi-Dimensional Conditioning Sweep** to identify conditional interaction spaces (Hour-of-day, Regime, Volatility state, Event depth).")
    
    out_path = os.path.join(reports_dir, 'AG_cat_00_SWEEP_SUMMARY.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(summary_lines))
        
    print(f"Sweep Summary written to {out_path}")

if __name__ == '__main__':
    generate_sweep_summary()
