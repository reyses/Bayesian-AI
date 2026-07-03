import glob
import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools", "viz", "core"))
sys.path.insert(0, os.path.join(ROOT, "research", "ai_auto_labeler", "pipeline"))
sys.path.insert(0, os.path.join(ROOT, "research", "ai_auto_labeler", "tools"))

from cubic_utils import find_raw_turns
from ai_labeler_v2 import zigzag_turns
from amplitude_scale import scale_scalar, scale_series
from tune_to_human import human_days, score, TOL

ONE_M = os.path.join(ROOT, "DATA", "ATLAS", "1m")
REPORT = os.path.join(ROOT, "research", "ai_auto_labeler", "reports", "diagnose_regime_spread.md")

def main():
    days = human_days()
    cache = {dk: pd.read_parquet(os.path.join(ONE_M, f"{dk}.parquet")) for dk in days}
    
    with open(REPORT, "w") as f:
        f.write("# Regime Spread Diagnostics\n\n")
        
        # 1. Per-day regime table & mini-sweep
        f.write("## 1. Per-Day Regime Table & Best-T Sweep\n\n")
        f.write("| Date | Picks | Day Scale (pt) | Best Fixed T (F1) |\n")
        f.write("|---|---|---|---|\n")
        
        day_scales = []
        best_ts = []
        
        N = 20
        T_GRID = list(range(2, 9))
        
        for dk, hp in sorted(days.items()):
            df = cache[dk]
            close = df["close"].values
            
            day_scale = scale_scalar(close)
            day_scales.append(day_scale)
            
            # Mini-sweep
            raw_turns, smooth, _, _ = find_raw_turns(close, N)
            
            best_f1 = -1
            best_t = -1
            
            for T in T_GRID:
                piv = zigzag_turns(smooth, raw_turns, T)
                # Map pivot idx to timestamp
                # In tune_to_human, human picks are (timestamp, "LONG"/"SHORT")
                # CAND_TYPE is usually 'bottom' or 'top'. In ai_labeler_v2, cand_type='top' is SHORT, 'bottom' is LONG
                # CAND_TYPE is usually 'bottom' or 'top'. In ai_labeler_v2, cand_type='top' is SHORT, 'bottom' is LONG
                mapped_piv = []
                for p in piv:
                    ts = df["timestamp"].values[p[0]]
                    dir_str = "SHORT" if p[1] == 'top' else "LONG"
                    mapped_piv.append((ts, dir_str))
                    
                rec, prec, n_piv = score(hp, mapped_piv)
                n_h = len(hp)
                
                recall = rec / n_h if n_h > 0 else 0
                precision = prec / n_piv if n_piv > 0 else 0
                f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = T
                    
            best_ts.append(best_t)
            f.write(f"| {dk} | {len(hp)} | {day_scale:.2f} | T={best_t} (F1 {best_f1:.3f}) |\n")
            
        f.write("\n")
        
        # 2. Spearman Correlation
        f.write("## 2. Cross-Day Correlation\n\n")
        corr, pval = spearmanr(day_scales, best_ts)
        f.write(f"Spearman correlation between day scale and best-T: **{corr:.3f}** (p-value: {pval:.3f})\n\n")
        if corr > 0:
            f.write("> **Analysis**: Positive correlation found. Cross-day adaptation has signal.\n\n")
        else:
            f.write("> **Analysis**: No positive correlation. Cross-day adaptation may not help.\n\n")
            
        # 3. Intraday Spread
        f.write("## 3. Intraday Spread (RTH vs Overnight)\n\n")
        f.write("| Date | RTH Scale | ON Scale | Ratio | Human Picks in RTH |\n")
        f.write("|---|---|---|---|---|\n")
        
        for dk, hp in sorted(days.items()):
            df = cache[dk]
            close = df["close"].values
            ts = df["timestamp"].values
            
            # W60 scale
            w60_scale = scale_series(close, 60)
            
            dt = pd.to_datetime(ts, unit='s', utc=True).tz_convert('America/New_York')
            time_dt = dt.time
            # RTH: 09:30:00 to 16:00:00
            import datetime
            rth_start = datetime.time(9, 30)
            rth_end = datetime.time(16, 0)
            
            rth_mask = (time_dt >= rth_start) & (time_dt <= rth_end)
            on_mask = ~rth_mask
            
            rth_scale = np.nanmedian(w60_scale[rth_mask]) if np.sum(rth_mask) > 0 else np.nan
            on_scale = np.nanmedian(w60_scale[on_mask]) if np.sum(on_mask) > 0 else np.nan
            ratio = rth_scale / on_scale if on_scale > 0 else np.nan
            
            # Picks in RTH
            hp_ts = [p[0] for p in hp]
            hp_dt = pd.to_datetime(hp_ts, unit='s', utc=True).tz_convert('America/New_York')
            hp_time = hp_dt.time
            hp_rth = sum((hp_time >= rth_start) & (hp_time <= rth_end))
            
            f.write(f"| {dk} | {rth_scale:.2f} | {on_scale:.2f} | {ratio:.2f}x | {hp_rth}/{len(hp)} ({hp_rth/len(hp)*100:.0f}%) |\n")
            
        f.write("\n")
        
        # 4. Ratio Tightening
        f.write("## 4. Swing Ratio Tightening\n\n")
        raw_swings = []
        scaled_ratios = []
        
        for dk, hp in days.items():
            df = cache[dk]
            close = df["close"].values
            ts = df["timestamp"].values
            w60_scale = scale_series(close, 60)
            
            hp_idx = sorted(int(np.searchsorted(ts, p[0])) for p in hp)
            for a, b in zip(hp_idx, hp_idx[1:]):
                if 0 <= a < len(close) and 0 <= b < len(close):
                    raw = abs(close[b] - close[a])
                    raw_swings.append(raw)
                    
                    s_val = w60_scale[a]
                    if np.isnan(s_val) or s_val == 0:
                        s_val = scale_scalar(close) # fallback
                    if not np.isnan(s_val) and s_val > 0:
                        scaled_ratios.append(raw / s_val)
                        
        rs = np.array(raw_swings)
        sr = np.array(scaled_ratios)
        
        raw_med = np.median(rs)
        raw_iqr = np.percentile(rs, 75) - np.percentile(rs, 25)
        raw_disp = raw_iqr / raw_med
        
        sr_med = np.median(sr)
        sr_iqr = np.percentile(sr, 75) - np.percentile(sr, 25)
        sr_disp = sr_iqr / sr_med
        
        f.write(f"- **Raw Swings**: median {raw_med:.2f}, relative dispersion (IQR/median) {raw_disp:.3f}\n")
        f.write(f"- **Scaled Ratios**: median {sr_med:.2f}, relative dispersion (IQR/median) {sr_disp:.3f}\n\n")
        
        if sr_disp < raw_disp * 0.8:
            f.write("> **Analysis**: Material tightening observed (>= 20% drop in relative dispersion). Premise Supported.\n")
        elif sr_disp < raw_disp:
            f.write("> **Analysis**: Weak tightening observed. Intraday adaptation may have some signal.\n")
        else:
            f.write("> **Analysis**: No tightening observed. The spread in human swing sizes is likely not regime scaling.\n")

if __name__ == "__main__":
    main()
