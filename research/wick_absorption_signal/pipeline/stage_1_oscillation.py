import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter, detrend, get_window, periodogram, find_peaks
from scipy.stats import linregress

RAW_DATA_PATH = "DATA/ATLAS/order_flow_delta_5s.parquet"
OUTPUT_DIR = Path("research/wick_absorption_signal/reports/stage_1_oscillation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TFS = {
    '5s': '5s',
    '15s': '15s',
    '1m': '1min',
    '5m': '5min',
    '15m': '15min',
    '1h': '1h'
}

TF_MINUTES = {
    '5s': 5/60.0,
    '15s': 15/60.0,
    '1m': 1.0,
    '5m': 5.0,
    '15m': 15.0,
    '1h': 60.0
}

N_SWEEP = [5, 8, 12, 20, 40]

def phase_randomize(x):
    X = np.fft.rfft(x)
    phases = np.random.uniform(0, 2*np.pi, len(X))
    X_rand = np.abs(X) * np.exp(1j * phases)
    return np.fft.irfft(X_rand, n=len(x))

def run_stage_1_oscillation():
    print("Loading raw 5s data...")
    df_raw = pd.read_parquet(RAW_DATA_PATH).sort_index()
    
    reconciliation_rows = []
    
    for tf_label, tf_pandas in TFS.items():
        print(f"Processing TF: {tf_label}...")
        df = df_raw.resample(tf_pandas).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()
        
        close = df['close'].values
        tf_min = TF_MINUTES[tf_label]
        
        # ---------------------------------------------------------
        # METHOD 1: Centered Cubic Sweep
        # ---------------------------------------------------------
        sweep_results_bars = []
        sweep_results_mins = []
        event_counts = []
        
        for N in N_SWEEP:
            # Drop SG filter edges explicitly
            trim = N // 2
            
            # 1st deriv (slope)
            slope = savgol_filter(close, window_length=N, polyorder=3, deriv=1, mode='nearest')
            slope[:trim] = np.nan
            slope[-trim:] = np.nan
            
            sign_slope = np.sign(slope)
            # Find zero crossings where slope changes sign
            # Valid crossings only
            valid = ~np.isnan(sign_slope)
            diffs = np.diff(sign_slope[valid])
            turns = np.where(diffs != 0)[0]
            
            event_count = len(turns)
            event_counts.append(event_count)
            
            if event_count > 1:
                # Average distance between turns = half-cycle. Full cycle = 2 * half-cycle
                period_bars = 2.0 * np.mean(np.diff(turns))
            else:
                period_bars = np.nan
                
            sweep_results_bars.append(period_bars)
            sweep_results_mins.append(period_bars * tf_min if not np.isnan(period_bars) else np.nan)
        
        # Quantitative Plateau Check (Regress Period in BARS vs N in BARS)
        valid_idx = ~np.isnan(sweep_results_bars)
        if np.sum(valid_idx) > 2:
            slope_bars, _, _, _, _ = linregress(np.array(N_SWEEP)[valid_idx], np.array(sweep_results_bars)[valid_idx])
        else:
            slope_bars = np.nan
            
        is_plateau = slope_bars < 0.3 # Less than 0.3 bar increase per window bar
        cubic_plateau_mins = np.nanmean(sweep_results_mins) if is_plateau else np.nan
        cubic_plateau_str = f"{cubic_plateau_mins:.2f}" if is_plateau else "Artifact"
        
        # Plot Cubic Sweep
        plt.figure(figsize=(10, 5))
        plt.plot(N_SWEEP, sweep_results_mins, marker='o', linestyle='-', color='blue')
        plt.title(f"Cubic Sweep: Detected Period vs Window Size N ({tf_label})\nRegression Slope (Bars): {slope_bars:.2f}")
        plt.xlabel("Centered Window Size N (bars)")
        plt.ylabel("Detected Full Cycle Period (minutes)")
        plt.grid(True, alpha=0.3)
        plt.savefig(OUTPUT_DIR / f"{tf_label}_cubic_sweep.png")
        plt.close()
        
        # ---------------------------------------------------------
        # METHOD 2: Fourier Periodogram & ACF
        # ---------------------------------------------------------
        # Log Returns
        log_rets = np.log(df['close'] / df['close'].shift(1)).dropna().values
        detrended_rets = detrend(log_rets, type='linear')
        
        # Fast ACF using FFT
        n = len(detrended_rets)
        n_fft = 2 ** int(np.ceil(np.log2(2 * n - 1)))
        X = np.fft.rfft(detrended_rets, n=n_fft)
        acf = np.fft.irfft(X * np.conj(X))[:n]
        acf = acf / acf[0] if acf[0] != 0 else acf
        
        troughs, _ = find_peaks(-acf)
        if len(troughs) > 0:
            acf_trough_bars = troughs[0]
            acf_trough_mins = acf_trough_bars * tf_min
        else:
            acf_trough_mins = np.nan
            
        hann_win = get_window('hann', len(detrended_rets))
        win_rets = detrended_rets * hann_win
        
        freqs, psd = periodogram(win_rets, fs=1.0)
        
        surr = phase_randomize(detrended_rets)
        surr_win = surr * hann_win
        _, psd_surr = periodogram(surr_win, fs=1.0)
        
        # Smooth PSDs lightly for visualization to identify robust peaks
        psd_smooth = pd.Series(psd).rolling(5, center=True).mean().values
        psd_surr_smooth = pd.Series(psd_surr).rolling(5, center=True).mean().values
        
        # Find dominant peak in raw PSD
        dom_idx = np.argmax(psd_smooth[1:]) + 1
        peak_freq = freqs[dom_idx]
        spectral_period_bars = 1.0 / peak_freq if peak_freq > 0 else np.nan
        spectral_period_mins = spectral_period_bars * tf_min if not np.isnan(spectral_period_bars) else np.nan
        
        # Check if peak is above surrogate (broadband red noise check)
        is_narrowband = psd_smooth[dom_idx] > (psd_surr_smooth[dom_idx] * 1.5)
        if not is_narrowband:
            spectral_period_mins = np.nan # It's just broadband noise
            
        spectral_str = f"{spectral_period_mins:.2f}" if not np.isnan(spectral_period_mins) else "Broadband"
        
        # Plot Periodogram
        plt.figure(figsize=(10, 5))
        # Convert frequencies to periods in minutes for X-axis (ignoring 0)
        valid_f = freqs > 0
        periods_min = (1.0 / freqs[valid_f]) * tf_min
        
        plt.plot(periods_min, psd_smooth[valid_f], label='Log-Returns PSD', color='black')
        plt.plot(periods_min, psd_surr_smooth[valid_f], label='Surrogate (Red Noise)', color='red', alpha=0.6, linestyle='--')
        plt.axvline(x=spectral_period_mins if is_narrowband else 0, color='blue', linestyle=':', label='Dominant Peak')
        plt.title(f"Fourier Periodogram vs Phase-Randomized Surrogate ({tf_label})")
        plt.xlabel("Period (minutes)")
        plt.ylabel("Power Spectral Density")
        plt.xlim(0, max(sweep_results_mins) * 2 if max(sweep_results_mins) > 0 else 60)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(OUTPUT_DIR / f"{tf_label}_periodogram.png")
        plt.close()
        
        # ---------------------------------------------------------
        # RECONCILIATION
        # ---------------------------------------------------------
        # Determine verdict
        if is_plateau and is_narrowband:
            # Do they agree? (within 20%)
            if abs(cubic_plateau_mins - spectral_period_mins) / spectral_period_mins < 0.20:
                verdict = "AGREEMENT (Real Cadence)"
            else:
                verdict = "DISAGREEMENT (Trust Spectrum)"
        else:
            verdict = "TURN-RELATIVE (No Cadence)"
            
        reconciliation_rows.append({
            'TF': tf_label,
            'Event Count (N=40)': event_counts[-1],
            'Cubic Plateau (Mins)': cubic_plateau_str,
            'Spectral Dominant (Mins)': spectral_str,
            'ACF Trough (Mins)': f"{acf_trough_mins:.2f}" if not np.isnan(acf_trough_mins) else "None",
            'Artifact Slope (Bars)': f"{slope_bars:.2f}",
            'Verdict': verdict
        })
    
    # Generate Output Markdown
    df_recon = pd.DataFrame(reconciliation_rows)
    
    def df_to_md_table(df):
        headers = " | ".join(df.columns)
        sep = " | ".join(["---"] * len(df.columns))
        rows = []
        for _, row in df.iterrows():
            rows.append(" | ".join(str(x) for x in row.values))
        return f"| {headers} |\n| {sep} |\n" + "\n".join(f"| {r} |" for r in rows)

    summary_md = f"""# Stage 1 (Corrected): True Oscillation Reconciliation

**Goal**: Find the real oscillation structure per TF *without imposing a frequency*.

## Reconciliation Table
{df_to_md_table(df_recon)}

## Details per Timeframe
"""

    for tf in TFS.keys():
        summary_md += f"\n### {tf} Analysis\n"
        summary_md += f"![{tf} Cubic Sweep](./{tf}_cubic_sweep.png)\n"
        summary_md += f"![{tf} Periodogram](./{tf}_periodogram.png)\n"

    with open(OUTPUT_DIR / "reconciliation_table.md", "w") as f:
        f.write(summary_md)

    print("Stage 1 Corrected complete. Saved to reports/stage_1_oscillation/reconciliation_table.md")

if __name__ == "__main__":
    run_stage_1_oscillation()
