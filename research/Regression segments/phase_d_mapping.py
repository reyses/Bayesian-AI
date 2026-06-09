import os
import json
import torch
import numpy as np
import sys
import glob

# Add parent to path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'training', 'rl_engine'))

from training.rl_engine.train_gpu_research_A import run_quadrant_sim
from training.rl_engine.network_research_A import ResearchANetwork
from training.rl_engine.vtrace_reconciliation import VTraceReconciliation
from core_v2.FPS.forward_pass_system import MultiDayForwardPassSystem

# Month whose stage2 segments are evaluated. The forward pass is run only on
# the first available day of this set (a "quick eval"), but the day MUST be one
# whose segments are loaded below or every trade falls through to UNCLASSIFIED.
SEGMENT_GLOB = 'artifacts/stage2_segments_2025_02_*.json'

def load_segments():
    all_data = []
    files = sorted(glob.glob(SEGMENT_GLOB))
    for file in files:
        with open(file, 'r') as f:
            all_data.extend(json.load(f))
    return all_data, files

def main():
    device = torch.device('cpu')
    print(f"Using device: {device} (FORCED TO BYPASS GPU LOCK)")

    # Initialize Networks
    master_net = ResearchANetwork(lstm_hidden=128).to(device)

    # Load Weights
    ckpt_path = 'training/rl_engine/checkpoints/research_A_segment_1_latest_epoch.pth'
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        master_net.lstm.load_state_dict(checkpoint['lstm'])
        master_net.load_state_dict(checkpoint['heads'], strict=False)
    else:
        print(f"WARNING: Checkpoint {ckpt_path} not found. Running with random weights.")

    segments_data, segment_files = load_segments()
    if not segment_files:
        print(f"WARNING: No segment files matched {SEGMENT_GLOB}. Nothing to evaluate.")
        return
    # Evaluate the FIRST day actually present in the loaded segments, so the
    # trade->segment lookup below can match (was hardcoded to a January day
    # while February segments were loaded -> every trade was UNCLASSIFIED).
    available_days = sorted({seg['day'] for seg in segments_data})
    days = available_days[:1]
    print(f"Loaded {len(segments_data)} segments across {len(available_days)} days; "
          f"evaluating {days[0]} for quick evaluation.")
    
    # Initialize MultiDayForwardPassSystem
    fps = MultiDayForwardPassSystem(
        atlas_root='C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS',
        features_root='C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS/FEATURES_5s_v2',
        labels_csv='C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS/labels.csv',
        days=days
    )

    vtrace = VTraceReconciliation(rho_bar=2.0128, c_bar=3.3986)
    config = {'temperature': 0.1, 'train_steps': 1}

    print("Running deterministic evaluation (1 agent, greedy)...")
    is_metrics = run_quadrant_sim(
        fps, 
        master_net, 
        None, # optimizer
        vtrace,
        config,
        device, 
        epoch_idx=0,
        N_AGENTS=1,
        is_eval=True # This forces 0% exploration
    )

    trade_metadata = is_metrics['metadata']
    trade_pnls = is_metrics['pnls']
    trade_durations = is_metrics['durations']
    trade_mfe_avail = is_metrics['mfe_avail']
    trade_mfe_trade = is_metrics['mfe_trade']
    trade_mae = is_metrics['mae']
    daily_bar_probs = is_metrics.get('daily_bar_probs', [])

    print(f"Total Trades Taken: {len(trade_pnls)}")

    metrics_by_type = {
        'PRISTINE': {'pnl': [], 'dur': [], 'mae': []},
        'RECOVERED': {'pnl': [], 'dur': [], 'mae': []},
        'PURE_CHAOS': {'pnl': [], 'dur': [], 'mae': []},
        'UNCLASSIFIED': {'pnl': [], 'dur': [], 'mae': []}
    }

    metrics_by_vol = {
        'EXTREME': {'pnl': [], 'dur': [], 'mae': []},
        'HIGH': {'pnl': [], 'dur': [], 'mae': []},
        'MEDIUM': {'pnl': [], 'dur': [], 'mae': []},
        'LOW': {'pnl': [], 'dur': [], 'mae': []},
        'UNCLASSIFIED': {'pnl': [], 'dur': [], 'mae': []}
    }

    prob_metrics_by_type = {
        'PRISTINE': {'sum_hold': 0.0, 'sum_long': 0.0, 'sum_short': 0.0, 'bars': 0},
        'RECOVERED': {'sum_hold': 0.0, 'sum_long': 0.0, 'sum_short': 0.0, 'bars': 0},
        'PURE_CHAOS': {'sum_hold': 0.0, 'sum_long': 0.0, 'sum_short': 0.0, 'bars': 0},
        'UNCLASSIFIED': {'sum_hold': 0.0, 'sum_long': 0.0, 'sum_short': 0.0, 'bars': 0}
    }

    prob_metrics_by_vol = {
        'EXTREME': {'sum_hold': 0.0, 'sum_long': 0.0, 'sum_short': 0.0, 'bars': 0},
        'HIGH': {'sum_hold': 0.0, 'sum_long': 0.0, 'sum_short': 0.0, 'bars': 0},
        'MEDIUM': {'sum_hold': 0.0, 'sum_long': 0.0, 'sum_short': 0.0, 'bars': 0},
        'LOW': {'sum_hold': 0.0, 'sum_long': 0.0, 'sum_short': 0.0, 'bars': 0},
        'UNCLASSIFIED': {'sum_hold': 0.0, 'sum_long': 0.0, 'sum_short': 0.0, 'bars': 0}
    }

    # First, map the trades
    for i in range(len(trade_pnls)):
        day, entry_bar, exit_bar, agent_dir = trade_metadata[i]
        pnl = trade_pnls[i]
        dur = trade_durations[i]
        mae = trade_mae[i]

        day_segments = [seg for seg in segments_data if seg['day'] == day]
        
        trade_status = 'UNCLASSIFIED'
        trade_vol = 'UNCLASSIFIED'

        for seg in day_segments:
            s_idx = seg.get('raw_start_idx', seg['start_idx'])
            e_idx = seg.get('raw_end_idx', seg['end_idx'])
            if s_idx <= entry_bar <= e_idx:
                trade_status = seg['status']
                vol_tier_raw = seg.get('volatility_tier', 'UNCLASSIFIED')
                if isinstance(vol_tier_raw, int):
                    if vol_tier_raw <= 3:
                        trade_vol = 'LOW'
                    elif vol_tier_raw <= 6:
                        trade_vol = 'MEDIUM'
                    elif vol_tier_raw <= 9:
                        trade_vol = 'HIGH'
                    else:
                        trade_vol = 'EXTREME'
                else:
                    trade_vol = vol_tier_raw
                break
        
        metrics_by_type[trade_status]['pnl'].append(pnl)
        metrics_by_type[trade_status]['dur'].append(dur)
        metrics_by_type[trade_status]['mae'].append(mae)

        metrics_by_vol[trade_vol]['pnl'].append(pnl)
        metrics_by_vol[trade_vol]['dur'].append(dur)
        metrics_by_vol[trade_vol]['mae'].append(mae)

    # Next, map the per-bar probabilities
    for dp in daily_bar_probs:
        day = dp['day']
        probs = dp['probs']  # Shape [N, 3]
        start_idx = dp['start_idx']
        
        day_segments = [seg for seg in segments_data if seg['day'] == day]
        for seg in day_segments:
            s_idx = seg.get('raw_start_idx', seg['start_idx'])
            e_idx = seg.get('raw_end_idx', seg['end_idx'])
            status = seg['status']
            vol_tier_raw = seg.get('volatility_tier', 'UNCLASSIFIED')
            
            trade_vol = 'UNCLASSIFIED'
            if isinstance(vol_tier_raw, int):
                if vol_tier_raw <= 3: trade_vol = 'LOW'
                elif vol_tier_raw <= 6: trade_vol = 'MEDIUM'
                elif vol_tier_raw <= 9: trade_vol = 'HIGH'
                else: trade_vol = 'EXTREME'
            else:
                trade_vol = vol_tier_raw

            p_start = max(0, s_idx - start_idx)
            p_end = min(len(probs), e_idx - start_idx + 1)
            
            if p_start < p_end:
                segment_probs = probs[p_start:p_end]
                # segment_probs is [K, 3]
                sum_h = float(segment_probs[:, 0].sum())
                sum_l = float(segment_probs[:, 1].sum())
                sum_s = float(segment_probs[:, 2].sum())
                k_bars = p_end - p_start
                
                prob_metrics_by_type[status]['sum_hold'] += sum_h
                prob_metrics_by_type[status]['sum_long'] += sum_l
                prob_metrics_by_type[status]['sum_short'] += sum_s
                prob_metrics_by_type[status]['bars'] += k_bars
                
                prob_metrics_by_vol[trade_vol]['sum_hold'] += sum_h
                prob_metrics_by_vol[trade_vol]['sum_long'] += sum_l
                prob_metrics_by_vol[trade_vol]['sum_short'] += sum_s
                prob_metrics_by_vol[trade_vol]['bars'] += k_bars

    report_lines = []
    report_lines.append("="*50)
    report_lines.append("PHASE D: NEURAL NETWORK PERFORMANCE BY CHAOS TYPE")
    report_lines.append("="*50)

    for status, data in metrics_by_type.items():
        count = len(data['pnl'])
        p_data = prob_metrics_by_type[status]
        bars = p_data['bars']
        
        avg_h = p_data['sum_hold'] / bars if bars > 0 else 0.0
        avg_l = p_data['sum_long'] / bars if bars > 0 else 0.0
        avg_s = p_data['sum_short'] / bars if bars > 0 else 0.0
        
        report_lines.append(f"\n[{status}] (Trades: {count} | Bars: {bars})")
        report_lines.append(f"  Neural P(Hold):  {avg_h*100:5.2f}%")
        report_lines.append(f"  Neural P(Long):  {avg_l*100:5.2f}%")
        report_lines.append(f"  Neural P(Short): {avg_s*100:5.2f}%")

        if count == 0:
            continue
        
        pnls = np.array(data['pnl'])
        
        # Mandatory Profit-Factor-Based Trade WR
        winners = pnls[pnls > 0]
        losers = pnls[pnls <= 0]
        gross_profit = winners.sum() if len(winners) > 0 else 0
        gross_loss = np.abs(losers.sum()) if len(losers) > 0 else 0
        
        if gross_loss == 0:
            trade_wr = float('inf') if gross_profit > 0 else 0.0
        else:
            trade_wr = (gross_profit / gross_loss) - 1.0

        avg_pnl = pnls.mean()
        avg_dur = np.mean(data['dur'])
        avg_mae = np.mean(data['mae'])

        report_lines.append(f"  Trade WR (PF-based): {trade_wr:+.2f}")
        report_lines.append(f"  Avg PnL:  {avg_pnl:.5f}")
        report_lines.append(f"  Avg Dur:  {avg_dur:.1f} bars")
        report_lines.append(f"  Avg MAE:  {avg_mae:.5f}")

    report_lines.append("\n" + "="*50)
    report_lines.append("PERFORMANCE BY VOLATILITY TIER")
    report_lines.append("="*50)

    for vol, data in metrics_by_vol.items():
        count = len(data['pnl'])
        p_data = prob_metrics_by_vol[vol]
        bars = p_data['bars']
        
        avg_h = p_data['sum_hold'] / bars if bars > 0 else 0.0
        avg_l = p_data['sum_long'] / bars if bars > 0 else 0.0
        avg_s = p_data['sum_short'] / bars if bars > 0 else 0.0
        
        report_lines.append(f"\n[{vol}] (Trades: {count} | Bars: {bars})")
        report_lines.append(f"  Neural P(Hold):  {avg_h*100:5.2f}%")
        report_lines.append(f"  Neural P(Long):  {avg_l*100:5.2f}%")
        report_lines.append(f"  Neural P(Short): {avg_s*100:5.2f}%")

        if count == 0:
            continue
        
        pnls = np.array(data['pnl'])
        
        winners = pnls[pnls > 0]
        losers = pnls[pnls <= 0]
        gross_profit = winners.sum() if len(winners) > 0 else 0
        gross_loss = np.abs(losers.sum()) if len(losers) > 0 else 0
        
        if gross_loss == 0:
            trade_wr = float('inf') if gross_profit > 0 else 0.0
        else:
            trade_wr = (gross_profit / gross_loss) - 1.0

        avg_pnl = pnls.mean()
        avg_dur = np.mean(data['dur'])
        avg_mae = np.mean(data['mae'])

        report_lines.append(f"  Trade WR (PF-based): {trade_wr:+.2f}")
        report_lines.append(f"  Avg PnL:  {avg_pnl:.5f}")
        report_lines.append(f"  Avg Dur:  {avg_dur:.1f} bars")
        report_lines.append(f"  Avg MAE:  {avg_mae:.5f}")

    output_str = "\n".join(report_lines)
    print(output_str)

    os.makedirs('reports/findings', exist_ok=True)
    with open('reports/findings/phase_d_february_results.md', 'w') as f:
        f.write("# Phase D: February 2025 Evaluation\n\n")
        f.write("```text\n")
        f.write(output_str)
        f.write("\n```\n")
    print(f"\n[RUNNER] Report saved to reports/findings/phase_d_february_results.md")

if __name__ == "__main__":
    main()
