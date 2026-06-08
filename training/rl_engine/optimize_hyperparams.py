import torch
import torch.optim as optim
import optuna
import os
import sys

# Ensure imports work seamlessly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from network_research_A import ResearchANetwork
from vtrace_reconciliation import VTraceReconciliation
from curriculum_config import load_config
from train_gpu_research_A import get_available_chunks, run_quadrant_sim, evaluate_is_mastery_gate, get_dynamic_n_agents
from core_v2.FPS.forward_pass_system import MultiDayForwardPassSystem

def objective(trial):
    print(f"\n=============================================")
    print(f" [OPTUNA] Starting Trial {trial.number}")
    print(f"=============================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config()
    
    # 1. Hyperparameter Search Space
    lr = trial.suggest_float('lr', 1e-4, 5e-2, log=True)
    vtrace_rho_bar = trial.suggest_float('vtrace_rho_bar', 0.5, 5.0)
    vtrace_c_bar = trial.suggest_float('vtrace_c_bar', 0.5, 5.0)
    lstm_hidden = trial.suggest_categorical('lstm_hidden', [64, 128, 256])
    
    # Dynamic parameters for parallel VRAM scaling
    N_AGENTS = get_dynamic_n_agents(target_total_mb=10500, max_agents=48) # Cap at 48 per worker so 3 parallel workers fit safely
    epochs_per_trial = 2
    
    print(f"[OPTUNA] Testing LR={lr:.5f}, Rho={vtrace_rho_bar:.2f}, C={vtrace_c_bar:.2f}, LSTM={lstm_hidden} | Agents={N_AGENTS}")
    
    # 2. Network Initialization
    master_net = ResearchANetwork(lstm_hidden=lstm_hidden).to(device)
    optimizer = optim.Adam(master_net.parameters(), lr=lr)
    vtrace = VTraceReconciliation(rho_bar=vtrace_rho_bar, c_bar=vtrace_c_bar)
    
    # Define paths
    atlas_root = "C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS"
    features_root = os.path.join(atlas_root, 'FEATURES_5s_v2')
    labels_csv = os.path.join(atlas_root, 'regime_labels_2d.csv')
    
    chunks = get_available_chunks(features_root, chunk_size=1)
    
    if len(chunks) == 0:
        raise ValueError("No data chunks found in features directory.")
        
    train_segment = chunks[0]
        
    fps_train = MultiDayForwardPassSystem(
        atlas_root=atlas_root, features_root=features_root, labels_csv=labels_csv, days=train_segment
    )
    
    final_score = -9999.0
    
    # 3. Micro-Training Loop
    try:
        for epoch in range(1, epochs_per_trial + 1):
            is_metrics_dict = run_quadrant_sim(
                fps_train, master_net, optimizer, vtrace, config, device, 
                epoch_idx=epoch-1, is_eval=False, N_AGENTS=N_AGENTS
            )
            
            gate_passed, eff_n, raw_n, failed_conds = evaluate_is_mastery_gate(is_metrics_dict)
            
            import numpy as np
            pnls = np.array(is_metrics_dict['pnls'])
            raw_N = len(pnls)
            if raw_N > 1:
                mean_pnl = np.mean(pnls)
                stderr = np.std(pnls) / np.sqrt(raw_N)
                edge_floor_ci = mean_pnl - 1.96 * stderr
            else:
                edge_floor_ci = -9999.0
                
            final_score = float(edge_floor_ci)
            
            print(f"[OPTUNA] Trial {trial.number} | Epoch {epoch} | Edge Floor: {edge_floor_ci:.2f}")
            
            # Prune hopelessly stuck configurations early (like the 0.025 LR crash)
            if edge_floor_ci < -50.0:
                print(f"[OPTUNA] Trial {trial.number} pruned due to terrible Edge Floor.")
                raise optuna.TrialPruned()
                
    except Exception as e:
        if isinstance(e, optuna.TrialPruned):
            raise
        print(f"[ERROR] Optuna Trial failed: {e}")
        return -9999.0
        
    return final_score

if __name__ == "__main__":
    study_name = "bayesian_research_a_study"
    
    # Create SQLite database to persist study across runs/crashes
    storage_name = "sqlite:///optuna_research_a.db"
    
    study = optuna.create_study(
        study_name=study_name, 
        storage=storage_name, 
        direction="maximize", 
        load_if_exists=True
    )
    
    print(f"[INFO] Launching Optuna Hyperparameter Optimization.")
    study.optimize(objective, n_trials=50)
    
    print("\n=============================================")
    print("[OPTUNA] Optimization Complete!")
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Value (Edge Floor CI): {study.best_trial.value}")
    print("Best Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    print("=============================================")
