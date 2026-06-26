import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import BayesianAtlasDataset
from mamba_node import MambaInferenceNode

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from core_v2.FPS.forward_pass_system import MultiDayForwardPassSystem
from core_v2.ledger import Ledger
from core_v2.exits import default_exit_suite

def extract_expected_columns(features_dir: str, reference_date: str) -> list:
    """Extracts the exact 385 column names that the Mamba baseline was trained on."""
    print(f"[INFO] Extracting baseline expected columns from {reference_date}...")
    dataset = BayesianAtlasDataset(features_dir=features_dir, date_str=reference_date, seq_len=10)
    return dataset.expected_columns

def run_causal_forward_pass():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Booting Strictly Causal Forward Pass...")
    
    atlas_root = r"DATA\ATLAS"
    features_root = os.path.join(atlas_root, 'FEATURES_5s_v2')
    labels_csv = os.path.join(atlas_root, 'regime_labels_2d.csv')
    
    # 1. Fetch exactly the 385 dimensions the model expects
    expected_cols = extract_expected_columns(features_root, "2024_01_02")
    
    # 2. Determine Evaluation Days
    # We use the first 5 IS days (Week 1 of January 2024) to match the evaluation spec
    oos_days = [
        "2024_01_02",
        "2024_01_03",
        "2024_01_04",
        "2024_01_05",
        "2024_01_08"
    ]
    print(f"\n[INFO] Evaluating strictly causal Mamba node on IS Week 1: {oos_days}")
    
    # 3. Initialize Isolated Mamba Node
    ckpt_path = r"training\mamba_engine\mamba_checkpoint_null_control.pth"
    node = MambaInferenceNode(checkpoint_path=ckpt_path, expected_columns=expected_cols, device=device)
    
    # 4. Stream Data Causally
    fps = MultiDayForwardPassSystem(
        atlas_root=atlas_root,
        features_root=features_root,
        labels_csv=labels_csv,
        days=oos_days
    )
    
    ledger = Ledger()
    exit_suite = default_exit_suite()
    
    gross_profit = 0.0
    gross_loss = 0.0
    num_trades = 0
    
    print(f"\n[EVAL] Executing single-bar streaming...")
    
    for bar_idx, bar_state in enumerate(fps):
        if bar_idx > 0 and bar_idx % 10000 == 0:
            print(f"       Processed {bar_idx} bars... (Current Trades: {num_trades}, PnL: ${gross_profit - gross_loss:.2f})")
            
        # Step the strictly causal inference node with the dictionary of features
        action = node.step(bar_state.v2)
        
        # Action evaluates to None if the 100-bar catcher is still warming up
        if action is None:
            continue
            
        # Execute Trades
        if ledger.is_flat:
            if action in [1, 2]:
                direction = 'long' if action == 1 else 'short'
                ledger.add_position(
                    direction=direction,
                    entry_price=bar_state.price,
                    entry_ts=bar_state.timestamp,
                    entry_tier='MAMBA_BASELINE',
                    entry_features=bar_state.v2_vector
                )
        else:
            ledger.update_bar(bar_state.v2_vector, bar_state.price, bar_state.timestamp)
            pos = ledger.primary
            
            exit_reason = None
            for rule in exit_suite:
                exit_reason = rule.evaluate(bar_state, pos)
                if exit_reason:
                    break
                    
            if exit_reason:
                record = ledger.remove_position(pos.contract_id, bar_state.price, bar_state.timestamp, exit_reason)
                trade_pnl = record['pnl']
                
                num_trades += 1
                if trade_pnl > 0:
                    gross_profit += trade_pnl
                else:
                    gross_loss += abs(trade_pnl)

    # 5. Final Report
    print(f"\n========================================================")
    print(f"       STRICT CAUSAL FORWARD PASS DIAGNOSTIC REPORT")
    print(f"========================================================")
    print(f" Days Evaluated : {len(oos_days)}")
    print(f" Total Trades   : {num_trades}")
    print(f" Gross Profit   : ${gross_profit:.2f}")
    print(f" Gross Loss     : ${gross_loss:.2f}")
    
    net_pnl = gross_profit - gross_loss
    profit_factor = 999.0 if gross_loss == 0.0 else gross_profit / gross_loss
    metric_n = profit_factor - 1.0
    
    if gross_loss > 0:
        wr = (gross_profit / gross_loss) - 1
    else:
        wr = float('inf')
    
    print(f" Net PnL        : ${net_pnl:.2f}")
    print(f" Profit Factor  : {profit_factor:.4f}")
    print(f" Metric (n)     : {metric_n:.4f}")
    print(f" Bayesian WR    : {wr:.4f}")
    print(f"========================================================")


if __name__ == "__main__":
    run_causal_forward_pass()
