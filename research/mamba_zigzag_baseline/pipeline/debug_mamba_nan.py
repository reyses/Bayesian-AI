import sys
sys.path.append('/mnt/c/Users/reyse/OneDrive/Desktop/Bayesian-AI/research/mamba_zigzag_baseline/pipeline')
from mamba_rl_network import MambaRLTradingNetwork
import torch

def debug_mamba():
    model = MambaRLTradingNetwork(sequence=100, mamba_d_model=128).cuda()
    print("Checking initial weights for NaNs...")
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN found in parameter {name} at initialization!")

    # Create dummy input of correct shape
    v2_grid = torch.randn(1, 8, 100, 37).cuda()
    l0 = torch.randn(1, 100, 1).cuda()
    ledger = torch.randn(1, 100, 4).cuda()
    
    print("Running forward pass...")
    try:
        policy_logits, value = model(v2_grid, l0, ledger)
        print(f"policy_logits: {policy_logits}")
        print(f"value: {value}")
    except Exception as e:
        print(f"Forward pass crashed: {e}")

if __name__ == "__main__":
    debug_mamba()
