import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

# Hardcoding to Pure-PyTorch Mamba block since Triton is unavailable on Windows
MAMBA_AVAILABLE = False

class PureMambaBlock(nn.Module):
    """A minimal pure-PyTorch implementation of the Mamba block."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        dt_rank = max(int(d_model / 16), 1)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + dt_rank, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)
        
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
    def forward(self, x):
        B, L, D = x.shape
        x_and_res = self.in_proj(x)
        x_m, res = x_and_res.split(self.d_inner, dim=-1)
        
        x_m = x_m.transpose(1, 2)
        x_m = self.conv1d(x_m)[:, :, :L]
        x_m = x_m.transpose(1, 2)
        x_m = F.silu(x_m)
        
        x_proj_out = self.x_proj(x_m)
        dt, B_param, C_param = torch.split(x_proj_out, [self.dt_proj.in_features, self.d_state, self.d_state], dim=-1)
        
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)
        
        h = torch.zeros((B, self.d_inner, self.d_state), device=x.device)
        y = []
        for t in range(L):
            dt_t = dt[:, t, :].unsqueeze(-1)
            A_t = torch.exp(dt_t * A)
            B_t = B_param[:, t, :].unsqueeze(1)
            dB_t = dt_t * B_t
            x_t = x_m[:, t, :].unsqueeze(-1)
            h = A_t * h + dB_t * x_t
            C_t = C_param[:, t, :].unsqueeze(1)
            y_t = torch.sum(h * C_t, dim=-1)
            y.append(y_t)
            
        y = torch.stack(y, dim=1)
        y = y + x_m * self.D
        y = y * F.silu(res)
        out = self.out_proj(y)
        return out


class MambaRLTradingNetwork(nn.Module):
    """
    Unified State-Aware Mamba-RL Trading Engine (Actor-Critic).
    Ingests Unblurred Flat Feed (8 timeframes flattened side-by-side).
    Outputs: Policy Logits (Actor) and Value Estimate (Critic).
    """
    def __init__(self, sequence=30, mamba_d_model=128):
        super(MambaRLTradingNetwork, self).__init__()
        
        # 1. Unblurred Flat Feed Dimensions
        # V2 Grid provides 37 features per timeframe. 8 timeframes total.
        self.grid_flat_dim = 8 * 37  # 296
        
        # 2. State Injection
        # L0 (1) + Ledger State (4)
        self.mamba_input_dim = self.grid_flat_dim + 1 + 4  # 301
        
        # 3. Temporal Sequence (Mamba)
        self.embedding = nn.Linear(self.mamba_input_dim, mamba_d_model)
        
        self.layers = nn.ModuleList()
        for _ in range(2):
            if MAMBA_AVAILABLE:
                self.layers.append(
                    Mamba(
                        d_model=mamba_d_model,
                        d_state=16,
                        d_conv=4,
                        expand=2
                    )
                )
            else:
                self.layers.append(PureMambaBlock(d_model=mamba_d_model, d_state=16, d_conv=4, expand=2))
                
        self.norm = nn.LayerNorm(mamba_d_model)
        
        # 4. PPO Heads (Actor & Critic)
        # Actor: 0=HOLD, 1=LONG, 2=SHORT, 3=SCRATCH
        self.actor_head = nn.Linear(mamba_d_model, 4)
        
        # Critic: State Value Estimate
        self.critic_head = nn.Linear(mamba_d_model, 1)

    def forward(self, v2_grid, l0_feature, ledger_state):
        """
        v2_grid: [Batch, 8 (TFs), 30 (Seq), 37 (Features)]
        l0_feature: [Batch, 30 (Seq), 1]
        ledger_state: [Batch, 30 (Seq), 4]
        """
        batch_size = v2_grid.size(0)
        seq_len = v2_grid.size(2)
        
        # --- Unblurred Flat Feed ---
        # Permute: [Batch, Seq, TFs, Features]
        x = v2_grid.permute(0, 2, 1, 3).contiguous()
        # Flatten TFs and Features: [Batch, Seq, 8 * 37] -> [Batch, Seq, 296]
        x = x.view(batch_size, seq_len, -1)
        
        # --- State Injection ---
        # Concatenate L0 (1) + Ledger (4): [Batch, Seq, 301]
        x = torch.cat([x, l0_feature, ledger_state], dim=-1)
        
        # --- Mamba Temporal Pass ---
        # Project 301 -> mamba_d_model (128)
        x = self.embedding(x)
        
        for layer in self.layers:
            x = layer(x)
                
        x = self.norm(x)
        
        # Extract the final timestep for the decision
        latest_step = x[:, -1, :] 
        
        # --- Output Heads (PPO) ---
        policy_logits = self.actor_head(latest_step)
        value_estimate = self.critic_head(latest_step)
        
        return policy_logits, value_estimate
