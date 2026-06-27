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
        
    def forward(self, x, h=None):
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
        
        if h is None:
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
        return out, h


class MambaRLTradingNetwork(nn.Module):
    """
    Unified State-Aware Mamba-RL Trading Engine (Actor-Critic).
    Ingests Unblurred Flat Feed (8 timeframes flattened side-by-side) + Macro Sub-Encoder.
    Outputs: Policy Logits (Actor), Value Estimate (Critic), and hidden_states.
    """
    def __init__(self, sequence=30, mamba_d_model=128):
        super(MambaRLTradingNetwork, self).__init__()
        
        # 1. Unblurred Flat Feed Dimensions
        # V2 Grid provides 52 features per timeframe. 8 timeframes total.
        self.grid_flat_dim = 8 * 52  # 416
        
        # 2. Macro Sub-Encoder (5 TFs * 52 features)
        # Tensor is 5 timeframes * 52 features = 260 dim
        self.macro_encoder = nn.Sequential(
            nn.Linear(260, 64),
            nn.SiLU(),
            nn.Linear(64, 32)
        )
        
        # 3. State Injection
        # L0 (1) + Ledger State (4) + Macro Encoded (32) + Time of Day (4)
        self.mamba_input_dim = self.grid_flat_dim + 1 + 4 + 32 + 4  # 457
        
        # 4. Temporal Sequence (Mamba)
        self.input_norm = nn.LayerNorm(self.mamba_input_dim)
        self.embedding = nn.Linear(self.mamba_input_dim, mamba_d_model)
        
        self.layers = nn.ModuleList()
        for _ in range(2):
            if MAMBA_AVAILABLE:
                pass
            else:
                self.layers.append(PureMambaBlock(d_model=mamba_d_model, d_state=16, d_conv=4, expand=2))
                
        self.norm = nn.LayerNorm(mamba_d_model)
        
        # 5. PPO Heads (Actor & Critic)
        # Actor: 0=HOLD, 1=LONG, 2=SHORT, 3=SCRATCH
        self.actor_head = nn.Linear(mamba_d_model, 4)
        
        # Critic: State Value Estimate
        self.critic_head = nn.Linear(mamba_d_model, 1)

    def forward(self, v2_grid, l0_feature, ledger_state, macro_tensor, time_of_day, hidden_states=None):
        """
        v2_grid: [Batch, 8 (TFs), Seq, 40 (Features)]
        l0_feature: [Batch, Seq, 1]
        ledger_state: [Batch, Seq, 4]
        macro_tensor: [Batch, Seq, 200]
        time_of_day: [Batch, Seq, 4]
        hidden_states: list of tensors, one per Mamba layer
        """
        batch_size = v2_grid.size(0)
        seq_len = v2_grid.size(2)
        
        # --- Unblurred Flat Feed ---
        # Permute: [Batch, Seq, TFs, Features]
        x = v2_grid.permute(0, 2, 1, 3).contiguous()
        # Flatten TFs and Features: [Batch, Seq, 8 * 40] -> [Batch, Seq, 320]
        x = x.view(batch_size, seq_len, -1)
        
        # --- Macro Sub-Encoder Fusion ---
        macro_encoded = self.macro_encoder(macro_tensor) # [Batch, Seq, 32]
        
        # --- State Injection ---
        # Concatenate L0 (1) + Ledger (4) + Macro (32) + Time of Day (4): [Batch, Seq, 361]
        x = torch.cat([x, l0_feature, ledger_state, macro_encoded, time_of_day], dim=-1)
        
        # --- Input Normalization ---
        x = self.input_norm(x)
        
        # --- Mamba Temporal Pass ---
        # Project 333 -> mamba_d_model (128)
        x = self.embedding(x)
        
        next_hidden_states = []
        if hidden_states is None:
            hidden_states = [None] * len(self.layers)
            
        for i, layer in enumerate(self.layers):
            x, h = layer(x, hidden_states[i])
            next_hidden_states.append(h)
                
        x = self.norm(x)
        
        # Extract the final timestep for the decision (or keep all if needed)
        # For sequence-to-sequence TBPTT, we typically return the whole sequence.
        # But for compatibility with single-step Actor-Critic stepping, we extract the last.
        latest_step = x[:, -1, :] 
        
        # --- Output Heads (PPO) ---
        policy_logits = self.actor_head(latest_step)
        value_estimate = self.critic_head(latest_step)
        
        return policy_logits, value_estimate, next_hidden_states
