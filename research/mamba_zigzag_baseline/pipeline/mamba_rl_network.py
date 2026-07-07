import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from mamba_scan import associative_ssm_scan

logger = logging.getLogger(__name__)

# Try to import official mamba-ssm
try:
    from mamba_ssm import Mamba
    try:
        from mamba_ssm.utils.generation import InferenceParams
    except ImportError:
        # Fallback if InferenceParams moved
        pass
    # FORCED off for the TRAINING path: Mamba.step() uses inference-only fused
    # kernels (causal_conv1d_update / selective_state_update) with NO autograd
    # backward, so the per-bar L=1 recurrence would sever the TBPTT gradient
    # through the carried hidden state. Evidence: tools/perf_mamba_ssm_probe.py.
    # PureMambaBlock's explicit scan is differentiable; keep it for training.
    MAMBA_AVAILABLE = False
except ImportError:
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
        
    @torch.compiler.disable
    def _run_conv1d(self, x_m, L):
        x_m = x_m.transpose(1, 2).contiguous()
        x_m = self.conv1d(x_m)[:, :, :L]
        x_m = x_m.transpose(1, 2).contiguous()
        return x_m

    def forward(self, x, h=None):
        B, L, D = x.shape
        x_and_res = self.in_proj(x)
        x_m, res = x_and_res.split(self.d_inner, dim=-1)
        
        x_m = self._run_conv1d(x_m, L)
        x_m = F.silu(x_m)
        
        x_proj_out = self.x_proj(x_m)
        dt, B_param, C_param = torch.split(x_proj_out, [self.dt_proj.in_features, self.d_state, self.d_state], dim=-1)
        
        dt = F.softplus(self.dt_proj(dt)).contiguous()
        B_param = B_param.contiguous()
        C_param = C_param.contiguous()
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

    # ── Sequence-window training paths (docs/JULES_SEQUENCE_WINDOW_TRAINING.md) ──
    # NOTE: forward() above is left untouched — the per-bar trainer's bitwise
    # baseline depends on it (its L=1 path has NO conv memory). The two methods
    # below restore the causal d_conv-1-bar receptive field and match each
    # other numerically (verified by tools/test_seq_equivalence.py).

    def step(self, x, h=None, conv_state=None):
        """L=1 recurrent step WITH carried conv state. Acting-pass twin of
        forward_sequence. Returns (out [B,1,D], h, conv_state)."""
        B, L, _ = x.shape
        x_and_res = self.in_proj(x)
        x_m, res = x_and_res.split(self.d_inner, dim=-1)  # [B, 1, d_inner]

        if conv_state is None:
            conv_state = torch.zeros(B, self.d_conv - 1, self.d_inner,
                                     device=x.device, dtype=x_m.dtype)
        conv_in = torch.cat([conv_state.to(x_m.dtype), x_m], dim=1)  # [B, d_conv, d_inner]
        new_conv_state = conv_in[:, 1:]
        x_c = F.conv1d(conv_in.transpose(1, 2), self.conv1d.weight,
                       self.conv1d.bias, groups=self.d_inner)  # [B, d_inner, 1]
        x_c = F.silu(x_c.transpose(1, 2))  # [B, 1, d_inner]

        x_proj_out = self.x_proj(x_c)
        dt, B_param, C_param = torch.split(
            x_proj_out, [self.dt_proj.in_features, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # [B, 1, d_inner]
        A = -torch.exp(self.A_log)

        if h is None:
            h = torch.zeros(B, self.d_inner, self.d_state,
                            device=x.device, dtype=x_c.dtype)
        dt_t = dt[:, 0, :].unsqueeze(-1)                     # [B, d_inner, 1]
        A_t = torch.exp(dt_t * A)                            # [B, d_inner, d_state]
        h = A_t * h + (dt_t * B_param[:, 0].unsqueeze(1)) * x_c[:, 0].unsqueeze(-1)
        y = torch.sum(h * C_param[:, 0].unsqueeze(1), dim=-1).unsqueeze(1)  # [B, 1, d_inner]

        y = y + x_c * self.D
        y = y * F.silu(res)
        return self.out_proj(y), h, new_conv_state

    def forward_sequence(self, x, h0=None, conv_state0=None):
        """Differentiable window forward: causal conv over carried context +
        log-depth associative SSM scan with initial state. Returns
        (out [B,L,D], h_last, conv_state_last)."""
        B, L, _ = x.shape
        x_and_res = self.in_proj(x)
        x_m, res = x_and_res.split(self.d_inner, dim=-1)  # [B, L, d_inner]

        if conv_state0 is None:
            conv_state0 = torch.zeros(B, self.d_conv - 1, self.d_inner,
                                      device=x.device, dtype=x_m.dtype)
        conv_in = torch.cat([conv_state0.to(x_m.dtype), x_m], dim=1)  # [B, L+3, d_inner]
        new_conv_state = conv_in[:, -(self.d_conv - 1):]
        x_c = F.conv1d(conv_in.transpose(1, 2), self.conv1d.weight,
                       self.conv1d.bias, groups=self.d_inner)  # [B, d_inner, L]
        x_c = F.silu(x_c.transpose(1, 2))  # [B, L, d_inner]

        x_proj_out = self.x_proj(x_c)
        dt, B_param, C_param = torch.split(
            x_proj_out, [self.dt_proj.in_features, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # [B, L, d_inner]
        A = -torch.exp(self.A_log)

        A_t = torch.exp(dt.unsqueeze(-1) * A)                        # [B, L, d_inner, d_state]
        b_t = (dt * x_c).unsqueeze(-1) * B_param.unsqueeze(2)        # [B, L, d_inner, d_state]
        h_seq = associative_ssm_scan(A_t, b_t, h0)
        y = (h_seq * C_param.unsqueeze(2)).sum(-1)                   # [B, L, d_inner]

        y = y + x_c * self.D
        y = y * F.silu(res)
        return self.out_proj(y), h_seq[:, -1], new_conv_state


class MambaRLTradingNetwork(nn.Module):
    """
    Unified State-Aware Mamba-RL Trading Engine (Actor-Critic).
    Ingests Unblurred Flat Feed (8 timeframes * 52 features = 416) + Macro Sub-Encoder (5 timeframes * 52 features = 260).
    Outputs: Policy Logits (Actor), Value Estimate (Critic), and hidden_states.
    """
    def __init__(self, sequence=30, mamba_d_model=128):
        super(MambaRLTradingNetwork, self).__init__()
        
        # 1. Unblurred Flat Feed Dimensions
        # V2 Grid provides 416 features per timeframe sequence.
        self.grid_flat_dim = 8 * 52  # 416
        
        # 2. Macro Sub-Encoder (5 TFs * 52 features + 1 validity mask)
        # Tensor is 261 dim
        self.macro_encoder = nn.Sequential(
            nn.Linear(261, 64),
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
                self.layers.append(Mamba(d_model=mamba_d_model, d_state=16, d_conv=4, expand=2))
            else:
                self.layers.append(PureMambaBlock(d_model=mamba_d_model, d_state=16, d_conv=4, expand=2))
                
        self.norm = nn.LayerNorm(mamba_d_model)
        
        # 5. PPO Heads (Actor & Critic)
        # Entry Actor: 0=HOLD, 1=LONG, 2=SHORT (active when flat)
        self.entry_head = nn.Linear(mamba_d_model, 3)
        
        # Exit Hazard Actor: 1-way logit (probability of exit vs hold, active when in position)
        self.exit_head = nn.Linear(mamba_d_model, 1)
        
        # Critic: State Value Estimate
        self.critic_head = nn.Linear(mamba_d_model, 1)

    def forward(self, v2_grid, l0_feature, ledger_state, macro_tensor, time_of_day, hidden_states=None):
        """
        v2_grid: [Batch, 8 (TFs), Seq, 52 (Features)]
        l0_feature: [Batch, Seq, 1]
        ledger_state: [Batch, Seq, 4]
        macro_tensor: [Batch, Seq, 260]
        time_of_day: [Batch, Seq, 4]
        hidden_states: list of tensors, one per Mamba layer
        """
        batch_size = v2_grid.size(0)
        seq_len = v2_grid.size(2)
        
        # --- Unblurred Flat Feed ---
        # Permute: [Batch, Seq, TFs, Features]
        x = v2_grid.permute(0, 2, 1, 3).contiguous()
        # Flatten TFs and Features: [Batch, Seq, 8 * 52] -> [Batch, Seq, 416]
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
            if MAMBA_AVAILABLE and isinstance(layer, Mamba):
                if hidden_states[i] is None:
                    # Initialize states for Mamba step: conv_state and ssm_state
                    conv_state = torch.zeros(batch_size, getattr(layer, 'd_inner', layer.config.d_inner if hasattr(layer, 'config') else 0), getattr(layer, 'd_conv', layer.config.d_conv if hasattr(layer, 'config') else 0), device=x.device, dtype=x.dtype)
                    ssm_state = torch.zeros(batch_size, getattr(layer, 'd_state', layer.config.d_state if hasattr(layer, 'config') else 16), getattr(layer, 'd_inner', layer.config.d_inner if hasattr(layer, 'config') else 0), device=x.device, dtype=x.dtype)
                    # Note: ssm_state shape is (B, d_state, d_inner) in mamba_ssm step? Wait, no, it's (B, d_inner, d_state) usually, let's just let mamba_ssm handle it if we use InferenceParams.
                    # Wait, InferenceParams API is cleaner. 
                    # Actually, we can just use layer.step directly if we match shapes.
                    # From mamba_ssm code: ssm_state is (B, d_inner, d_state)
                    ssm_state = torch.zeros(batch_size, getattr(layer, 'd_inner', layer.config.d_inner if hasattr(layer, 'config') else 0), getattr(layer, 'd_state', layer.config.d_state if hasattr(layer, 'config') else 16), device=x.device, dtype=x.dtype)
                    h = (conv_state, ssm_state)
                else:
                    h = hidden_states[i]
                
                if seq_len == 1:
                    x, conv_state, ssm_state = layer.step(x, h[0], h[1])
                    h = (conv_state, ssm_state)
                else:
                    x = layer(x)
                    h = None
                next_hidden_states.append(h)
            else:
                x, h = layer(x, hidden_states[i])
                next_hidden_states.append(h)
                
        x = self.norm(x)
        
        # Extract the final timestep for the decision (or keep all if needed)
        # For sequence-to-sequence TBPTT, we typically return the whole sequence.
        # But for compatibility with single-step Actor-Critic stepping, we extract the last.
        latest_step = x[:, -1, :] 
        
        # --- Output Heads (PPO) ---
        entry_logits = self.entry_head(latest_step)
        exit_logits = self.exit_head(latest_step)
        value_estimate = self.critic_head(latest_step)

        return entry_logits, exit_logits, value_estimate, next_hidden_states

    # ── Sequence-window training paths (docs/JULES_SEQUENCE_WINDOW_TRAINING.md) ──

    def _fuse_inputs(self, v2_grid, l0_feature, ledger_state, macro_tensor, time_of_day):
        """Shared input fusion → embedded trunk input [B, S, d_model].
        Same ops as the corresponding lines of forward()."""
        batch_size = v2_grid.size(0)
        seq_len = v2_grid.size(2)
        x = v2_grid.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)
        macro_encoded = self.macro_encoder(macro_tensor)
        x = torch.cat([x, l0_feature, ledger_state, macro_encoded, time_of_day], dim=-1)
        return self.embedding(self.input_norm(x))

    def forward_step(self, v2_grid, l0_feature, ledger_state, macro_tensor, time_of_day, states=None):
        """Acting-pass single-bar forward with carried (h, conv_state) per layer.
        Unlike forward(), the conv sees its true d_conv-1-bar history.
        states: list of (h, conv_state) tuples or None."""
        x = self._fuse_inputs(v2_grid, l0_feature, ledger_state, macro_tensor, time_of_day)
        if states is None:
            states = [None] * len(self.layers)
        next_states = []
        for i, layer in enumerate(self.layers):
            h, cs = states[i] if states[i] is not None else (None, None)
            x, h, cs = layer.step(x, h, cs)
            next_states.append((h, cs))
        x = self.norm(x)
        latest = x[:, -1, :]
        return (self.entry_head(latest), self.exit_head(latest),
                self.critic_head(latest), next_states)

    def forward_sequence(self, v2_grid, l0_feature, ledger_state, macro_tensor, time_of_day, states=None):
        """Learning-pass window forward: outputs for ALL bars.
        v2_grid [B, 8, W, 52], others [B, W, ·]. Returns
        (entry_logits [B,W,3], exit_logits [B,W,1], values [B,W,1], states)."""
        x = self._fuse_inputs(v2_grid, l0_feature, ledger_state, macro_tensor, time_of_day)
        if states is None:
            states = [None] * len(self.layers)
        next_states = []
        for i, layer in enumerate(self.layers):
            h, cs = states[i] if states[i] is not None else (None, None)
            x, h, cs = layer.forward_sequence(x, h, cs)
            next_states.append((h, cs))
        x = self.norm(x)
        return (self.entry_head(x), self.exit_head(x),
                self.critic_head(x), next_states)
