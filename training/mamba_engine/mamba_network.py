import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    logger.warning("mamba_ssm not found. Falling back to GRU sequence model proxy. To use true Mamba, run: pip install mamba-ssm causal-conv1d")

class MambaPhysicsEncoder(nn.Module):
    """
    Encodes the raw physics features (distances to bands, cubic slopes, etc.)
    into a continuous Regime State Vector.
    """
    def __init__(self, input_dim: int, d_model: int = 128, d_state: int = 16, d_conv: int = 4, expand: int = 2, num_layers: int = 2, num_classes: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Project raw features to model dimension
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Sequence modeling blocks
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if HAS_MAMBA:
                self.layers.append(
                    Mamba(
                        d_model=d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand
                    )
                )
            else:
                # Fallback Proxy for Mamba: Fast GRU
                self.layers.append(nn.GRU(d_model, d_model, batch_first=True))
                
        self.norm = nn.LayerNorm(d_model)
        
        # Action Head (Outputs logit probabilities for actions: LONG, SHORT, SCRATCH, HOLD)
        self.action_head = nn.Linear(d_model, num_classes)
        
        # Regime State Head (Outputs a continuous vector representing the 'Wet Memory' state)
        self.state_head = nn.Linear(d_model, d_state)

    def forward(self, x):
        """
        x: FloatTensor of shape (Batch, SequenceLength, Features)
        Returns:
            actions: Logits for the 4 discrete actions
            regime_state: Continuous physics state vector for the LLM context
        """
        # B = Batch Size, L = Sequence Length, D = Feature Dimension
        x = self.embedding(x)
        
        for layer in self.layers:
            if HAS_MAMBA:
                x = layer(x)
            else:
                # GRU returns (output, hidden)
                x, _ = layer(x)
                
        x = self.norm(x)
        
        # We only care about the latest step in the sequence for the current decision
        latest_step = x[:, -1, :] 
        
        actions = self.action_head(latest_step)
        regime_state = self.state_head(latest_step)
        
        return actions, regime_state
