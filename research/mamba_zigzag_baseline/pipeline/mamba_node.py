import torch
import numpy as np
from collections import deque
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mamba_network import MambaPhysicsEncoder

class MambaInferenceNode:
    """
    Isolated Inference Node for the Mamba Physics Encoder.
    Maintains a strictly causal sliding window of the last 100 bars.
    Returns the predicted action only when the buffer is fully populated.
    """
    def __init__(self, checkpoint_path, expected_columns, seq_len=100, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = seq_len
        self.expected_columns = expected_columns
        self.input_dim = len(expected_columns)
        
        # Strictly causal 100-bar catcher
        self.state_queue = deque(maxlen=self.seq_len)
        
        # Load Model
        self.model = MambaPhysicsEncoder(
            input_dim=self.input_dim, 
            d_model=128, 
            num_classes=4, 
            num_layers=2
        ).to(self.device)
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[MambaNode] Loaded weights from {checkpoint_path}")
        else:
            print(f"[MambaNode] WARNING: Checkpoint {checkpoint_path} not found!")
            
        self.model.eval()

    def step(self, v2_dict: dict):
        """
        Takes exactly one bar's features (as a dictionary to prevent schema drift).
        Returns:
            action (int): 0=HOLD, 1=LONG, 2=SHORT, 3=SCRATCH (or None if buffer is warming up)
        """
        # Extract only the 385 features the model was trained on, in the exact order
        try:
            bar_features = [v2_dict.get(col, 0.0) for col in self.expected_columns]
        except KeyError as e:
            # Fallback for safety if a column is missing entirely
            bar_features = [v2_dict.get(col, 0.0) for col in self.expected_columns]
            
        self.state_queue.append(bar_features)
        
        # Warmup period: wait until we have 100 bars
        if len(self.state_queue) < self.seq_len:
            return None
            
        # Build strictly causal sequence tensor: [1, 100, 385]
        seq_array = np.array(self.state_queue, dtype=np.float32)
        
        # Rolling Normalization:
        # To avoid the lookahead leak present in the training data's global daily mean,
        # we strictly normalize using ONLY the mean and std of the current 100-bar window.
        means = seq_array.mean(axis=0)
        stds = seq_array.std(axis=0) + 1e-8
        seq_norm = (seq_array - means) / stds
        
        x_tensor = torch.tensor(seq_norm).unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            logits, _ = self.model(x_tensor)
            action = torch.argmax(logits, dim=-1).item()
            
        return action
