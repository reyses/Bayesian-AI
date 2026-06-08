import torch
import torch.nn as nn
import torch.nn.functional as F

class ResearchANetwork(nn.Module):
    def __init__(self, channels=8, sequence=60, features=25, lstm_hidden=128):
        super(ResearchANetwork, self).__init__()
        
        # Spatial Extraction (CNN)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        
        cnn_out_dim = 64 * 21 # = 1344
        
        # In Research A, we use the 3 extra features (v2_vec[0] + time_of_day + day_of_week)
        lstm_input_dim = cnn_out_dim + 3 # = 1347
        
        # Temporal Memory (Transferrable Starter Brain Core)
        self.lstm = nn.LSTM(input_size=lstm_input_dim, hidden_size=lstm_hidden, batch_first=True)
        
        # Quadrant Output Heads (Position-Aware)
        # lstm_hidden (lstm) + 24 (delta) + 7 (scalar)
        fc1_input_dim = lstm_hidden + 24 + 7
        self.fc1 = nn.Linear(fc1_input_dim, 128)
        
        # Head 1: Direct Action Space (Hold, Long, Short)
        self.fc_action = nn.Linear(128, 3)

    def forward(self, v2_grid, l0_feature, delta_features=None, scalar_context=None, hidden_state=None):
        """
        v2_grid: [Batch, 8 (TFs), 60 (Seq), 25 (Features)]
        l0_feature: [Batch, 60 (Seq), 3]
        delta_features: [Batch, 24] - Raw delta of selected features
        scalar_context: [Batch, 3] - direction, unrealized_pnl_norm, age_norm
        """
        x = F.pad(v2_grid, (0, 0, 2, 0)) 
        x = F.relu(self.conv1(x))
        
        x = F.pad(x, (0, 0, 2, 0))
        x = F.relu(self.conv2(x))
        
        batch_size = x.size(0)
        seq_len = x.size(2)
        
        x = x.permute(0, 2, 1, 3).contiguous()  # [Batch, 60, 64, 19]
        x = x.view(batch_size, seq_len, -1)     # [Batch, 60, 1216]
        
        # Concatenate L0
        lstm_in = torch.cat([x, l0_feature], dim=-1)  # [Batch, 60, 1219]
        
        if hidden_state is None:
            lstm_out, new_hidden = self.lstm(lstm_in)
        else:
            lstm_out, new_hidden = self.lstm(lstm_in, hidden_state)
            
        final_step = lstm_out[:, -1, :]  # [Batch, 128]
        
        batch_size = final_step.size(0)
        device = final_step.device
        
        if delta_features is None:
            delta_features = torch.zeros(batch_size, 24, device=device)
        if scalar_context is None:
            scalar_context = torch.zeros(batch_size, 7, device=device)
            
        merged = torch.cat([final_step, delta_features, scalar_context], dim=-1) # [Batch, 155]
        
        out = F.relu(self.fc1(merged))
        
        q_action = self.fc_action(out)       # [Batch, 3]
        
        return q_action, new_hidden
