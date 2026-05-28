import torch
import torch.nn as nn
import torch.nn.functional as F

class MasterNetwork(nn.Module):
    """
    CNN+LSTM Master Network for PW-CRL High-Frequency Trend Following.
    V2 Native: Ingests 185 features from the Forward Pass System.
    The 184 grid features are reshaped into [Batch, Channels=8 (TFs), Sequence=60, Features=23].
    The 1 L0 feature is concatenated before the LSTM.
    """
    def __init__(self, channels=8, sequence=60, features=23, lstm_hidden=128):
        super(MasterNetwork, self).__init__()
        
        # Spatial Extraction (CNN)
        # We process the 23 grounded V2 features across 8 Timeframe Channels
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        
        # After 2 layers of 3x3 kernel on the feature dimension (23),
        # 23 -> 21 -> 19 features remaining.
        cnn_out_dim = 64 * 19 
        
        # We concatenate the 3 L0 features before the LSTM (v2_vec[0], TimeOfDay, DayOfWeek)
        lstm_input_dim = cnn_out_dim + 3
        
        # Temporal Memory (LSTM)
        self.lstm = nn.LSTM(input_size=lstm_input_dim, hidden_size=lstm_hidden, batch_first=True)
        
        # Parameter Heads (10 Heads x 3 Levels [Low, Nominal, High])
        self.fc1 = nn.Linear(lstm_hidden, 128)
        
        # Whitepaper Physics Heads
        self.fc_zfit = nn.Linear(128, 3)
        self.fc_lambda = nn.Linear(128, 3)
        self.fc_hurst = nn.Linear(128, 3)
        self.fc_theta = nn.Linear(128, 3)
        self.fc_pid = nn.Linear(128, 3)
        
        # Extended Engine Heads (exnmp)
        self.fc_vr_entry = nn.Linear(128, 3)
        self.fc_vr_max = nn.Linear(128, 3)
        self.fc_vr_bail = nn.Linear(128, 3)
        self.fc_freight = nn.Linear(128, 3)
        self.fc_wick = nn.Linear(128, 3)

    def forward(self, v2_grid, l0_feature, hidden_state=None):
        """
        v2_grid: [Batch, 8 (TFs), 60 (Seq), 23 (Features)]
        l0_feature: [Batch, 60 (Seq), 1]
        """
        # STRICT CAUSAL PADDING
        # We pad the time/sequence dimension at the top (start) to prevent forward-bias
        # Kernel size is 3, so pad 2 rows at the top.
        # F.pad format: (padLeft, padRight, padTop, padBottom) for last 2 dims
        x = F.pad(v2_grid, (0, 0, 2, 0)) 
        x = F.relu(self.conv1(x))
        
        x = F.pad(x, (0, 0, 2, 0))
        x = F.relu(self.conv2(x))
        
        # Reshape for LSTM: [Batch, Sequence, CNN_Output]
        # x shape: [Batch, 64, 60, 19]
        batch_size = x.size(0)
        seq_len = x.size(2)
        
        x = x.permute(0, 2, 1, 3).contiguous()  # [Batch, 60, 64, 19]
        x = x.view(batch_size, seq_len, -1)     # [Batch, 60, 1216]
        
        # Concatenate L0
        x = torch.cat([x, l0_feature], dim=-1)  # [Batch, 60, 1217]
        
        # LSTM Temporal Pass
        if hidden_state is not None:
            lstm_out, new_hidden = self.lstm(x, hidden_state)
        else:
            lstm_out, new_hidden = self.lstm(x)
            
        # Extract the final hidden state of the sequence for Q-value estimation
        final_step = lstm_out[:, -1, :]  # [Batch, 128]
        
        # Output Parameter Q-Values
        out = F.relu(self.fc1(final_step))
        
        q_zfit = self.fc_zfit(out)
        q_lambda = self.fc_lambda(out)
        q_hurst = self.fc_hurst(out)
        q_theta = self.fc_theta(out)
        q_pid = self.fc_pid(out)
        
        q_vr_entry = self.fc_vr_entry(out)
        q_vr_max = self.fc_vr_max(out)
        q_vr_bail = self.fc_vr_bail(out)
        q_freight = self.fc_freight(out)
        q_wick = self.fc_wick(out)
        
        # Group into a tuple of 10 heads
        heads = (q_zfit, q_lambda, q_hurst, q_theta, q_pid, 
                 q_vr_entry, q_vr_max, q_vr_bail, q_freight, q_wick)
        
        return heads, new_hidden
