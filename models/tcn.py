import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    """
    Remove padding to ensure causal convolution
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Residual block for TCN with dilated convolutions
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        # 1x1 convolution for residual connection if input and output dimensions differ
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network with stacked dilated convolutions
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size  # Ensures causal convolution
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=padding, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNModel(nn.Module):
    """
    TCN model for time series forecasting
    """
    def __init__(self, input_dim, static_dim, num_channels=[64, 128, 256, 128],
                 kernel_size=3, dropout=0.3):
        """
        Temporal Convolutional Network model for time series forecasting

        Args:
            input_dim: Number of input features
            static_dim: Number of static features
            num_channels: List of channel sizes for each TCN layer
            kernel_size: Kernel size for convolutions
            dropout: Dropout rate
        """
        super(TCNModel, self).__init__()

        # TCN for temporal features
        self.tcn = TemporalConvNet(input_dim, num_channels, kernel_size=kernel_size, dropout=dropout)

        # Projection for static features
        self.static_proj = nn.Sequential(
            nn.Linear(static_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Final prediction layers
        self.fc = nn.Sequential(
            nn.Linear(num_channels[-1] + 32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, temporal_features, static_features):
        """
        Forward pass

        Args:
            temporal_features: Temporal features [batch, seq_len, features]
            static_features: Static features [batch, features]

        Returns:
            output: Prediction [batch, 1]
        """
        # Reshape for TCN: [batch, features, seq_len]
        x = temporal_features.transpose(1, 2)

        # Apply TCN
        tcn_out = self.tcn(x)

        # Extract the last time step
        tcn_out = tcn_out[:, :, -1]

        # Process static features
        static_out = self.static_proj(static_features)

        # Combine features
        combined = torch.cat([tcn_out, static_out], dim=1)

        # Final prediction
        output = self.fc(combined)

        return output
