import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.3):
        """
        LSTM model for time series forecasting with simplified interface

        Args:
            input_size: Number of input features
            hidden_size: Hidden dimension size
            num_layers: Number of LSTM layers
            output_size: Number of output features
            dropout: Dropout rate
        """
        super(LSTMModel, self).__init__()

        # LSTM for temporal features
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Batch normalization for LSTM output
        self.bn_lstm = nn.BatchNorm1d(hidden_size)

        # Static features projection
        self.static_proj = nn.Sequential(
            nn.Linear(2, 16),  # Assuming 2D coordinates
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Final prediction layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + 16, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, temporal_features, static_features):
        """
        Forward pass

        Args:
            temporal_features: Temporal features [batch_size, seq_len, features]
            static_features: Static features [batch_size, features]

        Returns:
            output: Predicted values [batch_size, output_size]
        """
        # Process temporal features with LSTM
        lstm_out, _ = self.lstm(temporal_features)
        lstm_out = lstm_out[:, -1, :]  # Take only the last timestep output
        lstm_out = self.bn_lstm(lstm_out)

        # Process static features
        static_out = self.static_proj(static_features)

        # Combine features
        combined = torch.cat([lstm_out, static_out], dim=1)

        # Final prediction
        output = self.fc(combined)
        return output

class SimpleLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.3):
        """
        Simple LSTM model for time series forecasting without static features

        Args:
            input_size: Number of input features
            hidden_size: Hidden dimension size
            num_layers: Number of LSTM layers
            output_size: Number of output features
            dropout: Dropout rate
        """
        super(SimpleLSTMModel, self).__init__()

        # LSTM for temporal features
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Final prediction layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, temporal_features, static_features=None):
        """
        Forward pass

        Args:
            temporal_features: Temporal features [batch_size, seq_len, features]
            static_features: Optional static features (not used in this model)

        Returns:
            output: Predicted values [batch_size, output_size]
        """
        # Process temporal features with LSTM
        lstm_out, _ = self.lstm(temporal_features)
        lstm_out = lstm_out[:, -1, :]  # Take only the last timestep output

        # Final prediction
        output = self.fc(lstm_out)
        return output
