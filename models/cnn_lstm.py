import torch
import torch.nn as nn


class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim, static_dim, hidden_dim=128, num_filters=64, kernel_size=3, num_layers=2, dropout=0.3, bidirectional=True):
        """
        CNN-LSTM model for time series forecasting

        Args:
            input_dim: Number of input features
            static_dim: Number of static features (coordinates)
            hidden_dim: Hidden dimension size
            num_filters: Number of CNN filters
            kernel_size: CNN kernel size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super(CNNLSTMModel, self).__init__()
        # Save init parameters for later model loading
        self.input_dim = input_dim
        self.static_dim = static_dim
        self.hidden_dim = hidden_dim
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        # CNN for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, num_filters, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(num_filters, num_filters*2, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(num_filters*2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=num_filters*2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # If bidirectional, the output dimension is doubled
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.bn_lstm = nn.BatchNorm1d(lstm_output_dim)

        # Projection for static features
        self.static_proj = nn.Sequential(
            nn.Linear(static_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Final prediction layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim + 32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, temporal_features, static_features):
        batch_size = temporal_features.shape[0]

        # CNN feature extraction (transpose for CNN channel dimension)
        x = temporal_features.permute(0, 2, 1)  # [batch, features, time]
        cnn_out = self.cnn(x)

        # Prepare for LSTM
        cnn_out = cnn_out.permute(0, 2, 1)  # [batch, time, features]

        # LSTM processing
        lstm_out, _ = self.lstm(cnn_out)
        lstm_out = lstm_out[:, -1, :]  # Take only the last timestep
        lstm_out = self.bn_lstm(lstm_out)

        # Process static features
        static_out = self.static_proj(static_features)

        # Combine features
        combined = torch.cat([lstm_out, static_out], dim=1)

        # Final prediction
        output = self.fc(combined)
        return output
