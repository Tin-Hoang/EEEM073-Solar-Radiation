import torch
import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(self, input_dim, static_dim, hidden_dims=[256, 512, 256, 128], dropout=0.3, lookback=24):
        """
        MLP model for time series forecasting

        Args:
            input_dim: Number of input features
            static_dim: Number of static features (coordinates)
            hidden_dims: List of hidden dimensions
            dropout: Dropout rate
            lookback: Number of time steps to look back
        """
        super(MLPModel, self).__init__()
        # Save init parameters for later model loading
        self.input_dim = input_dim
        self.static_dim = static_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lookback = lookback

        # Calculate flattened input size
        self.flatten_dim = input_dim * lookback

        # MLP for temporal features
        layers = []
        layers.append(nn.Linear(self.flatten_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        for i in range(len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)

        # Projection for static features
        self.static_proj = nn.Sequential(
            nn.Linear(static_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Final prediction layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dims[-1] + 32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, temporal_features, static_features):
        batch_size = temporal_features.shape[0]

        # Flatten temporal features
        x_flat = temporal_features.reshape(batch_size, -1)

        # MLP processing
        mlp_out = self.mlp(x_flat)

        # Process static features
        static_out = self.static_proj(static_features)

        # Combine features
        combined = torch.cat([mlp_out, static_out], dim=1)

        # Final prediction
        output = self.fc(combined)
        return output
