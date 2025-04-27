import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1DModel(nn.Module):
    """
    1D CNN model for time series forecasting
    """
    def __init__(self, input_dim, static_dim, num_filters=[64, 128, 256],
                 kernel_sizes=[3, 3, 3], dropout=0.3):
        """
        1D Convolutional Neural Network model for time series forecasting

        Args:
            input_dim: Number of input features
            static_dim: Number of static features
            num_filters: List of filter sizes for each convolutional layer
            kernel_sizes: List of kernel sizes for each convolutional layer
            dropout: Dropout rate
        """
        super(CNN1DModel, self).__init__()

        # Input validation
        assert len(num_filters) == len(kernel_sizes), "num_filters and kernel_sizes must have the same length"

        # Store parameters
        self.input_dim = input_dim
        self.static_dim = static_dim
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.dropout = dropout

        # Build CNN layers
        self.cnn_layers = nn.ModuleList()
        in_channels = input_dim

        for i, (num_filter, kernel_size) in enumerate(zip(num_filters, kernel_sizes)):
            block = nn.Sequential(
                nn.Conv1d(in_channels, num_filter, kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(num_filter),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(kernel_size=2, stride=2) if i < len(num_filters)-1 else nn.Identity()
            )
            self.cnn_layers.append(block)
            in_channels = num_filter

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

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
            nn.Linear(num_filters[-1] + 32, 128),
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
        # Convert temporal features to CNN input format [batch, channels, seq_len]
        x = temporal_features.transpose(1, 2)

        # Apply CNN layers
        for layer in self.cnn_layers:
            x = layer(x)

        # Global average pooling over the sequence dimension
        x = self.global_avg_pool(x).squeeze(-1)

        # Process static features
        static_out = self.static_proj(static_features)

        # Combine features
        combined = torch.cat([x, static_out], dim=1)

        # Final prediction
        output = self.fc(combined)

        return output
