"""
TSMixer model implementation for time series forecasting.

This module implements the TSMixer model architecture, which applies the principles
of MLP-Mixer to time series data for forecasting tasks.

Dependencies:
    - torchtsmixer: Install with `pip install pytorch-tsmixer`
    - Documentation: https://github.com/ditschuk/pytorch-tsmixer

Reference:
    - TSMixer: An All-MLP Architecture for Time Series Forecasting
    - Paper: https://arxiv.org/abs/2303.06053
"""

import torch
import torch.nn as nn
from torchtsmixer import TSMixer as BaseTSMixer


class TSMixerModel(nn.Module):
    def __init__(self, input_dim, static_dim, lookback=24, horizon=24, ff_dim=128, num_blocks=2, norm_type='batch',
                 activation='relu', dropout=0.2):
        """
        TSMixer model for time series forecasting

        Args:
            input_dim: Number of input features (channels)
            static_dim: Number of static features (coordinates)
            lookback: Number of time steps to look back (sequence length)
            horizon: Number of time steps to predict (prediction length)
            ff_dim: Feed-forward dimension
            num_blocks: Number of TSMixer blocks
            norm_type: Normalization type ('batch' or 'layer')
            activation: Activation function ('relu' or 'gelu')
            dropout: Dropout rate
        """
        super(TSMixerModel, self).__init__()

        # Base TSMixer model for temporal features
        self.tsmixer = BaseTSMixer(
            sequence_length=lookback,
            prediction_length=horizon,
            input_channels=input_dim,
            output_channels=1,  # Single target value prediction
            ff_dim=ff_dim,
            num_blocks=num_blocks,
            activation_fn=activation,
            dropout_rate=dropout,
            norm_type=norm_type
        )

        # Projection for static features
        self.static_proj = nn.Sequential(
            nn.Linear(static_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Final prediction layer combining TSMixer output with static features
        self.fc = nn.Sequential(
            nn.Linear(horizon + 32, 64),  # TSMixer output + static projection
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # Single output value
        )

        self.lookback = lookback
        self.horizon = horizon

    def forward(self, temporal_features, static_features):
        """
        Forward pass of the TSMixer model

        Args:
            temporal_features: Temporal features of shape (batch_size, lookback, input_dim)
            static_features: Static features of shape (batch_size, static_dim)

        Returns:
            Predictions of shape (batch_size, 1) - a 2D tensor
        """
        # Process temporal features through TSMixer
        # TSMixer expects shape (batch_size, sequence_length, input_channels)
        tsmixer_out = self.tsmixer(temporal_features)  # Shape: (batch_size, prediction_length, 1)

        # Flatten the TSMixer output
        tsmixer_flat = tsmixer_out.squeeze(-1)  # Shape: (batch_size, prediction_length)

        # Process static features
        static_out = self.static_proj(static_features)  # Shape: (batch_size, 32)

        # Combine features
        combined = torch.cat([tsmixer_flat, static_out], dim=1)  # Shape: (batch_size, prediction_length + 32)

        # Final prediction
        output = self.fc(combined)  # Shape: (batch_size, 1)

        return output  # Shape: (batch_size, 1)
