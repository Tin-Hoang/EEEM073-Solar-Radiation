import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer model
    """
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sin/cos positional encoding
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # Register buffer (not a parameter, but should be saved and moved with the model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to input
        return x + self.pe[:, :x.size(1), :]


class TransformerModel(nn.Module):
    def __init__(self, input_dim, static_dim, d_model=128, nhead=8, num_layers=2, dim_feedforward=512, dropout=0.3):
        """
        Transformer model for time series forecasting

        Args:
            input_dim: Number of input features
            static_dim: Number of static features
            d_model: Dimension of the model
            nhead: Number of heads in multi-head attention
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of the feedforward network
            dropout: Dropout rate
        """
        super(TransformerModel, self).__init__()

        # Feature dimension projection to d_model
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Batch normalization for transformer output
        self.bn_transformer = nn.BatchNorm1d(d_model)

        # Projection for static features
        self.static_proj = nn.Sequential(
            nn.Linear(static_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Final prediction layers
        self.fc = nn.Sequential(
            nn.Linear(d_model + 32, 64),
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
        """
        Forward pass of the transformer model

        Args:
            temporal_features: Temporal features [batch, seq_len, features]
            static_features: Static features [batch, features]

        Returns:
            output: Prediction [batch, 1]
        """
        # Project input to d_model dimension
        x = self.input_projection(temporal_features)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Apply transformer encoder
        transformer_out = self.transformer_encoder(x)

        # Extract the last time step
        transformer_out = transformer_out[:, -1, :]
        transformer_out = self.bn_transformer(transformer_out)

        # Process static features
        static_out = self.static_proj(static_features)

        # Combine features
        combined = torch.cat([transformer_out, static_out], dim=1)

        # Final prediction
        output = self.fc(combined)
        return output
