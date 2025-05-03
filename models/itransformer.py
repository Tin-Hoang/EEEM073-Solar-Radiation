"""
iTransformer model implementation for time series forecasting.

This module implements the iTransformer architecture, which inverts the roles of
features and timestamps in traditional Transformers, treating each feature dimension
as a token and applying self-attention across the feature dimension.

Reference:
    - iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
    - Paper: https://arxiv.org/abs/2310.06625
    - GitHub: https://github.com/thuml/iTransformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Standard positional encoding for sequence positions.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    """
    Embed each feature dimension as a token.
    """
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Linear(c_in, d_model)

    def forward(self, x):
        # x: [batch_size, seq_length, features]
        # Transpose to make features the token dimension
        x = self.tokenConv(x.permute(0, 2, 1))  # [batch_size, features, d_model]
        return x


class iTransformerEncoder(nn.Module):
    """
    iTransformer encoder applies self-attention across the feature dimension.
    """
    def __init__(self, d_model, n_heads, d_ff, dropout, num_layers):
        super(iTransformerEncoder, self).__init__()

        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation="gelu",
                batch_first=True
            ) for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [batch_size, features, d_model]
        for layer in self.encoder_layers:
            x = layer(x)

        return self.layer_norm(x)


class iTransformerDecoder(nn.Module):
    """
    Simple linear decoder to predict target values.
    """
    def __init__(self, d_model, input_dim, pred_len):
        super(iTransformerDecoder, self).__init__()
        self.pred_len = pred_len

        # Project from d_model to prediction length for each feature
        self.projection = nn.Linear(d_model, pred_len)

    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim, d_model]

        Returns:
            [batch_size, pred_len, 1]
        """
        # Apply projection to each feature's embedding
        # x shape: [batch_size, input_dim, d_model]
        x = self.projection(x)  # [batch_size, input_dim, pred_len]

        # Average across features to get final prediction
        x = x.mean(dim=1)  # [batch_size, pred_len]

        return x.unsqueeze(-1)  # [batch_size, pred_len, 1]


class iTransformerModel(nn.Module):
    """
    iTransformer model for time series forecasting.

    This model inverts the traditional Transformer approach by treating each feature
    dimension as a token, and applying self-attention across the feature dimension.
    """
    def __init__(self, input_dim, static_dim, d_model=512, n_heads=8,
                 e_layers=3, d_ff=2048, dropout=0.1, lookback=24, pred_len=1):
        """
        Initialize iTransformer model.

        Args:
            input_dim: Number of input features
            static_dim: Number of static features
            d_model: Model dimension
            n_heads: Number of attention heads
            e_layers: Number of encoder layers
            d_ff: Dimension of feed-forward network
            dropout: Dropout rate
            lookback: Historical sequence length
            pred_len: Prediction sequence length
        """
        super(iTransformerModel, self).__init__()

        self.lookback = lookback
        self.pred_len = pred_len
        self.input_dim = input_dim

        # Token embedding to embed each feature as a token
        self.embedding = TokenEmbedding(lookback, d_model)

        # Positional encoding for feature dimension positions
        self.pos_encoder = PositionalEncoding(d_model)

        # Encoder for feature-wise attention
        self.encoder = iTransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            num_layers=e_layers
        )

        # Decoder to generate predictions
        self.decoder = iTransformerDecoder(
            d_model=d_model,
            input_dim=input_dim,
            pred_len=pred_len
        )

        # Static feature projection
        self.static_proj = nn.Sequential(
            nn.Linear(static_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Final prediction layer
        self.fc = nn.Sequential(
            nn.Linear(pred_len + 32, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, temporal_features, static_features):
        """
        Forward pass for iTransformer.

        Args:
            temporal_features: Temporal features [batch_size, lookback, input_dim]
            static_features: Static features [batch_size, static_dim]

        Returns:
            output: Predictions [batch_size, 1]
        """
        # Embed input sequence, treating each feature as a token
        x = self.embedding(temporal_features)  # [batch_size, input_dim, d_model]

        # Add positional encoding for feature positions
        x = self.pos_encoder(x)  # [batch_size, input_dim, d_model]

        # Apply encoder
        enc_out = self.encoder(x)  # [batch_size, input_dim, d_model]

        # Generate predictions
        dec_out = self.decoder(enc_out)  # [batch_size, pred_len, 1]
        dec_out = dec_out.squeeze(-1)  # [batch_size, pred_len]

        # Process static features
        static_out = self.static_proj(static_features)  # [batch_size, 32]

        # Combine features for final prediction
        combined = torch.cat([dec_out, static_out], dim=1)  # [batch_size, pred_len + 32]

        # Final prediction
        output = self.fc(combined)  # [batch_size, 1]

        return output
