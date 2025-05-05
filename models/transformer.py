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
    """Transformer model for time series forecasting with static features"""
    def __init__(self, input_dim, static_dim, d_model=128, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.3, activation='gelu'):
        """
        Transformer model for time series forecasting

        Args:
            input_dim: Number of input features
            static_dim: Number of static features
            d_model: Dimension of the model
            n_heads: Number of heads in multi-head attention
            e_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of the feedforward network
            dropout: Dropout rate
        """
        super(TransformerModel, self).__init__()
        # Save init parameters
        self.input_dim = input_dim
        self.static_dim = static_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation

        # Data projection
        self.enc_embedding = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation if activation != 'gelu' else 'gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=e_layers)

        # Static features projection
        self.static_proj = nn.Sequential(
            nn.Linear(static_dim, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Output projection
        self.output_layer = nn.Sequential(
            nn.Linear(d_model + d_model // 2, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, temporal_features, static_features):
        # Project input to model dimension and add positional encoding
        x = self.enc_embedding(temporal_features)
        x = self.pos_encoder(x)

        # Pass through transformer encoder
        enc_out = self.transformer_encoder(x)

        # Get the last output for prediction
        last_hidden = enc_out[:, -1, :]

        # Process static features
        static_out = self.static_proj(static_features)

        # Combine temporal and static features
        combined = torch.cat([last_hidden, static_out], dim=1)

        # Output layer
        output = self.output_layer(combined)

        return output.squeeze()
