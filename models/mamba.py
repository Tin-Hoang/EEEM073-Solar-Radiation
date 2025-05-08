"""
Mamba model for time series forecasting with temporal and static features.

This model is based on the Mamba architecture, which is a variant of the State Space Model.
It uses a selective scan operation with data-dependent parameters to process the input sequence.
- Source: https://github.com/state-spaces/mamba

Modified by: Tin Hoang (for GHI forecasting)
Date: 05/05/2025
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (SSM) - core building block of Mamba architecture.
    Implements the selective scan operation with data-dependent parameters.
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        dt_rank: int = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        dt_rank = dt_rank or max(1, d_model // 4)

        # S4D parameters (A, B, C)
        # We use the diagonalized state space formulation for efficiency
        self.d_model = d_model
        self.d_state = d_state

        # Initialize discrete state matrix A (fixed)
        # We use a simple exponentially-decaying state matrix
        # In the full Mamba, these are initialized more carefully
        self.register_buffer("A_log", torch.randn(self.d_state))

        # Projection matrices
        self.B = nn.Parameter(torch.randn(self.d_model, self.d_state))
        self.C = nn.Parameter(torch.randn(self.d_state, self.d_model))

        # Selective scan uses input-dependent delta (Δ)
        # This is a key difference from standard S4 models
        self.dt_proj = nn.Linear(self.d_model, dt_rank)
        self.dt_proj_weight = nn.Parameter(torch.randn(dt_rank, self.d_state))
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_scale = dt_scale
        self.dt_init = dt_init

        # Output projection and dropout
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize parameters as per Mamba paper
        nn.init.normal_(self.B, std=0.1)
        nn.init.normal_(self.C, std=0.1)

        if self.dt_init == "random":
            # Initialize dtº between dt_min and dt_max
            nn.init.uniform_(self.dt_proj.weight, a=-0.5, b=0.5)
            nn.init.constant_(self.dt_proj.bias, 0.0)
            nn.init.uniform_(self.dt_proj_weight, a=-0.5, b=0.5)
        else:
            # Constant initialization
            with torch.no_grad():
                # Initialize to average of dt_min and dt_max
                dt_init = 0.5 * (self.dt_min + self.dt_max)
                self.dt_proj.weight.zero_()
                self.dt_proj.bias.fill_(dt_init)
                self.dt_proj_weight.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            y: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        # Compute Δ (time step) based on the input x
        # This is the "selective" part that makes the model data-dependent
        delta = self.dt_proj(x)  # (batch, seq_len, dt_rank)
        delta = torch.einsum('bsr,rd->bsd', delta, self.dt_proj_weight)  # (batch, seq_len, d_state)
        delta = self.dt_min + (self.dt_max - self.dt_min) * torch.sigmoid(delta * self.dt_scale)

        # Convert A from log space to linear space for the discrete SSM
        A = torch.exp(self.A_log)  # (d_state,)

        # Discretize continuous parameters (A, B) using input-dependent delta (Δ)
        # These become the Ā, B̄ in discrete-time system
        # For each step, we compute: A_bar = exp(Δ * A)
        # This is a simplified version of the more sophisticated parameterization in the full Mamba
        A_bar = torch.exp(delta * A.view(1, 1, -1))  # (batch, seq_len, d_state)

        # Compute input projection for B
        x_B = torch.einsum('bsd,de->bse', x, self.B)  # (batch, seq_len, d_state)

        # Simplified scan operation - instead of the parallel scan for efficiency,
        # we use a simple recurrent formulation for clarity
        h = torch.zeros(batch_size, self.d_state, device=x.device)
        y = torch.zeros(batch_size, seq_len, self.d_model, device=x.device)

        for t in range(seq_len):
            # Get the current time step values explicitly
            A_bar_t = A_bar[:, t, :]  # Shape: (batch_size, d_state)
            x_B_t = x_B[:, t, :]      # Shape: (batch_size, d_state)

            # State update: h_{t+1} = A_bar_t * h_t + x_B_t
            h = A_bar_t * h + x_B_t

            # Output projection: y_t = C * h_t
            y_t = torch.einsum('bd,dm->bm', h, self.C)
            y[:, t] = y_t

        # Apply dropout
        y = self.dropout(y)

        return y


class MambaBlock(nn.Module):
    """
    Mamba Block - the core building block of the Mamba model.
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        dt_rank: int = None,
        d_conv: int = 4,
        conv_bias: bool = True,
        bias: bool = False,
        expand_factor: int = 2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand_factor = expand_factor
        self.d_inner = int(expand_factor * d_model)

        # Input projection - expands dimension to inner dimension
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # Local convolution to capture short-range dependencies
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=conv_bias,
        )

        # Normalization layers
        self.norm = nn.LayerNorm(d_model)
        self.norm_inner = nn.LayerNorm(self.d_inner)

        # SSM - the core component of Mamba
        self.ssm = SelectiveSSM(
            d_model=self.d_inner,
            d_state=d_state,
            dt_rank=dt_rank,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dropout=dropout,
        )

        # Output projection - reduces dimension back to model dimension
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

        # SiLU activation
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            y: (batch_size, seq_len, d_model)
        """
        # Apply normalization (Pre-LN style)
        residual = x
        x = self.norm(x)

        # Input projection with SiLU gating
        x_proj = self.in_proj(x)
        x_proj_1, x_proj_2 = torch.split(x_proj, self.d_inner, dim=-1)

        # Apply 1D convolution for local context
        # Transpose to channels-first for Conv1D
        x_conv = x_proj_1.transpose(1, 2)
        x_conv = self.conv1d(x_conv)
        # Causal convolution - remove future context
        x_conv = x_conv[:, :, :x_proj_1.size(1)]
        x_conv = x_conv.transpose(1, 2)

        # Optional inner normalization and activation
        x_conv = self.norm_inner(x_conv)
        x_conv = self.act(x_conv)

        # Apply SSM
        x_ssm = self.ssm(x_conv)

        # Gate the SSM output with second projection
        x_gated = x_ssm * self.act(x_proj_2)

        # Output projection
        out = self.out_proj(x_gated)

        # Residual connection
        return out + residual


class MambaModel(nn.Module):
    """
    Mamba model for time series forecasting with temporal and static features
    """
    def __init__(
        self,
        input_dim: int,
        static_dim: int,
        d_model: int = 128,
        d_state: int = 16,
        n_layers: int = 4,
        dt_rank: int = None,
        d_conv: int = 4,
        expand_factor: int = 2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.static_dim = static_dim
        self.d_model = d_model
        self.d_state = d_state
        self.n_layers = n_layers
        self.dropout = dropout
        self.dt_rank = dt_rank
        self.d_conv = d_conv
        self.expand_factor = expand_factor
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_scale = dt_scale

        # Input projections
        self.temporal_proj = nn.Linear(input_dim, d_model)

        # Static feature processing
        self.static_proj = nn.Sequential(
            nn.Linear(static_dim, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        # Mamba blocks
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                dt_rank=dt_rank,
                d_conv=d_conv,
                expand_factor=expand_factor,
                dt_min=dt_min,
                dt_max=dt_max,
                dt_init=dt_init,
                dt_scale=dt_scale,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Output projection for GHI prediction
        self.output_layer = nn.Sequential(
            nn.Linear(d_model + d_model // 2, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, temporal_features: torch.Tensor, static_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temporal_features: (batch_size, seq_len, input_dim)
            static_features: (batch_size, static_dim)
        Returns:
            output: (batch_size,) - GHI prediction
        """
        # Project temporal features
        x = self.temporal_proj(temporal_features)

        # Process through Mamba blocks
        for block in self.mamba_blocks:
            x = block(x)

        # Get the last output for prediction
        last_hidden = x[:, -1, :]

        # Process static features
        static_out = self.static_proj(static_features)

        # Combine temporal and static features
        combined = torch.cat([last_hidden, static_out], dim=1)

        # Output layer
        output = self.output_layer(combined)

        return output.squeeze()
