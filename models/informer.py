"""
Informer Model Integration for Solar Radiation Forecasting
Paper: https://arxiv.org/abs/2012.07436
Source: https://github.com/zhouhaoyi/Informer2020

Modified by: Tin Hoang (for GHI forecasting)
Date: 26/04/2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class ProbMask:
    """
    Probability mask for ProbSparse Attention
    """
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer model
    """
    def __init__(self, d_model, max_len=1000):
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
        self.d_model = d_model

    def forward(self, x):
        # Make sure x is of shape [batch, seq_len, d_model]
        if x.size(-1) != self.d_model:
            raise ValueError(f"Expected input with last dimension {self.d_model}, got {x.size(-1)}")

        # Add positional encoding to input
        return x + self.pe[:, :x.size(1), :]


class TokenEmbedding(nn.Module):
    """
    Token embedding for input features
    """
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1  # Use fixed padding of 1 for simplicity and consistency
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='zeros', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        """
        Args:
            x: Input sequence of shape [B, L, D] where B is batch size, L is sequence length, D is feature dimension

        Returns:
            Output sequence of shape [B, L, d_model]
        """
        # x: [B, L, D]
        x = x.permute(0, 2, 1)  # [B, D, L]
        x = self.tokenConv(x)  # [B, d_model, L]
        x = x.transpose(1, 2)  # [B, L, d_model]
        return x


class FixedEmbedding(nn.Module):
    """
    Fixed embedding for frequency information
    """
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False
        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class DataEmbedding(nn.Module):
    """
    Data Embedding, combining token embedding and positional embedding
    """
    def __init__(self, c_in, d_model, dropout=0.1, embed_type='fixed'):
        super(DataEmbedding, self).__init__()

        # Token embedding for value features
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)

        # Position encoding
        self.position_embedding = PositionalEncoding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: [Batch, Length, Feature]
        # First apply value embedding to transform to d_model dimension
        x = self.value_embedding(x)
        # Then add positional encoding
        x = self.position_embedding(x)
        return self.dropout(x)


class ProbAttention(nn.Module):
    """
    Optimized ProbSparse Attention implementation
    """
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # Calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)

        # Sample k keys for each query
        index_sample = torch.randint(L_K, (L_Q, sample_k), device=Q.device)
        K_sample = K_expand[:, :, torch.arange(L_Q, device=Q.device).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # Find the Top_k query with sparsity measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # Use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B, device=Q.device)[:, None, None],
                     torch.arange(H, device=Q.device)[None, :, None],
                     M_top, :].contiguous()
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # Use mean for non-causal attention
            V_mean = V.mean(dim=-2)
            context = V_mean.unsqueeze(-2).expand(B, H, L_Q, D).clone()
        else:
            # Use cumulative sum for causal attention
            context = V.cumsum(dim=-2).clone()
        return context

    def _update_context(self, context_in, V, scores, index, L_Q):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -float('inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Create a new tensor instead of modifying in-place
        context = context_in.clone()
        values_aggregated = torch.matmul(attn, V)

        context[torch.arange(B, device=context.device)[:, None, None],
               torch.arange(H, device=context.device)[None, :, None],
               index, :] = values_aggregated

        if self.output_attention:
            attns = torch.zeros([B, H, L_Q, L_V], device=attn.device)
            attns[torch.arange(B, device=attns.device)[:, None, None],
                 torch.arange(H, device=attns.device)[None, :, None],
                 index, :] = attn
            return context, attns
        else:
            return context, None

    def forward(self, queries, keys, values, attn_mask=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1).contiguous()  # [B, H, L_Q, D]
        keys = keys.transpose(2, 1).contiguous()  # [B, H, L_K, D]
        values = values.transpose(2, 1).contiguous()  # [B, H, L_V, D]

        # Compute sampling parameters
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        # Limit to sequence length
        U_part = min(U_part, L_K)
        u = min(u, L_Q)

        # Compute probabilistic attention
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # Apply scaling
        scale = self.scale or 1. / math.sqrt(D)
        scores_top = scores_top * scale

        # Get context and update with attention
        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(context, values, scores_top, index, L_Q)

        # Return to original shape
        context = context.transpose(2, 1).contiguous()  # [B, L_Q, H, D]

        return context, attn


class AttentionLayer(nn.Module):
    """
    Attention Layer with multi-head attention
    """
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # More efficient projections (avoiding unnecessary reshapes)
        queries = self.query_projection(queries).reshape(B, L, H, -1).contiguous()
        keys = self.key_projection(keys).reshape(B, S, H, -1).contiguous()
        values = self.value_projection(values).reshape(B, S, H, -1).contiguous()

        # Call the attention mechanism
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )

        # Reshape back efficiently
        if self.mix:
            out = out.transpose(1, 2).contiguous()
        else:
            out = out.contiguous()

        out = out.reshape(B, L, -1)

        # Apply output projection
        return self.out_projection(out), attn


class EncoderLayer(nn.Module):
    """
    Encoder Layer with self-attention and feed-forward network
    """
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

    def forward(self, x, attn_mask=None):
        # Self Attention
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        x = self.norm1(x)

        # Feed Forward Network
        y = x.transpose(1, 2)
        y = self.conv1(y)
        y = self.activation(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = y.transpose(1, 2)

        x = x + self.dropout(y)
        x = self.norm2(x)

        return x, attn


class Encoder(nn.Module):
    """
    Informer encoder with distilling
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        self.use_distil = conv_layers is not None

    def forward(self, x, attn_mask=None):
        # Initialize list to store attention outputs
        attns = []

        if self.use_distil:
            # Encoder with distilling
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers[:-1], self.conv_layers)):
                # Apply attention
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

                # Apply convolution for distillation
                x = conv_layer(x)

                # Update attention mask if needed (sequence length reduced)
                if attn_mask is not None:
                    seq_len = x.shape[1]
                    attn_mask = attn_mask[:, :, :seq_len, :seq_len] if len(attn_mask.shape) == 4 else attn_mask[:, :seq_len, :seq_len]

            # Last attention layer (no conv after it)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            # Standard encoder without distilling
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class ConvLayer(nn.Module):
    """
    Downsampling with Conv1d for sequence distillation
    """
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        # Efficient implementation with single sequential block
        self.downConv = nn.Sequential(
            nn.Conv1d(in_channels=c_in, out_channels=c_in,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(c_in),
            nn.ELU()
        )
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # Fast path for very short sequences
        if x.size(1) <= 2:
            return x

        # Efficient processing path - create new tensor copies to avoid memory issues
        x_trans = x.transpose(1, 2).contiguous()  # [B, C, L]
        x_conv = self.downConv(x_trans)
        x_pool = self.pool(x_conv)
        x_out = x_pool.transpose(1, 2).contiguous()  # [B, L/2, C]

        return x_out


class DecoderLayer(nn.Module):
    """
    Decoder Layer with self-attention, cross-attention and feed-forward network
    """
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # Self Attention
        x_out, _ = self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )
        x = x + self.dropout(x_out)
        x = self.norm1(x)

        # Cross Attention
        x_out, attn = self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )
        x = x + self.dropout(x_out)
        x = self.norm2(x)

        # Feed Forward Network
        y = x.transpose(1, 2)
        y = self.conv1(y)
        y = self.activation(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = y.transpose(1, 2)

        x = x + self.dropout(y)
        x = self.norm3(x)

        return x, attn


class Decoder(nn.Module):
    """
    Informer decoder
    """
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x, _ = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x


class InformerModel(nn.Module):
    """Simplified Informer model for time series forecasting with static features"""
    def __init__(self, input_dim, static_dim, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu'):
        super(InformerModel, self).__init__()
        # Save init parameters for later model loading
        self.input_dim = input_dim
        self.static_dim = static_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation

        # Input embedding using TokenEmbedding (Conv1d-based)
        self.enc_embedding = TokenEmbedding(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Create encoder layers with ProbSparse attention
        attn_layers = []
        for _ in range(e_layers):
            prob_attention = ProbAttention(
                mask_flag=False,  # No causal mask for encoder
                factor=5,
                scale=None,
                attention_dropout=dropout,
                output_attention=False
            )
            attention_layer = AttentionLayer(
                prob_attention,
                d_model,
                n_heads,
                d_keys=d_model // n_heads,
                d_values=d_model // n_heads,
                mix=False
            )
            encoder_layer = EncoderLayer(
                attention_layer,
                d_model,
                d_ff,
                dropout,
                activation
            )
            attn_layers.append(encoder_layer)

        # Create distillation conv layers - key for Informer's performance
        conv_layers = None
        if e_layers > 1:
            conv_layers = [ConvLayer(d_model) for _ in range(e_layers-1)]

        # Encoder
        self.encoder = Encoder(
            attn_layers,
            conv_layers=conv_layers,  # Enable distilling for better performance
            norm_layer=nn.LayerNorm(d_model)
        )

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
        # Embed input and add positional encoding
        # temporal_features: [batch, seq_len, input_dim]
        x = self.enc_embedding(temporal_features)  # [batch, seq_len, d_model]
        x = self.pos_encoder(x)                    # [batch, seq_len, d_model]

        # Pass through encoder
        enc_out, _ = self.encoder(x)               # [batch, seq_len, d_model]

        # Get the last output for prediction
        last_hidden = enc_out[:, -1, :]            # [batch, d_model]

        # Process static features
        static_out = self.static_proj(static_features)  # [batch, d_model // 2]

        # Combine temporal and static features
        combined = torch.cat([last_hidden, static_out], dim=1)  # [batch, d_model + d_model // 2]

        # Output layer
        output = self.output_layer(combined)       # [batch]

        return output.squeeze()
