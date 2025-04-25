"""
Model utilities for GHI forecasting.
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
from torchinfo import summary


def inverse_transform_predictions(y_pred, scaler):
    """
    Inverse transform scaled predictions back to original scale

    Args:
        y_pred: Scaled predictions
        scaler: Scaler used for scaling

    Returns:
        y_pred_orig: Predictions in original scale
    """
    # Reshape for inverse transformation
    y_pred_2d = y_pred.reshape(-1, y_pred.shape[-1])

    # Inverse transform
    y_pred_orig_2d = scaler.inverse_transform(y_pred_2d)

    # Reshape back to original dimensions
    y_pred_orig = y_pred_orig_2d.reshape(y_pred.shape)

    return y_pred_orig

def plot_training_history(history, metrics=None):
    """
    Plot training history

    Args:
        history: Keras training history
        metrics: List of metrics to plot (default: ['loss', 'mae'])

    Returns:
        fig: Matplotlib figure
    """
    if metrics is None:
        metrics = ['loss', 'mae']

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.plot(history.history[metric], label=f'Training {metric}')
        val_metric = f'val_{metric}'
        if val_metric in history.history:
            ax.plot(history.history[val_metric], label=f'Validation {metric}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} over epochs')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    return fig

def plot_predictions(y_true, y_pred, timestamps=None, horizon=None, sample_indices=None, max_samples=5):
    """
    Plot predictions vs ground truth for selected samples

    Args:
        y_true: True values
        y_pred: Predicted values
        timestamps: Optional timestamps for x-axis
        horizon: Forecast horizon
        sample_indices: Optional indices of samples to plot
        max_samples: Maximum number of samples to plot

    Returns:
        fig: Matplotlib figure
    """
    if sample_indices is None:
        # Randomly select samples if not provided
        sample_indices = np.random.choice(len(y_true), min(max_samples, len(y_true)), replace=False)
    else:
        # Limit number of samples to plot
        sample_indices = sample_indices[:max_samples]

    n_samples = len(sample_indices)
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 4 * n_samples))

    if n_samples == 1:
        axes = [axes]

    for i, idx in enumerate(sample_indices):
        ax = axes[i]
        true = y_true[idx]
        pred = y_pred[idx]

        if timestamps is not None and horizon is not None:
            # Use real timestamps if provided
            x_values = timestamps[idx:idx+horizon]
            ax.plot(x_values, true, 'b-', label='True')
            ax.plot(x_values, pred, 'r--', label='Predicted')
            ax.set_xlabel('Time')
        else:
            # Otherwise use step numbers
            steps = np.arange(len(true))
            ax.plot(steps, true, 'b-', label='True')
            ax.plot(steps, pred, 'r--', label='Predicted')
            ax.set_xlabel('Forecast Step')

        ax.set_ylabel('GHI')
        ax.set_title(f'Sample {idx}')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    return fig

def get_model_summary(model, batch_size=16, seq_length=24, feature_dim=10, static_dim=5):
    """
    Generate a detailed summary of a PyTorch model using torchinfo.

    Args:
        model: PyTorch model
        batch_size: Batch size for input shape
        seq_length: Sequence length for temporal features
        feature_dim: Number of features in temporal input
        static_dim: Number of features in static input

    Returns:
        ModelStatistics object with model summary
    """
    # Create dummy inputs with appropriate shapes
    temporal_shape = (batch_size, seq_length, feature_dim)
    static_shape = (batch_size, static_dim)

    # Generate the summary
    model_summary = summary(
        model,
        input_data=[
            torch.zeros(temporal_shape),
            torch.zeros(static_shape)
        ],
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
    )

    return model_summary

def print_model_info(model, batch_size=16, seq_length=24, feature_dim=10, static_dim=5):
    """
    Print compact model information including parameter count and layer structure.

    Args:
        model: PyTorch model
        batch_size: Batch size for input shape
        seq_length: Sequence length for temporal features
        feature_dim: Number of features in temporal input
        static_dim: Number of features in static input
    """
    # For notebooks that can't install torchinfo, fall back to manual parameter counting
    try:
        print(get_model_summary(model, batch_size, seq_length, feature_dim, static_dim))
    except:
        # Fallback if torchinfo is not available
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Model: {model.__class__.__name__}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")
        print("\nModel structure:")
        print(model)
