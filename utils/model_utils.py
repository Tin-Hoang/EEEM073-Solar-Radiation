"""
Model utilities for GHI forecasting.
"""

import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
import importlib

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch


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

def get_model_summary(model, temporal_shape=None, static_shape=None):
    """
    Generate a detailed summary of a PyTorch model using torchinfo.

    Args:
        model: PyTorch model
        temporal_shape: Shape of temporal features
        static_shape: Shape of static features

    Returns:
        ModelStatistics object with model summary
    """
    from torchinfo import summary

    # Create dummy inputs with appropriate shapes
    try:
        if temporal_shape is None or static_shape is None:
            model_summary = summary(model)
        else:
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
    except IndexError:
        # Fallback to default summary if dimension input error
        model_summary = summary(model)
    except Exception as e:
        print(f"Error generating model summary: {e}")
        raise e

    return model_summary

def print_model_info(model, temporal_shape=None, static_shape=None):
    """
    Print compact model information including parameter count and layer structure.

    Args:
        model: PyTorch model
        temporal_shape: Shape of temporal features
        static_shape: Shape of static features
    """
    # For notebooks that can't install torchinfo, fall back to manual parameter counting
    try:
        print(get_model_summary(model, temporal_shape, static_shape))
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

def save_model(model, filepath, metadata=None, temporal_features=None, static_features=None,
           target_field=None, config=None, time_feature_keys=None):
    """
    Save a PyTorch model with all necessary metadata to restore it later.

    Args:
        model: PyTorch model to save
        filepath: Path where to save the model
        metadata: Additional metadata dictionary
        temporal_features: List of temporal feature field names
        static_features: List of static feature field names
        target_field: Target field name
        config: Configuration used during training
        time_feature_keys: List of individual time feature keys (hour_sin, hour_cos, etc.)

    Returns:
        saved_path: Path to the saved model file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    # Create base metadata
    save_metadata = {
        "model_type": model.__class__.__name__,
        "model_module": model.__class__.__module__,
        "saved_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pytorch_version": torch.__version__,
    }

    # Check if wandb is running and add URL
    try:
        import wandb
        # Import is_wandb_enabled inside the function to avoid circular imports
        from utils.wandb_utils import is_wandb_enabled
        if is_wandb_enabled():
            save_metadata["wandb_url"] = wandb.run.get_url()
            save_metadata["wandb_run_id"] = wandb.run.id
            save_metadata["wandb_run_name"] = wandb.run.name
    except (ImportError, AttributeError):
        # wandb not installed or not initialized
        pass

    # Add provided metadata
    if metadata is not None:
        save_metadata.update(metadata)

    # Add feature information
    if temporal_features is not None:
        save_metadata["temporal_features"] = temporal_features

    if static_features is not None:
        save_metadata["static_features"] = static_features

    if target_field is not None:
        save_metadata["target_field"] = target_field

    # Add time feature keys for the new format
    if time_feature_keys is not None:
        save_metadata["time_feature_keys"] = time_feature_keys
    else:
        # Default time feature keys for backward compatibility
        default_time_keys = [
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'month_sin', 'month_cos', 'dow_sin', 'dow_cos'
        ]
        save_metadata["time_feature_keys"] = default_time_keys

    # Add configuration information
    if config is not None:
        # Convert config values to JSON serializable types if needed
        serializable_config = {}
        for key, value in config.items():
            if isinstance(value, (int, float, str, bool, list, dict, tuple, type(None))):
                serializable_config[key] = value
            else:
                # Try to convert to string representation
                try:
                    serializable_config[key] = str(value)
                except:
                    # Skip if not serializable
                    pass
        save_metadata["config"] = serializable_config

    # Add model architecture parameters
    model_init_params = {}
    # Try to extract initialization parameters from the model
    for attr_name in dir(model):
        # Skip private attributes and functions
        if attr_name.startswith('_') or callable(getattr(model, attr_name)):
            continue

        # Get attribute value
        attr_value = getattr(model, attr_name)

        # Store only serializable parameters
        if isinstance(attr_value, (int, float, str, bool, list, dict, tuple)):
            model_init_params[attr_name] = attr_value

    save_metadata["model_init_params"] = model_init_params

    # Create the save dictionary
    save_dict = {
        "state_dict": model.state_dict(),
        "metadata": save_metadata
    }

    # Save the model and metadata
    torch.save(save_dict, filepath)

    print(f"Model saved to {filepath} with metadata.")
    return filepath

def load_model(filepath, device=None):
    """
    Load a model from a saved file along with its metadata.

    Args:
        filepath: Path to the saved model file
        device: Device to load the model to (None for auto-detection)

    Returns:
        model: Loaded PyTorch model
        metadata: Model metadata
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the saved dictionary
    save_dict = torch.load(filepath, map_location=device)

    # Extract state_dict and metadata
    state_dict = save_dict["state_dict"]
    metadata = save_dict["metadata"]

    # Get model class information
    model_type = metadata.get("model_type")
    model_module = metadata.get("model_module")

    if not model_type or not model_module:
        raise ValueError(f"Missing model type or module information in {filepath}")

    try:
        # Dynamically import the module and get the class
        module = importlib.import_module(model_module)
        model_class = getattr(module, model_type)
    except (ImportError, AttributeError) as e:
        print(f"Error importing model class {model_type} from {model_module}: {e}")
        raise ValueError(f"Could not import model class. Make sure the model module is available.")

    # Get initialization parameters from metadata
    model_init_params = metadata.get("model_init_params", {})

    # Try to create model instance
    try:
        model = model_class(**model_init_params)
    except Exception as e:
        print(f"Error creating model instance: {e}")
        print("Trying to create model with default parameters...")
        try:
            model = model_class()
        except Exception as e:
            print(f"Failed to create model even with default parameters: {e}")
            raise ValueError(f"Could not instantiate model class {model_type}. Check if the model class definition matches saved parameters.")

    # Load state dictionary
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading state dictionary: {e}")
        # Try with strict=False as fallback
        print("Trying to load state dictionary with strict=False...")
        model.load_state_dict(state_dict, strict=False)
        print("Warning: Model loaded with strict=False. Some parameters may not have been restored correctly.")

    # Move model to device
    model = model.to(device)

    # Print model information
    print(f"Loaded {model_type} model from {filepath}")
    print(f"  Saved date: {metadata.get('saved_date', 'Unknown')}")
    print(f"  PyTorch version used for saving: {metadata.get('pytorch_version', 'Unknown')}")
    if "temporal_features" in metadata:
        print(f"  Temporal features: {', '.join(metadata['temporal_features'])}")
    if "static_features" in metadata:
        print(f"  Static features: {', '.join(metadata['static_features'])}")
    if "target_field" in metadata:
        print(f"  Target field: {metadata['target_field']}")
    if "time_feature_keys" in metadata:
        print(f"  Time feature keys: {', '.join(metadata['time_feature_keys'])}")
    if "wandb_url" in metadata:
        print(f"  Weights & Biases run: {metadata['wandb_url']}")

    return model, metadata
