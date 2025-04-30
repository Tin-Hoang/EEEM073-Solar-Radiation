"""
Weights & Biases (wandb) utilities for experiment tracking.
"""

import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import wandb
from functools import wraps

# Default settings
USE_WANDB = False
KEEP_RUN_OPEN = False
WANDB_USERNAME = "tin-hoang"
WANDB_PROJECT = "EEEM073-Solar-Radiation"


def is_wandb_enabled():
    """
    Check if wandb is enabled (with an active run)

    Returns:
        bool: True if wandb is enabled and has an active run
    """
    is_enabled = USE_WANDB and wandb.run is not None
    return is_enabled


def set_wandb_flag(value):
    """
    Set global USE_WANDB flag

    Args:
        value: Boolean value to set

    Returns:
        bool: The new value
    """
    global USE_WANDB
    USE_WANDB = bool(value)
    return USE_WANDB


def set_keep_run_open(value):
    """
    Set global KEEP_RUN_OPEN flag

    Args:
        value: Boolean value to set

    Returns:
        bool: The new value
    """
    global KEEP_RUN_OPEN
    KEEP_RUN_OPEN = bool(value)
    return KEEP_RUN_OPEN


def setup_wandb(username=None, project=None, force_enable=False):
    """
    Set up Weights & Biases tracking

    Args:
        username: Wandb username (default: None)
        project: Wandb project name (default: None)
        force_enable: Force enable wandb even if already configured

    Returns:
        bool: Whether wandb is enabled
    """
    global USE_WANDB, WANDB_USERNAME, WANDB_PROJECT

    # Use provided values or defaults
    wandb_username = username or WANDB_USERNAME
    wandb_project = project or WANDB_PROJECT

    # Only enable wandb if both username and project are provided
    if wandb_username and wandb_project:
        # Set the global flag
        set_wandb_flag(True)

        print(f"Weights & Biases tracking enabled with username '{wandb_username}' and project '{wandb_project}'")
        print(f"USE_WANDB flag is now: {USE_WANDB}")

        # Initialize wandb if needed or forced
        if force_enable or wandb.run is None:
            try:
                wandb.init(
                    project=wandb_project,
                    entity=wandb_username,
                    name=f"Manual-Init-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
                    config={
                        "initialized_by": "setup_wandb",
                        "force_enable": force_enable
                    }
                )
                print(f"Successfully initialized wandb run: {wandb.run.name}")
            except Exception as e:
                print(f"Error initializing wandb: {e}")
    else:
        # If parameters are missing, disable wandb
        set_wandb_flag(False)
        print("Weights & Biases tracking disabled. Provide both username and project to enable.")

    return USE_WANDB


def initialize_wandb(model_name="Model", **config_params):
    """
    Initialize a new wandb run if needed

    Args:
        model_name: Name of the model for the run name
        **config_params: Additional configuration parameters

    Returns:
        bool: Whether initialization was successful
    """
    global USE_WANDB, WANDB_USERNAME, WANDB_PROJECT

    if not USE_WANDB:
        print("WARNING: initialize_wandb called but USE_WANDB is False, will not initialize")
        return False

    # Don't initialize if we already have a run
    if wandb.run is not None:
        print(f"wandb already initialized with run name: {wandb.run.name}")
        return True

    # Create a unique run name
    run_name = f"{model_name}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    try:
        # Initialize wandb
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_USERNAME,
            name=run_name,
            config=config_params
        )
        print(f"Successfully initialized wandb run: {wandb.run.name}")
        return True
    except Exception as e:
        print(f"Error initializing wandb: {e}")
        return False


def track_experiment(func):
    """
    Decorator to track experiments with wandb

    This decorator wraps training functions to automatically log metrics to wandb
    when wandb tracking is enabled.

    The decorated function accepts an additional keyword parameter:
    - keep_run_open (bool): If True, don't close the wandb run after the function completes
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract model name if available
        model_name = kwargs.get('model_name', 'Model')

        # Print debug info
        print(f"track_experiment: USE_WANDB={USE_WANDB}, wandb.run={wandb.run}, keep_run_open={KEEP_RUN_OPEN}")

        # Start wandb run if tracking is enabled
        if USE_WANDB:
            train_config = kwargs.get('config', None)
            temporal_features_shape = train_config.get('TEMPORAL_FEATURES_SHAPE', None)
            static_features_shape = train_config.get('STATIC_FEATURES_SHAPE', None)
            # Try to find the model - check common parameter names and args
            model = kwargs.get('model', None)
            # If model wasn't found in kwargs, check if it might be the first positional argument
            if model is None and len(args) > 0:
                model = args[0]

            try:
                # Import get_model_summary here to avoid circular imports
                from utils.model_utils import get_model_summary
                model_summary = get_model_summary(model,
                                                temporal_features_shape,
                                                static_features_shape)
                model_summary = repr(model_summary)
            except:
                model_summary = repr(model)
            # Get model dictionary
            try:
                model_dict = dict(model.__dict__)
            except:
                model_dict = {}
            # Create config parameters from kwargs
            config = {
                'model_name': model_name,
                'epochs': kwargs.get('epochs', 50),
                'patience': kwargs.get('patience', 10),
                'learning_rate': kwargs.get('lr', 0.001),
                'config': train_config,
                'model_architecture': model_summary,
                'model_dict': model_dict
            }

            # Create the run if it doesn't exist
            if wandb.run is None:
                print(f"Creating new wandb run for {model_name}")

                # Create a unique run name
                run_name = f"{model_name}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

                # Initialize wandb
                wandb.init(
                    project=WANDB_PROJECT,
                    entity=WANDB_USERNAME,
                    name=run_name,
                    config=config
                )

                # Flag to indicate this wrapper created the run
                wrapper_created_run = True
            else:
                print(f"Using existing wandb run: {wandb.run.name}")
                wrapper_created_run = False
            # Run the original function
            history = func(*args, **kwargs)

            # Close wandb run if we created it and keep_run_open is False
            if wrapper_created_run and not KEEP_RUN_OPEN:
                print(f"Finishing wandb run from track_experiment (keep_run_open={KEEP_RUN_OPEN})")
                wandb.finish()
            elif KEEP_RUN_OPEN:
                print(f"Keeping wandb run open as requested (keep_run_open={KEEP_RUN_OPEN})")

            return history
        else:
            # Just run the function without wandb tracking
            return func(*args, **kwargs)

    return wrapper

def plot_training_history(history, model_name=""):
    """
    Plot training and validation loss history

    Args:
        history: Dictionary of training history
        model_name: Name of the model for the plot title
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title(f'{model_name} - Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_mae'], label='Train')
    plt.plot(history['val_mae'], label='Validation')
    plt.title(f'{model_name} - MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def create_evaluation_plots(metrics, model_name=''):
    """
    Create evaluation plots for wandb logging

    Args:
        metrics: Dictionary of evaluation metrics
        model_name: Name of the model

    Returns:
        fig: Matplotlib figure object
    """
    y_true = metrics['y_true']
    y_pred = metrics['y_pred']
    nighttime = metrics['nighttime_mask'].flatten() > 0.5

    # Calculate residuals
    residuals = y_true - y_pred

    # Sample a subset for visualization
    max_samples = 1000
    if len(y_true) > max_samples:
        sample_indices = np.random.choice(len(y_true), max_samples, replace=False)
        y_true_sample = y_true[sample_indices]
        y_pred_sample = y_pred[sample_indices]
        residuals_sample = residuals[sample_indices]
        nighttime_sample = nighttime[sample_indices]
    else:
        y_true_sample = y_true
        y_pred_sample = y_pred
        residuals_sample = residuals
        nighttime_sample = nighttime

    fig = plt.figure(figsize=(15, 12))

    # Actual vs Predicted (colored by time of day)
    ax1 = fig.add_subplot(2, 2, 1)
    day_indices = ~nighttime_sample
    night_indices = nighttime_sample

    ax1.scatter(y_true_sample[day_indices], y_pred_sample[day_indices],
                alpha=0.5, c='skyblue', label='Daytime')
    ax1.scatter(y_true_sample[night_indices], y_pred_sample[night_indices],
                alpha=0.5, c='navy', label='Nighttime')

    max_val = max(np.max(y_true_sample), np.max(y_pred_sample))
    ax1.plot([0, max_val], [0, max_val], 'r--')
    ax1.set_title(f'{model_name} - Actual vs Predicted GHI')
    ax1.set_xlabel('Actual GHI (W/m²)')
    ax1.set_ylabel('Predicted GHI (W/m²)')
    ax1.legend()
    ax1.grid(True)

    # Histogram of residuals
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.hist(residuals_sample, bins=50, alpha=0.7, color='green')
    ax2.axvline(x=0, color='r', linestyle='--')
    ax2.set_title(f'{model_name} - Residuals Distribution')
    ax2.set_xlabel('Residual (W/m²)')
    ax2.set_ylabel('Frequency')
    ax2.grid(True)

    # Residuals vs Predicted
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(y_pred_sample, residuals_sample, alpha=0.5, color='purple')
    ax3.axhline(y=0, color='r', linestyle='--')
    ax3.set_title(f'{model_name} - Residuals vs Predicted')
    ax3.set_xlabel('Predicted GHI (W/m²)')
    ax3.set_ylabel('Residual (W/m²)')
    ax3.grid(True)

    # Metrics summary
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    metrics_text = f"Overall Metrics:\n"
    metrics_text += f"  MSE: {metrics['mse']:.2f}\n"
    metrics_text += f"  RMSE: {metrics['rmse']:.2f}\n"
    metrics_text += f"  MAE: {metrics['mae']:.2f}\n"
    metrics_text += f"  R²: {metrics['r2']:.4f}\n\n"

    metrics_text += f"Daytime Metrics:\n"
    metrics_text += f"  MSE: {metrics['day_mse']:.2f}\n"
    metrics_text += f"  RMSE: {metrics['day_rmse']:.2f}\n"
    metrics_text += f"  MAE: {metrics['day_mae']:.2f}\n"
    metrics_text += f"  R²: {metrics['day_r2']:.4f}\n\n"

    metrics_text += f"Nighttime Metrics:\n"
    metrics_text += f"  MSE: {metrics['night_mse']:.2f}\n"
    metrics_text += f"  RMSE: {metrics['night_rmse']:.2f}\n"
    metrics_text += f"  MAE: {metrics['night_mae']:.2f}\n"
    metrics_text += f"  R²: {metrics['night_r2']:.4f if not np.isnan(metrics['night_r2']) else 'N/A'}\n\n"

    metrics_text += f"Residual Stats:\n"
    metrics_text += f"  Mean: {np.mean(residuals):.2f}\n"
    metrics_text += f"  StdDev: {np.std(residuals):.2f}\n"

    ax4.text(0.05, 0.95, metrics_text, fontsize=10, va='top')

    plt.tight_layout()
    return fig

def save_models_with_metadata(models, model_names, metrics_list, log_to_wandb=True):
    """
    Save trained models with performance metadata

    Args:
        models: List of PyTorch models
        model_names: List of model names
        metrics_list: List of metrics dictionaries for each model
        log_to_wandb: Whether to log models as wandb artifacts
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"models_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # Save each model with its metadata
    for model, name, metrics in zip(models, model_names, metrics_list):
        # Create unique model filename
        model_filename = f"{name.lower().replace('-', '_')}_model.pt"
        model_path = os.path.join(save_dir, model_filename)

        # Create metadata for the model
        metadata = {
            "model_name": name,
            "saved_date": timestamp,
            "performance": {
                "rmse": float(metrics['rmse']),
                "mae": float(metrics['mae']),
                "r2": float(metrics['r2']),
                "day_rmse": float(metrics['day_rmse']),
                "day_mae": float(metrics['day_mae']),
                "day_r2": float(metrics['day_r2'])
            }
        }

        # Save the model
        torch.save({
            'model_state_dict': model.state_dict(),
            'metadata': metadata
        }, model_path)

        # Create metadata JSON file
        metadata_path = os.path.join(save_dir, f"{name.lower().replace('-', '_')}_metadata.json")
        with open(metadata_path, 'w') as f:
            import json
            json.dump(metadata, f, indent=2)

        print(f"Model {name} saved to {model_path}")

        # Log as wandb artifact if enabled
        if USE_WANDB and log_to_wandb:
            wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_USERNAME,
                name=f"{name}-Model-Save",
                config=metadata
            )

            # Create artifact
            artifact = wandb.Artifact(
                name=f"{name.lower().replace('-', '_')}_model",
                type="model",
                description=f"Trained {name} model for GHI forecasting"
            )

            # Add model file to artifact
            artifact.add_file(model_path)

            # Add metadata file to artifact
            artifact.add_file(metadata_path)

            # Log artifact
            wandb.log_artifact(artifact)

            # Finish wandb run
            wandb.finish()

    print(f"All models saved successfully to {save_dir}")
    return save_dir
