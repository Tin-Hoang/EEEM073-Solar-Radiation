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
WANDB_USERNAME = "tin-hoang"
WANDB_PROJECT = "EEEM073-Solar-Radiation"

def setup_wandb(username=None, project=None):
    """
    Set up Weights & Biases tracking

    Args:
        username: Wandb username (default: None)
        project: Wandb project name (default: None)

    Returns:
        bool: Whether wandb is enabled
    """
    global USE_WANDB, WANDB_USERNAME, WANDB_PROJECT

    # Use provided values or defaults
    wandb_username = username or WANDB_USERNAME
    wandb_project = project or WANDB_PROJECT

    # Only enable wandb if both username and project are provided
    if wandb_username and wandb_project:
        USE_WANDB = True
        print(f"Weights & Biases tracking enabled with username '{wandb_username}' and project '{wandb_project}'")
    else:
        USE_WANDB = False
        print("Weights & Biases tracking disabled. Provide both username and project to enable.")

    return USE_WANDB

def track_experiment(func):
    """
    Decorator to track experiments with wandb

    This decorator wraps training functions to automatically log metrics to wandb
    when wandb tracking is enabled.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract model name if available
        model_name = kwargs.get('model_name', 'Model')

        # Start wandb run if tracking is enabled
        if USE_WANDB:
            # Create a unique run name
            run_name = f"{model_name}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

            # Get config parameters from kwargs
            config = {
                'model_name': model_name,
                'epochs': kwargs.get('epochs', 50),
                'patience': kwargs.get('patience', 10),
                'learning_rate': kwargs.get('lr', 0.001),
            }

            # Add physics parameters if they exist
            if 'lambda_night' in kwargs:
                config.update({
                    'lambda_night': kwargs.get('lambda_night'),
                    'lambda_neg': kwargs.get('lambda_neg'),
                    'lambda_clear': kwargs.get('lambda_clear'),
                })

            # Initialize wandb
            wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_USERNAME,
                name=run_name,
                config=config
            )

            # Run the original function
            history = func(*args, **kwargs)

            # Close wandb run
            wandb.finish()
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

    # Log training history to wandb
    if USE_WANDB:
        # Create a dataframe of training metrics
        history_df = pd.DataFrame({
            'epoch': list(range(len(history['train_loss']))),
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'train_mae': history['train_mae'],
            'val_mae': history['val_mae']
        })

        # Start a new wandb run if not already active
        if wandb.run is None:
            wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_USERNAME,
                name=f"{model_name}-History",
                config={"model": model_name}
            )

        # Log as a table
        wandb.log({f"{model_name}_history": wandb.Table(dataframe=history_df)})

        # Capture and log the matplotlib figure
        history_fig = plt.gcf()
        wandb.log({f"{model_name}_history_plot": wandb.Image(history_fig)})

        # Create custom interactive line charts
        for metric in ['loss', 'mae']:
            data = [[e, history[f'train_{metric}'][e], history[f'val_{metric}'][e]]
                   for e in range(len(history[f'train_{metric}']))]
            table = wandb.Table(columns=["epoch", f"train_{metric}", f"val_{metric}"], data=data)
            wandb.log({f"{model_name}_{metric}_curve": table})

        # If this function started a wandb run, finish it
        if wandb.run.name == f"{model_name}-History":
            wandb.finish()

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
    nighttime = metrics['nighttime'].flatten() > 0.5

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
