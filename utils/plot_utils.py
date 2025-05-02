"""
Plotting utilities for GHI forecasting visualization.
"""

import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
import torch
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import copy


def plot_time_series(data, location_idx=0, title=None, features=None, target_variable="ghi", n_steps=None, start_idx=0):
    """
    Plot time series data for a single location with the option to select a subset of timesteps

    Args:
        data: Data dictionary
        location_idx: Index of location to plot
        title: Plot title
        features: List of features to plot (if None, will use available features)
        target_variable: Name of the target variable
        n_steps: Number of timesteps to show (None for all)
        start_idx: Starting timestep index

    Returns:
        fig: Matplotlib figure object
    """
    # Determine what features to plot
    if features is None:
        # If specific features aren't provided, use what's available in the data
        available_features = []
        for feature in data.keys():
            if isinstance(data[feature], np.ndarray) and feature not in ['timestamps', 'time_index',
                                                                         'latitude', 'longitude',
                                                                         'elevation', 'timezone',
                                                                         'coordinates', 'meta']:
                if len(data[feature].shape) > 1 and data[feature].shape[1] > location_idx:
                    available_features.append(feature)
    else:
        # Use the provided features if they exist in the data
        available_features = [f for f in features if f in data and
                              isinstance(data[f], np.ndarray) and
                              len(data[f].shape) > 1 and
                              data[f].shape[1] > location_idx]

    # Add target variable if not already included
    if target_variable not in available_features and target_variable in data:
        if len(data[target_variable].shape) > 1 and data[target_variable].shape[1] > location_idx:
            available_features.append(target_variable)

    # Check if we have any features to plot
    if not available_features:
        print(f"No features available to plot for location index {location_idx}")
        return None

    # Check if timestamps are available
    if 'timestamps' not in data:
        print("No timestamps available for plotting")
        return None

    # Get timestamps and determine range to display
    timestamps = data['timestamps']
    total_timesteps = len(timestamps)

    if n_steps is None:
        # Show all timesteps if not specified
        end_idx = total_timesteps
    else:
        # Limit to specified range
        end_idx = min(start_idx + n_steps, total_timesteps)

    # Create the time slice for data selection
    time_slice = slice(start_idx, end_idx)

    # Get the timestamps for the selected range
    selected_timestamps = timestamps[time_slice]
    time_shape = len(selected_timestamps)

    # Create subplots
    n_features = len(available_features)
    fig, axes = plt.subplots(n_features, 1, figsize=(15, 4*n_features), sharex=True)

    if n_features == 1:
        axes = [axes]  # Make iterable if only one subplot

    for i, feature in enumerate(available_features):
        # Get feature data for this location within the selected time range
        feature_data = data[feature][time_slice, location_idx]

        # Check if shapes match
        if len(feature_data) != time_shape:
            # Shape mismatch - try to handle common cases
            if len(feature_data) > time_shape:
                # Feature has more data points than timestamps, truncate
                print(f"Warning: Feature {feature} has more data points ({len(feature_data)}) than timestamps ({time_shape}). Truncating.")
                feature_data = feature_data[:time_shape]
            else:
                # Feature has fewer data points than timestamps, skip
                print(f"Warning: Feature {feature} has fewer data points ({len(feature_data)}) than timestamps ({time_shape}). Skipping this feature.")
                continue

        # Convert data types and handle scaling for better visualization
        if feature_data.dtype in [np.uint8, np.uint16, np.int8, np.int16]:
            # These are likely scaled values, convert to float for display
            if feature in ['air_temperature', 'dew_point']:
                # These are often stored as °C * 100
                feature_data = feature_data.astype(float) / 100.0
            elif feature in ['ghi', 'dni', 'dhi', 'clearsky_ghi', 'clearsky_dni', 'clearsky_dhi']:
                # Radiation values are often stored as W/m² * 10
                feature_data = feature_data.astype(float) / 10.0
            elif feature in ['relative_humidity']:
                # Humidity values are often stored as percentage * 100
                feature_data = feature_data.astype(float) / 100.0
            else:
                # Generic scaling for other features
                feature_data = feature_data.astype(float)

        # Plot the data
        axes[i].plot(selected_timestamps, feature_data, '-', label=feature)
        axes[i].set_ylabel(feature)
        if i == 0:
            axes[i].set_title(title or f'Time Series for Location {location_idx} (Steps {start_idx} to {end_idx-1})')
        axes[i].grid(True)
        axes[i].legend()

    axes[-1].set_xlabel('Time')
    plt.tight_layout()

    # Print location information
    try:
        if 'latitude' in data and 'longitude' in data:
            lat = data['latitude'][location_idx]
            lon = data['longitude'][location_idx]
            print(f"Location: Latitude {lat:.4f}, Longitude {lon:.4f}", end="")

            if 'elevation' in data:
                elevation = data['elevation'][location_idx]
                print(f", Elevation {elevation}m", end="")

            if 'timezone' in data:
                timezone = data['timezone'][location_idx]
                print(f", Timezone UTC{'+' if timezone >= 0 else ''}{timezone}", end="")

            print()  # End the line
    except (KeyError, IndexError) as e:
        print(f"Could not display complete location information: {e}")

    print(f"Showing timesteps {start_idx} to {end_idx-1} out of {total_timesteps} total timesteps")

    return fig


def plot_solar_day_night(data, location_idx=0, n_steps=None, start_idx=0, show_threshold=True):
    """
    Plot solar zenith angle, nighttime mask, and GHI to verify day/night detection and solar radiation patterns

    Args:
        data: Data dictionary
        location_idx: Index of location to plot
        n_steps: Number of timesteps to show (None for all)
        start_idx: Starting timestep index
        show_threshold: Whether to show the 90° threshold line

    Returns:
        fig: Matplotlib figure object
    """
    # Check required data
    if 'solar_zenith_angle' not in data:
        print("Solar zenith angle not found in data.")
        return None

    if 'nighttime_mask' not in data:
        print("Nighttime mask not found in data.")
        return None

    if 'timestamps' not in data:
        print("Timestamps not found in data.")
        return None

    if 'ghi' not in data:
        print("GHI data not found. Will plot without GHI.")
        has_ghi = False
    else:
        has_ghi = True

    # Get timestamps and determine range to display
    timestamps = data['timestamps']
    total_timesteps = len(timestamps)

    if n_steps is None:
        # Show all timesteps if not specified
        end_idx = total_timesteps
    else:
        # Limit to specified range
        end_idx = min(start_idx + n_steps, total_timesteps)

    # Create the time slice for data selection
    time_slice = slice(start_idx, end_idx)

    # Get the timestamps for the selected range
    selected_timestamps = timestamps[time_slice]

    # Get solar zenith angle and nighttime mask data
    solar_zenith = data['solar_zenith_angle'][time_slice, location_idx].copy()
    nighttime = data['nighttime_mask'][time_slice, location_idx].copy()

    # Get GHI data if available
    if has_ghi:
        ghi_data = data['ghi'][time_slice, location_idx].copy()
        # Check if GHI needs scaling (common in datasets)
        if ghi_data.dtype in [np.int16, np.uint16, np.int32, np.uint32] and np.max(ghi_data) > 2000:
            ghi_data = ghi_data.astype(float) / 10.0
            print(f"Applied scaling factor of 10 to GHI data")

    # Important: Check the actual range of the solar zenith angle data
    sza_min = np.min(solar_zenith)
    sza_max = np.max(solar_zenith)
    print(f"Raw Solar Zenith Angle range: {sza_min} to {sza_max} (dtype: {solar_zenith.dtype})")

    # Check if scaling is needed based on data range and type
    scaled = False

    # Option 1: SZA stored as degrees * 100 (common in int16/uint16 datasets)
    if solar_zenith.dtype in [np.int16, np.uint16, np.int32, np.uint32] and sza_max > 180:
        solar_zenith = solar_zenith.astype(float) / 100.0
        print(f"Applied scaling factor of 100 to solar zenith angle data")
        print(f"Scaled Solar Zenith Angle range: {np.min(solar_zenith)} to {np.max(solar_zenith)} degrees")
        scaled = True

    # Option 2: SZA stored in radians instead of degrees (less common)
    elif sza_max < np.pi and sza_max > 0:
        solar_zenith = np.rad2deg(solar_zenith)
        print(f"Converted solar zenith angle from radians to degrees")
        print(f"Converted Solar Zenith Angle range: {np.min(solar_zenith)} to {np.max(solar_zenith)} degrees")
        scaled = True

    # If SZA is very small (near 0) and we're still in daytime, try a reciprocal interpretation
    # This is a special case where some datasets use the cosine of zenith angle instead
    elif sza_max < 10 and np.mean(nighttime) < 0.5:
        print(f"WARNING: Solar zenith angle values are very small. Data might be cosine of zenith angle.")
        print(f"Trying to convert from cosine of zenith angle to zenith angle in degrees.")

        # Only apply if it doesn't have extreme values that would break arccos
        if np.all((solar_zenith >= -1.0) & (solar_zenith <= 1.0)):
            # Convert from cosine of zenith angle to zenith angle in degrees
            solar_zenith = np.arccos(np.clip(solar_zenith, -1.0, 1.0)) * 180 / np.pi
            print(f"Converted cosine values to Solar Zenith Angle in degrees")
            print(f"Calculated Solar Zenith Angle range: {np.min(solar_zenith)} to {np.max(solar_zenith)} degrees")
            scaled = True
        else:
            print(f"WARNING: Values outside valid cosine range [-1, 1]")

    # If we couldn't determine the right scaling, warn the user
    if not scaled:
        print(f"WARNING: Couldn't determine appropriate scaling for solar zenith angle.")
        print(f"Display may be incorrect. SZA should be in degrees with ~0-180 range.")

    # Create figure with 3 subplots if GHI is available, otherwise 2 subplots
    n_plots = 3 if has_ghi else 2
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4*n_plots), sharex=True)

    # Plot solar zenith angle
    axes[0].plot(selected_timestamps, solar_zenith, 'b-', label='Solar Zenith Angle')
    axes[0].set_ylabel('Solar Zenith Angle (degrees)')
    axes[0].set_title(f'Solar Zenith Angle and Nighttime Mask for Location {location_idx}')
    axes[0].grid(True)

    # Add horizontal line at 90 degrees if requested
    if show_threshold:
        axes[0].axhline(y=90, color='r', linestyle='--', label='90° Threshold')

    # Set reasonable y-limits based on the data
    if np.max(solar_zenith) > 0:
        y_max = min(180, np.max(solar_zenith) * 1.1)
        axes[0].set_ylim(0, y_max)

    axes[0].legend()

    # Plot nighttime mask
    axes[1].plot(selected_timestamps, nighttime, 'g-', label='Nighttime Mask')
    axes[1].set_ylabel('Nighttime Mask')
    axes[1].set_ylim(-0.1, 1.1)  # Add some padding to better see the binary values
    axes[1].grid(True)
    axes[1].legend()

    # Plot GHI if available
    if has_ghi:
        axes[2].plot(selected_timestamps, ghi_data, 'r-', label='GHI')
        axes[2].set_ylabel('GHI (W/m²)')
        axes[2].set_xlabel('Time')
        axes[2].grid(True)
        axes[2].legend()
    else:
        axes[1].set_xlabel('Time')

    # Add shaded regions to indicate nighttime (SZA >= 90°)
    for i in range(len(solar_zenith)):
        if solar_zenith[i] >= 90 or nighttime[i] > 0.5:
            # Add vertical span in all subplots
            for ax in axes:
                ax.axvspan(selected_timestamps[i],
                          selected_timestamps[min(i+1, len(selected_timestamps)-1)],
                          alpha=0.2, color='gray')

    # Print location information and statistics
    location_info = f"Location: "
    if 'latitude' in data and 'longitude' in data:
        lat = data['latitude'][location_idx]
        lon = data['longitude'][location_idx]
        location_info += f"Latitude {lat:.4f}, Longitude {lon:.4f}"

        if 'elevation' in data:
            elevation = data['elevation'][location_idx]
            location_info += f", Elevation {elevation}m"

        if 'timezone' in data:
            timezone = data['timezone'][location_idx]
            location_info += f", Timezone UTC{'+' if timezone >= 0 else ''}{timezone}"

    # Calculate and display statistics about day/night detection
    night_hours = np.sum(nighttime)
    total_hours = len(nighttime)
    day_hours = total_hours - night_hours

    print(location_info)
    print(f"Time period: {selected_timestamps[0]} to {selected_timestamps[-1]}")
    print(f"Day hours: {day_hours} ({day_hours/total_hours:.1%}), Night hours: {night_hours} ({night_hours/total_hours:.1%})")

    # Check for inconsistencies between solar zenith angle and nighttime mask
    inconsistencies = np.sum((solar_zenith >= 90) != (nighttime > 0.5))
    if inconsistencies > 0:
        print(f"WARNING: {inconsistencies} inconsistencies detected between solar zenith angle and nighttime mask!")

    plt.tight_layout()
    return fig


def plot_time_features(timestamps, time_features=None, n_days=7):
    """
    Visualize time features created by create_time_features

    Args:
        timestamps: Array of datetime objects
        time_features: Optional pre-computed time features. If None, will be generated using create_time_features.
                      Can be either a dictionary of individual features or a combined array
        n_days: Number of days to plot (default: 7)

    Returns:
        fig: Matplotlib figure object
    """
    # Limit to n_days
    if len(timestamps) > 24 * n_days:
        timestamps = timestamps[:24 * n_days]

    # Handle time_features format
    if time_features is not None:
        if isinstance(time_features, dict):
            # Dictionary format (new)
            hour_sin = time_features['hour_sin'][:24 * n_days] if 'hour_sin' in time_features else None
            hour_cos = time_features['hour_cos'][:24 * n_days] if 'hour_cos' in time_features else None
            day_sin = time_features['day_sin'][:24 * n_days] if 'day_sin' in time_features else None
            day_cos = time_features['day_cos'][:24 * n_days] if 'day_cos' in time_features else None
            month_sin = time_features['month_sin'][:24 * n_days] if 'month_sin' in time_features else None
            month_cos = time_features['month_cos'][:24 * n_days] if 'month_cos' in time_features else None
            dow_sin = time_features['dow_sin'][:24 * n_days] if 'dow_sin' in time_features else None
            dow_cos = time_features['dow_cos'][:24 * n_days] if 'dow_cos' in time_features else None
        else:
            # Array format (old)
            time_features = time_features[:24 * n_days]
            hour_sin, hour_cos = time_features[:, 0], time_features[:, 1]
            day_sin, day_cos = time_features[:, 2], time_features[:, 3]
            month_sin, month_cos = time_features[:, 4], time_features[:, 5]
            dow_sin, dow_cos = time_features[:, 6], time_features[:, 7]
    else:
        # Generate time features from timestamps
        from utils.normalize_utils import create_time_features
        time_features_dict = create_time_features(timestamps)
        hour_sin = time_features_dict['hour_sin']
        hour_cos = time_features_dict['hour_cos']
        day_sin = time_features_dict['day_sin']
        day_cos = time_features_dict['day_cos']
        month_sin = time_features_dict['month_sin']
        month_cos = time_features_dict['month_cos']
        dow_sin = time_features_dict['dow_sin']
        dow_cos = time_features_dict['dow_cos']

    # Create figure
    fig, axs = plt.subplots(4, 2, figsize=(15, 12), sharex=True)
    fig.suptitle('Cyclical Time Features', fontsize=16)

    # Plot features
    axs[0, 0].plot(timestamps, hour_sin, label='Hour (sin)')
    axs[0, 0].set_ylabel('Hour (sin)')

    axs[0, 1].plot(timestamps, hour_cos, label='Hour (cos)', color='orange')
    axs[0, 1].set_ylabel('Hour (cos)')

    axs[1, 0].plot(timestamps, day_sin, label='Day of Year (sin)')
    axs[1, 0].set_ylabel('Day of Year (sin)')

    axs[1, 1].plot(timestamps, day_cos, label='Day of Year (cos)', color='orange')
    axs[1, 1].set_ylabel('Day of Year (cos)')

    axs[2, 0].plot(timestamps, month_sin, label='Month (sin)')
    axs[2, 0].set_ylabel('Month (sin)')

    axs[2, 1].plot(timestamps, month_cos, label='Month (cos)', color='orange')
    axs[2, 1].set_ylabel('Month (cos)')

    axs[3, 0].plot(timestamps, dow_sin, label='Day of Week (sin)')
    axs[3, 0].set_ylabel('Day of Week (sin)')

    axs[3, 1].plot(timestamps, dow_cos, label='Day of Week (cos)', color='orange')
    axs[3, 1].set_ylabel('Day of Week (cos)')

    # Format x-axis
    for ax in axs.flatten():
        ax.grid(True, alpha=0.3)

    # Format x-axis dates better
    date_formatter = DateFormatter('%Y-%m-%d %H:%M')
    for ax in axs[-1]:
        ax.xaxis.set_major_formatter(date_formatter)
        ax.xaxis.set_major_locator(mdates.DayLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    return fig


def plot_training_history(history, model_name=""):
    """
    Plot training and validation loss history

    Args:
        history: Dictionary of training history
        model_name: Name of the model for the plot title
    """
    # Check if sample counts are available
    has_sample_counts = 'train_samples' in history and 'val_samples' in history

    if has_sample_counts:
        # Create a 2x2 figure for total and average metrics
        fig = plt.figure(figsize=(16, 12))

        # Total Loss (MSE)
        plt.subplot(2, 2, 1)
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['val_loss'], label='Validation')
        plt.title(f'{model_name} - Total Loss (MSE)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Average Loss per sample
        plt.subplot(2, 2, 2)
        # Calculate average loss per sample
        train_avg_loss = [loss/samples for loss, samples in zip(history['train_loss'], history['train_samples'])]
        val_avg_loss = [loss/samples for loss, samples in zip(history['val_loss'], history['val_samples'])]
        plt.plot(train_avg_loss, label='Train')
        plt.plot(val_avg_loss, label='Validation')
        plt.title(f'{model_name} - Average Loss per Sample')
        plt.xlabel('Epoch')
        plt.ylabel('Avg Loss per Sample')
        plt.legend()
        plt.grid(True)

        # Total MAE
        plt.subplot(2, 2, 3)
        plt.plot(history['train_mae'], label='Train')
        plt.plot(history['val_mae'], label='Validation')
        plt.title(f'{model_name} - Total MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True)

        # Average MAE per sample
        plt.subplot(2, 2, 4)
        # Calculate average MAE per sample
        train_avg_mae = [mae/samples for mae, samples in zip(history['train_mae'], history['train_samples'])]
        val_avg_mae = [mae/samples for mae, samples in zip(history['val_mae'], history['val_samples'])]
        plt.plot(train_avg_mae, label='Train')
        plt.plot(val_avg_mae, label='Validation')
        plt.title(f'{model_name} - Average MAE per Sample')
        plt.xlabel('Epoch')
        plt.ylabel('Avg MAE per Sample')
        plt.legend()
        plt.grid(True)
    else:
        # Original 1x2 figure if sample counts aren't available
        fig = plt.figure(figsize=(12, 5))

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
    return fig


def plot_evaluation_metrics(metrics, model_name=''):
    """
    Create evaluation metrics plots

    Args:
        metrics: Dictionary of evaluation metrics
        model_name: Name of the model

    Returns:
        fig: Matplotlib figure object
    """
    y_true = metrics['y_true']
    y_pred = metrics['y_pred']

    # Check if we have meaningful nighttime data (any non-zero values)
    has_nighttime = metrics['nighttime_mask'] is not None and np.any(metrics['nighttime_mask'] > 0.5)
    if has_nighttime:
        nighttime = metrics['nighttime_mask'].flatten() > 0.5
    else:
        # Create all daytime mask
        nighttime = np.zeros(len(y_true), dtype=bool)

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

    # Check if inference speed metrics are available
    has_inference_speed = 'samples_per_second' in metrics and 'avg_time_per_sample' in metrics

    # Create figure with 2x2 layout (removed separate inference speed plots)
    fig = plt.figure(figsize=(15, 12))

    # Actual vs Predicted (colored by time of day)
    ax1 = fig.add_subplot(2, 2, 1)
    day_indices = ~nighttime_sample
    night_indices = nighttime_sample

    ax1.scatter(y_true_sample[day_indices], y_pred_sample[day_indices],
                alpha=0.5, c='skyblue', label='Daytime')

    if has_nighttime and np.any(night_indices):
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
    metrics_text += f"  WAPE: {metrics['wape']:.2f}\n"
    metrics_text += f"  R²: {metrics['r2']:.4f}\n\n"

    metrics_text += f"Daytime Metrics:\n"
    metrics_text += f"  MSE: {metrics['day_mse']:.2f}\n"
    metrics_text += f"  RMSE: {metrics['day_rmse']:.2f}\n"
    metrics_text += f"  MAE: {metrics['day_mae']:.2f}\n"
    metrics_text += f"  WAPE: {metrics['day_wape']:.2f}\n"
    metrics_text += f"  R²: {metrics['day_r2']:.4f}\n\n"

    if has_nighttime:
        metrics_text += f"Nighttime Metrics:\n"
        metrics_text += f"  MSE: {metrics['night_mse']:.2f}\n"
        metrics_text += f"  RMSE: {metrics['night_rmse']:.2f}\n"
        metrics_text += f"  MAE: {metrics['night_mae']:.2f}\n"
        metrics_text += f"  WAPE: {metrics['night_wape']:.2f}\n"
        # Fix the f-string formatting - move conditional outside format specifier
        r2_str = f"{metrics['night_r2']:.4f}" if not np.isnan(metrics['night_r2']) else "N/A"
        metrics_text += f"  R²: {r2_str}\n\n"
    else:
        metrics_text += f"Nighttime Metrics: Not available\n\n"

    metrics_text += f"Residual Stats:\n"
    metrics_text += f"  Mean: {np.mean(residuals):.2f}\n"
    metrics_text += f"  StdDev: {np.std(residuals):.2f}\n"

    # Add inference speed metrics to the text summary if available
    if has_inference_speed:
        metrics_text += f"\nInference Speed:\n"
        metrics_text += f"  Throughput: {metrics['samples_per_second']:.2f} samples/sec\n"
        metrics_text += f"  Latency: {metrics['avg_time_per_sample'] * 1000:.3f} ms/sample\n"

        # Add context about total number of samples and time if available
        if 'total_samples' in metrics and 'total_inference_time' in metrics:
            metrics_text += f"  Total samples: {metrics['total_samples']}\n"
            metrics_text += f"  Total time: {metrics['total_inference_time']:.4f} seconds\n"

        # Add device information
        metrics_text += f"  Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}\n"

    ax4.text(0.05, 0.95, metrics_text, fontsize=10, va='top')

    plt.tight_layout()
    return fig


def compare_models(model_metrics_dict, dataset_name=""):
    """
    Compare performance metrics across models

    Args:
        model_metrics_dict: Dictionary of {model_name: metrics} where metrics
                           is the output from evaluate_model function
        dataset_name: Name of the dataset (train/val/test)

    Returns:
        fig: Matplotlib figure with comparison plots
    """
    # Make a deep copy of the input dictionary to avoid modifying the original
    model_metrics_dict_copy = copy.deepcopy(model_metrics_dict)

    # Extract model names and metrics from the dictionary
    model_names = list(model_metrics_dict_copy.keys())
    metrics_list = [model_metrics_dict_copy[model] for model in model_names]

    if not model_names:
        raise ValueError("No models provided for comparison")

    # Define metrics to compare
    metrics = ['mse', 'rmse', 'mae', 'wape', 'r2']
    metric_labels = ['MSE', 'RMSE', 'MAE', 'WAPE', 'R²']

    # Include inference speed metrics if available
    inference_metrics = []
    inference_labels = []
    first_model_metrics = next(iter(model_metrics_dict_copy.values()))
    if 'samples_per_second' in first_model_metrics:
        inference_metrics = ['samples_per_second', 'avg_time_per_sample']
        inference_labels = ['Samples/sec', 'ms/sample']

        # Convert ms/sample for better display
        for metrics_dict in metrics_list:
            if 'avg_time_per_sample' in metrics_dict:
                metrics_dict['avg_time_per_sample'] *= 1000  # Convert to milliseconds

    # Create comparison DataFrame
    comparison = pd.DataFrame(index=metric_labels + inference_labels, columns=model_names)

    for i, model in enumerate(model_names):
        model_metrics = model_metrics_dict_copy[model]
        for j, metric in enumerate(metrics):
            if metric in model_metrics:
                comparison.iloc[j, i] = model_metrics[metric]
            else:
                comparison.iloc[j, i] = np.nan

        # Add inference metrics if available
        for j, metric in enumerate(inference_metrics):
            if metric in model_metrics:
                comparison.iloc[j + len(metrics), i] = model_metrics[metric]
            else:
                comparison.iloc[j + len(metrics), i] = np.nan

    print(f"\nModel Comparison - {dataset_name} Set:")
    print(comparison)

    # Create a figure with GridSpec layout
    fig = create_comparison_visualization(comparison, model_names, dataset_name)

    # Save the figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('plots', exist_ok=True)
    fig.savefig(f'plots/model_comparison_{dataset_name}_{timestamp}.png')

    return fig


def plot_radar_chart(ax, df, model_names, model_color_map, title):
    """
    Create a radar chart for metrics comparison

    Args:
        ax: Matplotlib axis
        df: DataFrame with metrics (can be either a dataframe with metrics as columns and models as index,
                                   or metrics as index and models as columns)
        model_names: List of model names
        model_color_map: Dictionary mapping model names to colors
        title: Chart title
    """
    # Metrics to include in radar chart
    metrics = ['MSE', 'RMSE', 'MAE', 'WAPE', 'R²']

    # Check dataframe orientation and adjust if needed
    if list(df.index) == model_names:
        # DataFrame has models as index, metrics as columns (daytime/nighttime format)
        df_oriented = df
    else:
        # DataFrame has metrics as index, models as columns (regular format)
        # Extract only the needed metrics and transpose to get models as index
        available_metrics = [m for m in metrics if m in df.index]
        df_oriented = df.loc[available_metrics].transpose()

    # Create a normalized version of the dataframe for radar chart
    normalized_df = pd.DataFrame(index=model_names, columns=metrics)

    # Define parameters for enhanced normalization
    SENSITIVITY_FACTOR = 2.0  # Controls how much differences are amplified
    MIN_RADIUS = 0.4  # Minimum radius for worst values (0.0-1.0)

    # Process each metric individually using enhanced normalization
    for metric in metrics:
        if metric in df_oriented.columns:
            values = df_oriented[metric]

            if metric == 'R²':
                # For metrics where higher is better
                min_val = values.min()
                max_val = values.max()

                if max_val > min_val:
                    # Calculate base normalization (0-1 scale)
                    base_norm = (values - min_val) / (max_val - min_val)
                    # Apply power transformation to amplify differences
                    normalized_values = MIN_RADIUS + (1 - MIN_RADIUS) * (base_norm ** (1/SENSITIVITY_FACTOR))
                else:
                    normalized_values = pd.Series(0.5, index=values.index)
            else:
                # For metrics where lower is better (MSE, RMSE, MAE)
                min_val = values.min()
                max_val = values.max()

                if max_val > min_val:
                    # Invert the normalization (so lower values get higher scores)
                    base_norm = 1 - (values - min_val) / (max_val - min_val)
                    # Apply power transformation
                    normalized_values = MIN_RADIUS + (1 - MIN_RADIUS) * (base_norm ** (1/SENSITIVITY_FACTOR))
                else:
                    normalized_values = pd.Series(0.5, index=values.index)

            normalized_df[metric] = normalized_values

    # Number of metrics on the radar chart
    N = len(metrics)

    # Compute the angles for the radar chart
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the polygon

    # Add metrics labels (with the first one repeated to close the polygon)
    metrics_labels = metrics + [metrics[0]]

    # Set up the radar chart
    ax.set_theta_offset(np.pi / 2)  # Start from top
    ax.set_theta_direction(-1)  # Clockwise

    # Plot each model
    for model in model_names:
        # Get the normalized values for this model
        values = normalized_df.loc[model].values.flatten().tolist()
        values += values[:1]  # Close the polygon

        # Plot the radar chart for this model
        ax.plot(angles, values, '-', linewidth=2, label=model, color=model_color_map[model])
        ax.fill(angles, values, alpha=0.1, color=model_color_map[model])

    # Set radar chart labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_labels[:-1])
    ax.set_title(title, size=12)

    # Add legend if not already present elsewhere
    if not hasattr(ax, 'legend_added'):
        legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98),
                           ncol=min(len(model_names), 5), fontsize=10)
        ax.legend_added = True


def create_comparison_visualization(comparison_df, model_names, dataset_name=""):
    """
    Create comprehensive comparison visualizations for multiple models

    Args:
        comparison_df: DataFrame with metrics
        model_names: List of model names
        dataset_name: Name of the dataset (train/val/test)

    Returns:
        fig: Matplotlib figure
    """
    # Check if inference metrics are present
    has_inference_metrics = False
    if 'Samples/sec' in comparison_df.index and 'ms/sample' in comparison_df.index:
        has_inference_metrics = True
    rows = 2
    cols = 2

    # Create a figure with appropriate layout
    fig = plt.figure(figsize=(18, 7 * rows))
    gs = GridSpec(rows, cols, figure=fig, height_ratios=[1] * rows, width_ratios=[1, 1],
                 hspace=0.3, wspace=0.3)

    # Define consistent colors for each model
    model_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    # Create a color map for models
    model_color_map = {model: model_colors[i % len(model_colors)] for i, model in enumerate(model_names)}

    # 1. Create performance metrics bar chart
    ax_metrics = fig.add_subplot(gs[0, 0])
    plot_performance_metrics_bar(ax_metrics, comparison_df, model_names, model_color_map)

    # 2. Create inference speed chart
    if has_inference_metrics:
        ax_speed = fig.add_subplot(gs[0, 1])
        plot_inference_speed_bar(ax_speed, comparison_df, model_names, model_color_map)

    # 3. Create radar chart
    ax_radar = fig.add_subplot(gs[1, 0], polar=True)
    plot_radar_chart(ax_radar, comparison_df, model_names, model_color_map, title='Performance Metrics Comparison')

    # 4. Create heatmap
    ax_heatmap = fig.add_subplot(gs[1, 1])
    plot_heatmap(ax_heatmap, comparison_df, title='Performance Metrics Heatmap')

    # Set overall title
    plt.suptitle(f'Model Comparison - {dataset_name} Dataset', fontsize=16, y=0.96)

    # Add legend below the title
    handles = [plt.Rectangle((0,0),1,1, color=model_color_map[model]) for model in model_names]
    fig.legend(handles, model_names, loc='upper center', ncol=min(6, len(model_names)),
               bbox_to_anchor=(0.5, 0.95), fontsize=12)

    # Adjust layout to make room for the legend below the title
    fig.subplots_adjust(top=0.90, bottom=0.10)

    return fig


def plot_performance_metrics_bar(ax, comparison_df, model_names, model_color_map):
    """
    Create a grouped bar chart for performance metrics comparison

    Args:
        ax: Matplotlib axis
        comparison_df: DataFrame with metrics
        model_names: List of model names
        model_color_map: Dictionary mapping model names to colors
    """
    performance_metrics = ['MSE', 'RMSE', 'MAE', 'WAPE', 'R²']

    # Create a copy of the metrics data
    scaled_metrics_df = comparison_df.loc[performance_metrics].copy()

    # Determine the scaling factor for MSE based on the max values of metrics
    mse_values = scaled_metrics_df.loc['MSE']
    mae_values = scaled_metrics_df.loc['MAE']
    r2_values = scaled_metrics_df.loc['R²']

    # Calculate scaling factor to make MSE values comparable to MAE
    scaling_factor_mse = 1
    if mse_values.max() > mae_values.max() * 10:
        # Determine appropriate scaling factor (10, 100, 1000, etc.)
        scaling_factor_mse = 10 ** (np.floor(np.log10(mse_values.max() / mae_values.max())))
        # Scale down MSE values
        scaled_metrics_df.loc['MSE'] = mse_values / scaling_factor_mse

    # Calculate scaling factor to make R² values comparable to MAE
    scaling_factor_r2 = 100

    # Scale up R² values
    scaled_metrics_df.loc['R²'] = r2_values * scaling_factor_r2

    # Set up the bar chart - grouped by metrics
    x = np.arange(len(performance_metrics))  # x positions for the metrics
    width = 0.8 / len(model_names)  # width of each bar, adjusted for number of models

    # Create a grouped bar chart with metrics on x-axis and models as groups
    rects_list = []
    for i, model in enumerate(model_names):
        # Position bars for each model within each metric group
        pos = x - 0.4 + (i + 0.5) * width
        values = [scaled_metrics_df.loc[metric, model] for metric in performance_metrics]
        rects = ax.bar(pos, values, width, color=model_color_map[model], label=model)
        rects_list.append((rects, model))

    # Add labels, title, and custom x-axis tick labels
    ax.set_xlabel('Performance Metrics')
    ax.set_ylabel('Value')
    ax.set_title('Performance Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([
        f'MSE{f" (÷{scaling_factor_mse:.0f})" if scaling_factor_mse > 1 else ""}',
        'RMSE',
        'MAE',
        'WAPE',
        f'R²{f" (×{scaling_factor_r2:.0f})" if scaling_factor_r2 != 1 else ""}'
    ])
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add data labels on bars
    for rects, model in rects_list:
        for i, rect in enumerate(rects):
            height = rect.get_height()
            metric = performance_metrics[i]
            orig_val = comparison_df.loc[metric, model]

            # Format label based on metric type
            if metric == 'MSE':
                label_text = f'{orig_val:.1f}'
            elif metric == 'R²':
                label_text = f'{orig_val:.4f}'
            else:
                label_text = f'{orig_val:.1f}'

            ax.text(rect.get_x() + rect.get_width()/2., height + 0.05*scaled_metrics_df.loc[metric].max(),
                     label_text, ha='center', va='bottom', fontsize=8, rotation=0)


def plot_inference_speed_bar(ax, comparison_df, model_names, model_color_map):
    """
    Create a bar chart for inference speed comparison

    Args:
        ax: Matplotlib axis
        comparison_df: DataFrame with metrics
        model_names: List of model names
        model_color_map: Dictionary mapping model names to colors
    """
    # Check if Samples/sec metric is available
    if 'Samples/sec' in comparison_df.index:
        # Get values
        values = comparison_df.loc['Samples/sec']

        # Create bar chart with consistent model colors
        bars = ax.bar(range(len(model_names)), values,
                      color=[model_color_map[model] for model in model_names])

        # Add data labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(values),
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)

        # Set labels and title
        ax.set_title('Inference Throughput Comparison')
        ax.set_ylabel('Samples/sec')
        ax.set_xlabel('Model')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=30, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)


def plot_heatmap(ax, df, title):
    """
    Create a heatmap for metrics comparison

    Args:
        ax: Matplotlib axis
        df: DataFrame with metrics (can be either metrics as rows and models as columns,
                                   or models as rows and metrics as columns)
        title: Chart title
    """
    # Check dataframe orientation
    metrics_list = ['MSE', 'RMSE', 'MAE', 'WAPE', 'R²', 'Samples/sec', 'ms/sample']

    # Determine if we need to transpose the dataframe
    # If the index contains mostly metric names, keep as is
    # If the columns contain mostly metric names, transpose
    metrics_in_index = sum(1 for m in metrics_list if m in df.index)
    metrics_in_columns = sum(1 for m in metrics_list if m in df.columns)

    if metrics_in_columns > metrics_in_index:
        # Transpose so metrics are in rows, models in columns
        df_oriented = df.transpose()
    else:
        # Already in the right format (metrics in rows, models in columns)
        df_oriented = df

    # Create a normalized version for the heatmap
    available_metrics = [m for m in df_oriented.index if m in metrics_list]
    heatmap_df = pd.DataFrame(index=available_metrics, columns=df_oriented.columns, dtype=float)

    # Normalize values between 0-1 for the heatmap (1 is always better)
    for metric in available_metrics:
        values = df_oriented.loc[metric].astype(float)  # Ensure numeric values
        min_val = values.min()
        max_val = values.max()

        if max_val > min_val:
            if metric in ['R²', 'Samples/sec']:  # Higher is better
                heatmap_df.loc[metric] = (values - min_val) / (max_val - min_val)
            else:  # Lower is better (MSE, RMSE, MAE, ms/sample)
                heatmap_df.loc[metric] = 1 - ((values - min_val) / (max_val - min_val))
        else:
            heatmap_df.loc[metric] = 0.5  # Default if all values are equal

    # Prepare annotations with original values
    annot = np.empty_like(heatmap_df.values, dtype='object')
    for i, metric in enumerate(heatmap_df.index):
        for j, model in enumerate(heatmap_df.columns):
            orig_val = df_oriented.loc[metric, model]
            if metric == 'R²':
                annot[i, j] = f"{orig_val:.4f}"
            elif metric == 'ms/sample':
                # Use more decimal places for very small values
                if orig_val < 0.01:
                    annot[i, j] = f"{orig_val:.5f}"
                else:
                    annot[i, j] = f"{orig_val:.2f}"
            else:
                annot[i, j] = f"{orig_val:.2f}"

    # Create heatmap with appropriate parameters based on dataframe size
    if len(heatmap_df.columns) > 10 or len(heatmap_df.index) > 10:
        # For larger heatmaps, disable colorbar to save space
        sns.heatmap(heatmap_df, annot=annot, fmt='', cmap='RdYlGn',
                    linewidths=0.5, cbar=False, ax=ax)
    else:
        # For smaller heatmaps, include colorbar with label
        sns.heatmap(heatmap_df, annot=annot, fmt='', cmap='RdYlGn',
                    cbar_kws={'label': 'Normalized Score (Higher is Better)'},
                    linewidths=0.5, ax=ax)

    ax.set_title(title)
    ax.set_yticklabels(heatmap_df.index, rotation=0)


def compare_models_daytime_nighttime(model_metrics_dict, dataset_name=""):
    """
    Create comprehensive comparison visualizations for multiple models,
    breaking down performance by daytime/nighttime/overall.

    Args:
        model_metrics_dict: Dictionary of {model_name: metrics} where metrics
                           is the output from evaluate_model function
        dataset_name: Name of the dataset (train/val/test)

    Returns:
        fig: Matplotlib figure with comparison plots
    """
    # Make a deep copy of the input dictionary to avoid modifying the original
    model_metrics_dict_copy = copy.deepcopy(model_metrics_dict)

    # Extract model names
    model_names = list(model_metrics_dict_copy.keys())

    if not model_names:
        raise ValueError("No models provided for comparison")

    # Create DataFrames for each time period (overall, daytime, nighttime)
    overall_data = {
        'MSE': [], 'RMSE': [], 'MAE': [], 'WAPE': [], 'R²': []
    }

    daytime_data = {
        'MSE': [], 'RMSE': [], 'MAE': [], 'WAPE': [], 'R²': []
    }

    nighttime_data = {
        'MSE': [], 'RMSE': [], 'MAE': [], 'WAPE': [], 'R²': []
    }

    # Extract metrics for each model
    for model_name in model_names:
        metrics = model_metrics_dict_copy[model_name]

        # Overall metrics
        overall_data['MSE'].append(metrics['mse'])
        overall_data['RMSE'].append(metrics['rmse'])
        overall_data['MAE'].append(metrics['mae'])
        overall_data['WAPE'].append(metrics['wape'])
        overall_data['R²'].append(metrics['r2'])

        # Daytime metrics
        daytime_data['MSE'].append(metrics['day_mse'])
        daytime_data['RMSE'].append(metrics['day_rmse'])
        daytime_data['MAE'].append(metrics['day_mae'])
        daytime_data['WAPE'].append(metrics['day_wape'])
        daytime_data['R²'].append(metrics['day_r2'])

        # Nighttime metrics
        nighttime_data['MSE'].append(metrics['night_mse'])
        nighttime_data['RMSE'].append(metrics['night_rmse'])
        nighttime_data['MAE'].append(metrics['night_mae'])
        nighttime_data['WAPE'].append(metrics['night_wape'])
        nighttime_data['R²'].append(metrics['night_r2'])

    # Create DataFrames
    overall_df = pd.DataFrame(overall_data, index=model_names)
    daytime_df = pd.DataFrame(daytime_data, index=model_names)
    nighttime_df = pd.DataFrame(nighttime_data, index=model_names)

    # Define grid layout without the combined heatmap
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1, 1],
                 hspace=0.3, wspace=0.3)

    # Define consistent colors for each model
    model_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    # Create a color map for models
    model_color_map = {model: model_colors[i % len(model_colors)] for i, model in enumerate(model_names)}

    # Plot radar charts for each time period in the first row
    ax1 = fig.add_subplot(gs[0, 0], polar=True)
    plot_radar_chart(ax1, overall_df[['MSE', 'RMSE', 'MAE', 'WAPE', 'R²']], model_names, model_color_map, title='Overall Performance')

    ax2 = fig.add_subplot(gs[0, 1], polar=True)
    plot_radar_chart(ax2, daytime_df, model_names, model_color_map, title='Daytime Performance')

    ax3 = fig.add_subplot(gs[0, 2], polar=True)
    plot_radar_chart(ax3, nighttime_df, model_names, model_color_map, title='Nighttime Performance')

    # Create heatmaps for each time period in the second row
    ax4 = fig.add_subplot(gs[1, 0])
    plot_heatmap(ax4, overall_df[['MSE', 'RMSE', 'MAE', 'WAPE', 'R²']].transpose(), "Overall Metrics Heatmap")

    ax5 = fig.add_subplot(gs[1, 1])
    plot_heatmap(ax5, daytime_df.transpose(), "Daytime Metrics Heatmap")

    ax6 = fig.add_subplot(gs[1, 2])
    plot_heatmap(ax6, nighttime_df.transpose(), "Nighttime Metrics Heatmap")

    # Set overall title with more space
    plt.suptitle(f'Model Performance Comparison - {dataset_name}', fontsize=18, y=0.98)

    # Adjust figure layout
    fig.subplots_adjust(top=0.90, bottom=0.05)

    # Save the figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/model_comparison_daytime_nighttime_{dataset_name}_{timestamp}.png')

    return fig


def plot_predictions_over_time(models, model_names, data_loader, target_scaler, num_samples=200, start_idx=0):
    """
    Plot time series predictions for multiple models with nighttime shading if available

    Args:
        models: List of PyTorch models
        model_names: List of model names
        data_loader: Data loader
        target_scaler: Scaler for the target variable
        num_samples: Number of consecutive time steps to plot
        start_idx: Starting index in the dataset
    """
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Collect data samples
    all_batches = []
    for batch in data_loader:
        all_batches.append(batch)
        if len(all_batches) * batch['target'].shape[0] > start_idx + num_samples:
            break

    # Combine batches into a single dataset
    all_temporal = []
    all_static = []
    all_targets = []
    all_nighttime = []
    all_time_index_local = []
    has_nighttime = False
    has_time_index_local = False

    for batch in all_batches:
        all_temporal.append(batch['temporal_features'])
        all_static.append(batch['static_features'])
        all_targets.append(batch['target'])
        # Check if nighttime data is available
        if 'nighttime_mask' in batch:
            has_nighttime = True
            all_nighttime.append(batch['nighttime_mask'])
        # Check if time_index_local is available
        if 'time_index_local' in batch:
            has_time_index_local = True
            # Store the time index values as they are
            if isinstance(batch['time_index_local'], list):
                all_time_index_local.extend(batch['time_index_local'])
            else:
                all_time_index_local.append(batch['time_index_local'])

    all_temporal = torch.cat(all_temporal, dim=0)
    all_static = torch.cat(all_static, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    if has_nighttime:
        all_nighttime = torch.cat(all_nighttime, dim=0)

    # Get the subset for visualization
    temporal = all_temporal[start_idx:start_idx+num_samples].to(device)
    static = all_static[start_idx:start_idx+num_samples].to(device)
    targets = all_targets[start_idx:start_idx+num_samples].cpu().numpy()

    if has_nighttime:
        nighttime = all_nighttime[start_idx:start_idx+num_samples].cpu().numpy()
        # Ensure nighttime is a 1D array
        if len(nighttime.shape) > 1:
            nighttime = nighttime.flatten() if nighttime.shape[1] == 1 else nighttime[:,0]

    # Get time index for x-axis if available
    x_values = None
    if has_time_index_local and len(all_time_index_local) >= start_idx + num_samples:
        # Extract the time values for the plotting window
        x_values = all_time_index_local[start_idx:start_idx+num_samples]

        # Try to convert to datetime objects if they are strings
        if isinstance(x_values[0], str):
            try:
                # Try different datetime formats
                date_formats = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%dT%H:%M:%S', '%Y%m%d%H%M%S']
                for date_format in date_formats:
                    try:
                        x_values = [datetime.strptime(t, date_format) for t in x_values]
                        print(f"Successfully parsed dates with format: {date_format}")
                        break
                    except ValueError:
                        continue

                # If we couldn't parse with any format, notify and use indices
                if isinstance(x_values[0], str):
                    print(f"Could not parse date format: {x_values[0]}, using indices instead")
                    x_values = None

            except (ValueError, TypeError) as e:
                # If conversion fails, fall back to using indices
                print(f"Error converting time_index_local to datetime: {e}, using indices instead")
                x_values = None

    # Generate predictions
    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            outputs = model(temporal, static).cpu().numpy()
            predictions.append(outputs)

    # Inverse transform to original scale
    y_true_orig = target_scaler.inverse_transform(targets.reshape(-1, 1)).flatten()
    y_pred_orig_list = [target_scaler.inverse_transform(pred.reshape(-1, 1)).flatten() for pred in predictions]

    # Create visualization
    fig = plt.figure(figsize=(15, 8))
    ax = plt.gca()

    # Define colors and line styles for predictions
    colors = ['blue', 'red', 'green', 'magenta', 'cyan', 'orange']
    line_styles = ['--', ':', '-.', '--', ':', '--']

    # Set x-axis values based on availability of time_index_local
    if x_values:
        # Plot actual values with time index
        actual_line, = plt.plot(x_values, y_true_orig, 'k-', label='Actual GHI', linewidth=2)

        # Plot predictions with time index
        pred_lines = []
        handles = [actual_line]
        labels = ['Actual GHI']

        for i, (pred, name) in enumerate(zip(y_pred_orig_list, model_names)):
            color = colors[i % len(colors)]
            style = line_styles[i % len(line_styles)]
            line, = plt.plot(x_values, pred, color=color, linestyle=style, label=f'{name} Predicted', alpha=0.7)
            pred_lines.append(line)
            handles.append(line)
            labels.append(f'{name} Predicted')

        # Format the x-axis to show dates properly
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=0)  # Make labels horizontal
        fig.subplots_adjust(bottom=0.15)  # Adjust bottom margin for horizontal labels

        # If we have nighttime data, shade those regions
        if has_nighttime:
            # Modify nighttime shading to work with datetime x-axis
            nighttime_bool = (nighttime > 0.5)
            night_regions = []
            start = None
            for i, is_night in enumerate(nighttime_bool):
                if is_night and start is None:
                    start = i
                elif not is_night and start is not None:
                    night_regions.append((start, i))
                    start = None
            if start is not None:
                night_regions.append((start, len(nighttime_bool)))

            for start, end in night_regions:
                if start < len(x_values) and end <= len(x_values):
                    ax.axvspan(x_values[start], x_values[min(end, len(x_values)-1)],
                              alpha=0.2, color='gray', label='_nolegend_')
    else:
        # Use default integer indices for x-axis
        actual_line, = plt.plot(y_true_orig, 'k-', label='Actual GHI', linewidth=2)

        # Plot predictions and collect handles/labels
        pred_lines = []
        handles = [actual_line]
        labels = ['Actual GHI']

        for i, (pred, name) in enumerate(zip(y_pred_orig_list, model_names)):
            color = colors[i % len(colors)]
            style = line_styles[i % len(line_styles)]
            line, = plt.plot(pred, color=color, linestyle=style, label=f'{name} Predicted', alpha=0.7)
            pred_lines.append(line)
            handles.append(line)
            labels.append(f'{name} Predicted')

        # If we have nighttime data, shade those regions
        if has_nighttime:
            nighttime_bool = (nighttime > 0.5)
            night_regions = []
            start = None
            for i, is_night in enumerate(nighttime_bool):
                if is_night and start is None:
                    start = i
                elif not is_night and start is not None:
                    night_regions.append((start, i))
                    start = None
            if start is not None:
                night_regions.append((start, len(nighttime_bool)))

            for start, end in night_regions:
                ax.axvspan(start, end, alpha=0.2, color='gray', label='_nolegend_')

    # Add nighttime legend if applicable
    if has_nighttime and len(night_regions) > 0:
        night_patch = Patch(facecolor='gray', alpha=0.2, label='Nighttime')
        handles.append(night_patch)
        labels.append('Nighttime')

    # Calculate and display error metrics for the visualization window
    for i, (pred, name) in enumerate(zip(y_pred_orig_list, model_names)):
        rmse = np.sqrt(np.mean((y_true_orig - pred) ** 2))
        mae = np.mean(np.abs(y_true_orig - pred))
        plt.annotate(f"{name}: RMSE={rmse:.2f}, MAE={mae:.2f}",
                     xy=(0.02, 0.97 - 0.03*i),
                     xycoords='axes fraction',
                     fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    plt.title('GHI Predictions Over Time')
    plt.xlabel('Time' if x_values else 'Time Step')
    plt.ylabel('GHI (W/m²)')

    # Set the legend with the correct handles and labels
    plt.legend(handles, labels, loc='upper right')

    plt.grid(True)
    plt.tight_layout()
    # Save the figure
    os.makedirs('plots', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'plots/predictions_over_time_{timestamp}.png')
    return fig
