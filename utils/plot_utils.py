"""
Plotting utilities for GHI forecasting visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates


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


def plot_predictions(metrics, model_name=''):
    """
    Plot model predictions and evaluation metrics

    Args:
        metrics: Dictionary of evaluation metrics
        model_name: Name of the model for the plot title
    """
    y_true = metrics['y_true']
    y_pred = metrics['y_pred']
    nighttime = metrics['nighttime'].flatten() > 0.5

    # Calculate residuals
    residuals = y_true - y_pred

    # Sample a subset for visualization
    sample_size = min(1000, len(metrics['y_true']))
    sample_indices = np.random.choice(len(metrics['y_true']), sample_size, replace=False)

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.scatter(y_true[sample_indices], y_pred[sample_indices], alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.title(f'{model_name} - Actual vs Predicted GHI')
    plt.xlabel('Actual GHI (W/m²)')
    plt.ylabel('Predicted GHI (W/m²)')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.hist(residuals, bins=50, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title(f'{model_name} - Residuals Distribution')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.scatter(y_pred[sample_indices], residuals[sample_indices], alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'{model_name} - Residuals vs Predicted')
    plt.xlabel('Predicted GHI (W/m²)')
    plt.ylabel('Residual')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.axis('off')
    stats_text = f"Metrics:\nMSE: {metrics['mse']:.2f}\nRMSE: {metrics['rmse']:.2f}\nMAE: {metrics['mae']:.2f}\nR²: {metrics['r2']:.4f}\n\n"
    stats_text += f"Residual Stats:\nMean: {np.mean(residuals):.2f}\nStd: {np.std(residuals):.2f}"
    plt.text(0.1, 0.5, stats_text, fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_predictions_over_time(models, model_names, data_loader, target_scaler, num_samples=200, start_idx=0):
    """
    Plot time series predictions for multiple models

    Args:
        models: List of PyTorch models
        model_names: List of model names
        data_loader: Data loader
        target_scaler: Scaler for the target variable
        num_samples: Number of consecutive time steps to plot
        start_idx: Starting index in the dataset
    """
    import torch

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

    for batch in all_batches:
        all_temporal.append(batch['temporal_features'])
        all_static.append(batch['static_features'])
        all_targets.append(batch['target'])
        all_nighttime.append(batch['nighttime'])

    all_temporal = torch.cat(all_temporal, dim=0)
    all_static = torch.cat(all_static, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_nighttime = torch.cat(all_nighttime, dim=0)

    # Get the subset for visualization
    temporal = all_temporal[start_idx:start_idx+num_samples].to(models[0].device)
    static = all_static[start_idx:start_idx+num_samples].to(models[0].device)
    targets = all_targets[start_idx:start_idx+num_samples].cpu().numpy()
    nighttime = all_nighttime[start_idx:start_idx+num_samples].cpu().numpy()

    # Generate predictions
    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            outputs = model(temporal, static).cpu().numpy()
            predictions.append(outputs)

    # Inverse transform to original scale
    y_true_orig = target_scaler.inverse_transform(targets)
    y_pred_orig_list = [target_scaler.inverse_transform(pred) for pred in predictions]

    # Create visualization
    plt.figure(figsize=(15, 8))

    # Plot nighttime regions
    night_regions = nighttime.flatten() > 0.5
    for i in range(len(night_regions)-1):
        if night_regions[i]:
            plt.axvspan(i, i+1, color='gray', alpha=0.2)

    # Plot predictions
    plt.plot(y_true_orig, 'k-', label='Actual GHI', linewidth=2)

    colors = ['b-', 'r-', 'g-', 'm-']
    for i, (pred, name) in enumerate(zip(y_pred_orig_list, model_names)):
        plt.plot(pred, colors[i], label=f'{name} Predicted', alpha=0.7)

    plt.title('GHI Predictions Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('GHI (W/m²)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_solar_day_night(data, location_idx=0, n_steps=None, start_idx=0, show_threshold=True):
    """
    Plot solar zenith angle and nighttime mask to verify day/night detection

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

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot solar zenith angle
    ax1.plot(selected_timestamps, solar_zenith, 'b-', label='Solar Zenith Angle')
    ax1.set_ylabel('Solar Zenith Angle (degrees)')
    ax1.set_title(f'Solar Zenith Angle and Nighttime Mask for Location {location_idx}')
    ax1.grid(True)

    # Add horizontal line at 90 degrees if requested
    if show_threshold:
        ax1.axhline(y=90, color='r', linestyle='--', label='90° Threshold')

    # Set reasonable y-limits based on the data
    if np.max(solar_zenith) > 0:
        y_max = min(180, np.max(solar_zenith) * 1.1)
        ax1.set_ylim(0, y_max)

    ax1.legend()

    # Plot nighttime mask
    ax2.plot(selected_timestamps, nighttime, 'g-', label='Nighttime Mask')
    ax2.set_ylabel('Nighttime Mask')
    ax2.set_xlabel('Time')
    ax2.set_ylim(-0.1, 1.1)  # Add some padding to better see the binary values
    ax2.grid(True)
    ax2.legend()

    # Add shaded regions to indicate nighttime (SZA >= 90°)
    for i in range(len(solar_zenith)):
        if solar_zenith[i] >= 90 or nighttime[i] > 0.5:
            # Add vertical span in both subplots
            ax1.axvspan(selected_timestamps[i],
                        selected_timestamps[min(i+1, len(selected_timestamps)-1)],
                        alpha=0.2, color='gray')
            ax2.axvspan(selected_timestamps[i],
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


def plot_time_features(timestamps, time_features, n_days=7):
    """
    Visualize time features created by create_time_features

    Args:
        timestamps: Array of datetime objects
        time_features: Optional pre-computed time features. If None, will be generated using create_time_features
        n_days: Number of days to plot (default: 7)

    Returns:
        fig: Matplotlib figure object
    """
    # Limit to n_days
    if len(timestamps) > 24 * n_days:
        timestamps = timestamps[:24 * n_days]
        if time_features is not None:
            time_features = time_features[:24 * n_days]

    # Extract components
    hour_sin, hour_cos = time_features[:, 0], time_features[:, 1]
    day_sin, day_cos = time_features[:, 2], time_features[:, 3]
    month_sin, month_cos = time_features[:, 4], time_features[:, 5]
    dow_sin, dow_cos = time_features[:, 6], time_features[:, 7]

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
