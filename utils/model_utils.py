"""
Model utilities for GHI forecasting.
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import wandb
from datetime import datetime, timedelta

def create_sequences(data, target, seq_length, forecast_horizon, step=1):
    """
    Create X, y sequences for time series forecasting

    Args:
        data: Input features array of shape (time_steps, n_features)
        target: Target array of shape (time_steps, 1)
        seq_length: Length of input sequence (lookback window)
        forecast_horizon: Number of steps to forecast ahead
        step: Step size for sliding window

    Returns:
        X: Input sequences of shape (n_samples, seq_length, n_features)
        y: Target values of shape (n_samples, forecast_horizon)
    """
    X, y = [], []
    for i in range(0, len(data) - seq_length - forecast_horizon + 1, step):
        X.append(data[i:i + seq_length])
        y.append(target[i + seq_length:i + seq_length + forecast_horizon])

    return np.array(X), np.array(y)

def create_multisite_sequences(data_dict, feature_list, target_variable, seq_length, forecast_horizon, step=1):
    """
    Create sequences for multiple locations, handling each site correctly

    Args:
        data_dict: Dictionary containing the dataset
        feature_list: List of feature names to use
        target_variable: Name of the target variable
        seq_length: Length of input sequence (lookback window)
        forecast_horizon: Number of steps to forecast ahead
        step: Step size for sliding window

    Returns:
        X: Input sequences
        y: Target values
        location_indices: Array indicating which location each sequence belongs to
    """
    all_X = []
    all_y = []
    all_locations = []

    # Get the total number of locations and prepare the feature and target arrays
    n_locations = data_dict['latitude'].shape[0]

    # First create combined feature array with properly scaled values
    feature_arrays = []
    for feature in feature_list:
        if feature in data_dict:
            # Scale the feature properly
            scaled_feature = scale_feature(data_dict[feature], feature)
            # Reshape to (time, locations, 1) for concatenation
            reshaped = scaled_feature.reshape(scaled_feature.shape[0], scaled_feature.shape[1], 1)
            feature_arrays.append(reshaped)

    # Create combined feature array of shape (time, locations, n_features)
    combined_features = np.concatenate(feature_arrays, axis=2)

    # Scale the target variable
    if target_variable in data_dict:
        target = scale_feature(data_dict[target_variable], target_variable)
    else:
        raise ValueError(f"Target variable {target_variable} not found in data")

    # Add solar zenith angle information if available (preferred over nighttime mask)
    if 'solar_zenith_angle' in data_dict and 'solar_zenith_angle' not in feature_list:
        solar_zenith = scale_feature(data_dict['solar_zenith_angle'], 'solar_zenith_angle')
        solar_zenith = solar_zenith.reshape(solar_zenith.shape[0], solar_zenith.shape[1], 1)
        combined_features = np.concatenate([combined_features, solar_zenith], axis=2)
    # Fall back to nighttime mask if solar zenith angle is not available
    elif 'nighttime_mask' in data_dict:
        nighttime_mask = data_dict['nighttime_mask'].reshape(data_dict['nighttime_mask'].shape[0],
                                                            data_dict['nighttime_mask'].shape[1], 1)
        combined_features = np.concatenate([combined_features, nighttime_mask], axis=2)

    # Create sequences for each location separately
    for loc_idx in range(n_locations):
        # Extract features and target for this location
        loc_features = combined_features[:, loc_idx, :]
        loc_target = target[:, loc_idx:loc_idx+1]  # Keep dimension for concatenation

        # Create sequences
        X, y = create_sequences(loc_features, loc_target, seq_length, forecast_horizon, step)

        # Only add if we have some sequences
        if len(X) > 0:
            all_X.append(X)
            all_y.append(y)
            all_locations.append(np.full(len(X), loc_idx))

    # Combine all locations
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    location_indices = np.concatenate(all_locations, axis=0)

    return X, y, location_indices

def normalize_data(train_data, val_data, test_data, selected_features):
    """
    Normalize data using appropriate scaling for each feature

    Args:
        train_data: Training data dictionary
        val_data: Validation data dictionary
        test_data: Test data dictionary
        selected_features: List of features to normalize

    Returns:
        norm_train_data: Normalized training data
        norm_val_data: Normalized validation data
        norm_test_data: Normalized test data
        scalers: Dictionary of fitted scalers
    """
    scalers = {}

    # Initialize normalized data dictionaries
    norm_train_data = {}
    norm_val_data = {}
    norm_test_data = {}

    # Add time features
    norm_train_data['time_features'] = train_time_features
    norm_val_data['time_features'] = val_time_features
    norm_test_data['time_features'] = test_time_features

    # Location data - normalize latitude and longitude
    coord_scaler = MinMaxScaler()
    train_coords = np.column_stack([train_data['latitude'], train_data['longitude']])
    coord_scaler.fit(train_coords)

    norm_train_data['coordinates'] = coord_scaler.transform(train_coords)
    val_coords = np.column_stack([val_data['latitude'], val_data['longitude']])
    norm_val_data['coordinates'] = coord_scaler.transform(val_coords)
    test_coords = np.column_stack([test_data['latitude'], test_data['longitude']])
    norm_test_data['coordinates'] = coord_scaler.transform(test_coords)

    scalers['coord_scaler'] = coord_scaler

    # Process elevation as a separate feature
    elev_scaler = StandardScaler()
    train_elev = train_data['elevation'].reshape(-1, 1)
    elev_scaler.fit(train_elev)

    norm_train_data['elevation'] = elev_scaler.transform(train_elev).reshape(train_data['elevation'].shape)
    norm_val_data['elevation'] = elev_scaler.transform(val_data['elevation'].reshape(-1, 1)).reshape(val_data['elevation'].shape)
    norm_test_data['elevation'] = elev_scaler.transform(test_data['elevation'].reshape(-1, 1)).reshape(test_data['elevation'].shape)

    scalers['elev_scaler'] = elev_scaler

    # Nighttime mask doesn't need normalization
    norm_train_data['nighttime_mask'] = train_data['nighttime_mask']
    norm_val_data['nighttime_mask'] = val_data['nighttime_mask']
    norm_test_data['nighttime_mask'] = test_data['nighttime_mask']

    # Process all selected features
    for feature in selected_features + [TARGET_VARIABLE, 'clear_sky_ghi']:
        if feature not in train_data:
            print(f"Warning: Feature {feature} not found in training data")
            continue

        # Apply domain-specific scaling
        train_feature_scaled = scale_features(train_data[feature], feature)
        val_feature_scaled = scale_features(val_data[feature], feature)
        test_feature_scaled = scale_features(test_data[feature], feature)

        # Create and fit scaler for the feature
        feature_scaler = MinMaxScaler()
        reshaped_data = train_feature_scaled.reshape(-1, 1)
        feature_scaler.fit(reshaped_data)

        # Transform and reshape data
        norm_train_data[feature] = feature_scaler.transform(train_feature_scaled.reshape(-1, 1)).reshape(train_data[feature].shape)
        norm_val_data[feature] = feature_scaler.transform(val_feature_scaled.reshape(-1, 1)).reshape(val_data[feature].shape)
        norm_test_data[feature] = feature_scaler.transform(test_feature_scaled.reshape(-1, 1)).reshape(test_data[feature].shape)

        # Store scaler
        scalers[f'{feature}_scaler'] = feature_scaler

    return norm_train_data, norm_val_data, norm_test_data, scalers

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
