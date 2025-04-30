from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import os
from utils.data_persistence import save_scalers, load_scalers


def create_time_features(timestamps):
    """
    Create cyclical time features from timestamps

    Args:
        timestamps: Array of datetime objects

    Returns:
        time_features_dict: Dictionary of individual time features
    """
    hour_of_day = np.array([t.hour for t in timestamps])
    day_of_year = np.array([t.timetuple().tm_yday for t in timestamps])
    month = np.array([t.month for t in timestamps])
    day_of_week = np.array([t.weekday() for t in timestamps])

    # Create cyclical features using sine and cosine transformations
    hour_sin = np.sin(2 * np.pi * hour_of_day / 24)
    hour_cos = np.cos(2 * np.pi * hour_of_day / 24)
    day_sin = np.sin(2 * np.pi * day_of_year / 366)
    day_cos = np.cos(2 * np.pi * day_of_year / 366)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    dow_sin = np.sin(2 * np.pi * day_of_week / 7)
    dow_cos = np.cos(2 * np.pi * day_of_week / 7)

    # Return individual time features in a dictionary
    return {
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'day_sin': day_sin,
        'day_cos': day_cos,
        'month_sin': month_sin,
        'month_cos': month_cos,
        'dow_sin': dow_sin,
        'dow_cos': dow_cos
    }


def normalize_data(data, selected_features, target_variable, scalers=None, fit_scalers=True):
    """
    Normalize a single dataset using appropriate scaling for each feature

    This function can be called separately for train/val/test with the same scalers

    Args:
        data: Dictionary of data to normalize
        selected_features: List of features to normalize
        target_variable: Target variable to predict
        scalers: Optional dictionary of pre-fitted scalers to use (for val/test sets)
        fit_scalers: Whether to fit new scalers (True for training, False for val/test)

    Returns:
        normalized_data: Dictionary of normalized data
        scalers: Dictionary of fitted scalers for reuse
    """
    # Initialize result dictionary and scalers if needed
    normalized_data = {}
    if scalers is None and fit_scalers:
        scalers = {}
    elif scalers is None and not fit_scalers:
        raise ValueError("Must provide scalers when fit_scalers=False")

    # Always include time_index if available (no normalization needed)
    if 'time_index' in data:
        normalized_data['time_index'] = data['time_index']

    # Process time features - these don't need scaling
    if 'timestamps' in data:
        # Convert timestamps to a storable format (ISO strings)
        # This ensures we don't get TypeError when saving to HDF5
        normalized_data['time_index_local'] = np.array([ts.isoformat() for ts in data['timestamps']], dtype='S')

        # Create time features (individual cyclical features)
        time_features_dict = create_time_features(data['timestamps'])
        # Add each time feature to the normalized data
        for key, value in time_features_dict.items():
            normalized_data[key] = value
    else:
        # If time features have already been processed, copy them individually
        time_feature_keys = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                             'month_sin', 'month_cos', 'dow_sin', 'dow_cos']
        for key in time_feature_keys:
            if key in data:
                normalized_data[key] = data[key]

    # Process coordinates
    if 'latitude' in data and 'longitude' in data:
        # We need to scale coordinates
        coords = np.column_stack([data['latitude'], data['longitude']])

        if fit_scalers:
            # Create and fit scaler for coordinates
            coord_scaler = MinMaxScaler()
            coord_scaler.fit(coords)
            scalers['coord_scaler'] = coord_scaler
        else:
            # Use existing scaler
            coord_scaler = scalers['coord_scaler']

        # Transform coordinates
        normalized_data['coordinates'] = coord_scaler.transform(coords)

    elif 'coordinates' in data:
        # If coordinates are already present, apply scaling
        if fit_scalers:
            coord_scaler = MinMaxScaler()
            coord_scaler.fit(data['coordinates'])
            scalers['coord_scaler'] = coord_scaler
        else:
            coord_scaler = scalers['coord_scaler']

        normalized_data['coordinates'] = coord_scaler.transform(data['coordinates'])

    # Process elevation
    if 'elevation' in data:
        if fit_scalers:
            elev_scaler = StandardScaler()
            elev_data = data['elevation'].reshape(-1, 1)
            elev_scaler.fit(elev_data)
            scalers['elev_scaler'] = elev_scaler
        else:
            elev_scaler = scalers['elev_scaler']

        # Transform elevation and reshape back to original shape
        elev_data = data['elevation'].reshape(-1, 1)
        normalized_data['elevation'] = elev_scaler.transform(elev_data).reshape(data['elevation'].shape)

    # Copy nighttime mask without normalization
    if 'nighttime_mask' in data:
        normalized_data['nighttime_mask'] = data['nighttime_mask']

    # Process all selected features and target variable
    all_features = list(selected_features)
    if target_variable not in all_features:
        all_features.append(target_variable)

    for feature in all_features:
        if feature not in data:
            print(f"Warning: Feature {feature} not found in data")
            continue

        # Create or use existing scaler
        if fit_scalers:
            feature_scaler = MinMaxScaler()
            reshaped_data = data[feature].reshape(-1, 1)
            feature_scaler.fit(reshaped_data)
            scalers[f'{feature}_scaler'] = feature_scaler
        else:
            feature_scaler = scalers[f'{feature}_scaler']

        # Transform and reshape data
        normalized_data[feature] = feature_scaler.transform(
            data[feature].reshape(-1, 1)
        ).reshape(data[feature].shape)

    return normalized_data, scalers


def create_sequences(data, lookback=24, selected_features=None, target_variable=None):
    """
    Create time series sequences for each location

    Args:
        data: Dictionary of normalized data
        lookback: Number of timesteps to look back
        selected_features: List of feature names to use
        target_variable: Target variable to predict

    Returns:
        seq_data: Dictionary with time series data in original structure
        seq_targets: Target GHI values
    """
    # Determine number of locations
    if 'coordinates' in data:
        n_locations = data['coordinates'].shape[0]
    else:
        n_locations = data[selected_features[0]].shape[1]

    # Determine number of timesteps
    n_timesteps = data[selected_features[0]].shape[0]

    # Determine effective number of timesteps after sequence creation
    effective_timesteps = n_timesteps - lookback

    # Create targets array with original shape (keeping the time dimension)
    seq_targets = data[target_variable][lookback:, :]

    # Initialize seq_data with original data structure
    seq_data = {}

    # Preserve time_index if available (shifted by lookback)
    if 'time_index' in data:
        seq_data['time_index'] = data['time_index'][lookback:]

    # Copy time features individually (shifted by lookback)
    time_feature_keys = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                         'month_sin', 'month_cos', 'dow_sin', 'dow_cos']
    for key in time_feature_keys:
        if key in data:
            seq_data[key] = data[key][lookback:]

    # Copy coordinates directly (shape will be n_locations, 2)
    if 'coordinates' in data:
        seq_data['coordinates'] = data['coordinates']

    # Copy elevation directly (shape will be n_locations)
    if 'elevation' in data:
        seq_data['elevation'] = data['elevation']

    # Copy all time series features with shape (time, locations)
    for feature in selected_features:
        if feature in data and feature not in ['coordinates', 'elevation'] and feature not in time_feature_keys:
            # Keep the original structure but shift by lookback
            seq_data[feature] = data[feature][lookback:, :]

    # Copy nighttime_mask and clearsky_ghi if present
    if 'nighttime_mask' in data:
        seq_data['nighttime_mask'] = data['nighttime_mask'][lookback:, :]

    if 'clearsky_ghi' in data:
        seq_data['clearsky_ghi'] = data['clearsky_ghi'][lookback:, :]

    # Instead of creating 3D sequences, we keep the original 2D structure
    # This works better with the TimeSeriesDataset which handles the reshaping

    print(f"Created sequences with shapes:")
    for key, value in seq_data.items():
        if isinstance(value, np.ndarray):
            print(f"  - {key}: {value.shape}")
    print(f"  - targets: {seq_targets.shape}")

    return seq_data, seq_targets


def apply_scalers(data, scalers, selected_features=None, target_variable=None, inverse=False):
    """
    Apply scalers to transform or inverse transform features.

    Args:
        data: Dictionary of data to transform or raw numpy arrays
        scalers: Dictionary of scalers
        selected_features: List of features to transform (if None, use all in scalers)
        target_variable: Target variable name
        inverse: If True, inverse transform data; if False, transform data

    Returns:
        transformed_data: Dictionary of transformed data
    """
    transformed_data = {}

    # If selected_features is None, extract feature names from scalers
    if selected_features is None:
        selected_features = []
        for key in scalers.keys():
            if key.endswith('_scaler') and not key.startswith('coord') and not key.startswith('elev'):
                feature = key[:-7]  # Remove '_scaler' suffix
                if feature != target_variable:
                    selected_features.append(feature)

    # Include target variable if provided
    all_features = list(selected_features)
    if target_variable is not None and target_variable not in all_features:
        all_features.append(target_variable)

    # Preserve time_index without transformation
    if 'time_index' in data:
        transformed_data['time_index'] = data['time_index']

    # Handle timestamps if available
    if 'timestamps' in data:
        # Convert to storable format (ISO strings)
        transformed_data['timestamps'] = np.array([ts.isoformat() for ts in data['timestamps']], dtype='S')

    # Process coordinates if available
    if 'coordinates' in data and 'coord_scaler' in scalers:
        if inverse:
            transformed_data['coordinates'] = scalers['coord_scaler'].inverse_transform(data['coordinates'])
        else:
            transformed_data['coordinates'] = scalers['coord_scaler'].transform(data['coordinates'])

    # Process elevation if available
    if 'elevation' in data and 'elev_scaler' in scalers:
        elev_data = data['elevation'].reshape(-1, 1)
        if inverse:
            transformed_data['elevation'] = scalers['elev_scaler'].inverse_transform(elev_data).reshape(data['elevation'].shape)
        else:
            transformed_data['elevation'] = scalers['elev_scaler'].transform(elev_data).reshape(data['elevation'].shape)

    # Process time features (pass through without scaling)
    if 'time_features' in data:
        transformed_data['time_features'] = data['time_features']

    # Process nighttime mask (pass through without scaling)
    if 'nighttime_mask' in data:
        transformed_data['nighttime_mask'] = data['nighttime_mask']

    # Process all features
    for feature in all_features:
        if feature in data and f'{feature}_scaler' in scalers:
            scaler = scalers[f'{feature}_scaler']
            # Reshape data to 2D for scaler
            original_shape = data[feature].shape
            reshaped_data = data[feature].reshape(-1, 1)

            # Transform or inverse transform
            if inverse:
                transformed = scaler.inverse_transform(reshaped_data)
            else:
                transformed = scaler.transform(reshaped_data)

            # Reshape back to original shape
            transformed_data[feature] = transformed.reshape(original_shape)

    return transformed_data
