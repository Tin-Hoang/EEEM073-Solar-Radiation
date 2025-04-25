from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


def create_time_features(timestamps):
    """
    Create cyclical time features from timestamps

    Args:
        timestamps: Array of datetime objects

    Returns:
        time_features: Array of time features
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

    return np.column_stack([
        hour_sin, hour_cos,
        day_sin, day_cos,
        month_sin, month_cos,
        dow_sin, dow_cos
    ])


def normalize_data(train_data, val_data, test_data, selected_features, target_variable):
    """
    Normalize data using appropriate scaling for each feature

    Args:
        train_data: Training data dictionary
        val_data: Validation data dictionary
        test_data: Test data dictionary
        selected_features: List of features to normalize
        target_variable: Target variable to predict

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
    norm_train_data['time_features'] = create_time_features(train_data['timestamps'])
    norm_val_data['time_features'] = create_time_features(val_data['timestamps'])
    norm_test_data['time_features'] = create_time_features(test_data['timestamps'])

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
    for feature in selected_features + [target_variable]:
        if feature not in train_data:
            print(f"Warning: Feature {feature} not found in training data")
            continue

        # Create and fit scaler for the feature (data is already scaled in data_loading_utils)
        feature_scaler = MinMaxScaler()
        reshaped_data = train_data[feature].reshape(-1, 1)
        feature_scaler.fit(reshaped_data)

        # Transform and reshape data
        norm_train_data[feature] = feature_scaler.transform(train_data[feature].reshape(-1, 1)).reshape(train_data[feature].shape)
        norm_val_data[feature] = feature_scaler.transform(val_data[feature].reshape(-1, 1)).reshape(val_data[feature].shape)
        norm_test_data[feature] = feature_scaler.transform(test_data[feature].reshape(-1, 1)).reshape(test_data[feature].shape)

        # Store scaler
        scalers[f'{feature}_scaler'] = feature_scaler

    return norm_train_data, norm_val_data, norm_test_data, scalers


def create_sequences(data, lookback=24, selected_features=None, target_variable=None):
    """
    Create time series sequences for each location

    Args:
        data: Dictionary of normalized data
        lookback: Number of timesteps to look back

    Returns:
        seq_data: Dictionary of sequence arrays
        seq_targets: Target GHI values
        seq_metadata: Dictionary of metadata for each sequence
    """
    # Initialize lists to store sequence data
    seq_features = []
    seq_coords = []
    seq_time = []
    seq_targets = []
    seq_nighttime = []
    seq_clear_sky = []
    seq_elevation = []

    # Determine number of locations
    if 'coordinates' in data:
        n_locations = data['coordinates'].shape[0]
    else:
        n_locations = data[selected_features[0]].shape[1]

    # For each location, create sequences
    for loc in range(n_locations):
        # Prepare feature array - combine all selected features
        feature_arrays = []
        for feature in selected_features:
            if feature in data:
                feature_arrays.append(data[feature][:, loc].reshape(-1, 1))

        # Stack features along the feature dimension
        if feature_arrays:
            features = np.hstack(feature_arrays)

            # Extract target and metadata
            ghi = data[target_variable][:, loc]
            coord = data['coordinates'][loc]
            time_feat = data['time_features']
            nighttime_mask = data['nighttime_mask'][:, loc]
            clear_sky = data['clearsky_ghi'][:, loc]
            elevation = data['elevation'][loc]

            # Create sequences
            for i in range(len(features) - lookback):
                seq_features.append(features[i:i+lookback])
                seq_time.append(time_feat[i:i+lookback])
                seq_coords.append(np.tile(coord, (lookback, 1)))
                seq_elevation.append(np.tile(elevation, lookback))
                seq_targets.append(ghi[i+lookback])
                seq_nighttime.append(nighttime_mask[i+lookback])
                seq_clear_sky.append(clear_sky[i+lookback])

    # Convert lists to arrays
    seq_data = {
        'features': np.array(seq_features),
        'time': np.array(seq_time),
        'coordinates': np.array(seq_coords),
        'elevation': np.array(seq_elevation),
        'nighttime_mask': np.array(seq_nighttime),
        'clearsky_ghi': np.array(seq_clear_sky)
    }

    return seq_data, np.array(seq_targets)
