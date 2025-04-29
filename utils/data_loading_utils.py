"""
Data loading utilities for GHI forecasting.
"""

import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from datetime import datetime, timedelta


def load_sites_sample(h5_file, n_sites=1000, random_state=42):
    """
    Sample a subset of sites from the H5 file

    Args:
        h5_file: Path to H5 file
        n_sites: Number of sites to sample (None to use all sites)
        random_state: Random seed for reproducibility

    Returns:
        sampled_indices: Array of sampled site indices
    """
    with h5py.File(h5_file, 'r') as f:
        total_sites = f['meta'].shape[0]

        if n_sites is None:
            # Use all available sites
            return np.arange(total_sites)
        else:
            # Sample a subset of sites
            np.random.seed(random_state)
            sampled_indices = np.random.choice(total_sites, min(n_sites, total_sites), replace=False)
            sampled_indices = np.sort(sampled_indices)
            return sampled_indices

def convert_to_local_time(utc_timestamps, timezone_offsets):
    """
    Convert UTC timestamps to local time for each site

    Args:
        utc_timestamps: Array of UTC timestamps
        timezone_offsets: Timezone offset in hours for each site

    Returns:
        local_timestamps: Dictionary of site_idx -> array of local timestamps
    """
    local_timestamps = {}

    for site_idx, tz_offset in enumerate(timezone_offsets):
        # Convert hours to timedelta
        offset = timedelta(hours=int(tz_offset))
        # Apply offset to each timestamp
        local_ts = [ts + offset for ts in utc_timestamps]
        local_timestamps[site_idx] = local_ts

    return local_timestamps


def load_data_chunk(h5_file, site_indices, features, target_variable="ghi", apply_scaling=True):
    """
    Load data for specific sites and features from H5 file
    with proper handling of time zones

    Args:
        h5_file: Path to H5 file
        site_indices: Array of site indices to load
        features: List of feature names to load
        target_variable: Name of the target variable
        apply_scaling: Whether to apply feature scaling (default: True)

    Returns:
        data_dict: Dictionary containing the loaded data
    """
    data_dict = {}

    with h5py.File(h5_file, 'r') as f:
        # Load time index
        time_index = f['time_index'][:]

        # Parse timestamps (keeping them in UTC initially)
        time_strings = [t.decode('utf-8') for t in time_index]
        utc_timestamps = pd.to_datetime(time_strings)

        # Load metadata for selected sites
        meta = f['meta'][site_indices]
        lats = meta['latitude']
        lons = meta['longitude']
        elevations = meta['elevation']
        timezones = meta['timezone']

        # Convert UTC timestamps to local time for each site
        local_timestamps_dict = convert_to_local_time(utc_timestamps, timezones)

        # Store the raw UTC timestamps for reference
        data_dict['utc_timestamps'] = utc_timestamps

        # We'll use the first site's local time as reference for plotting
        # This is arbitrary but necessary for visualization
        data_dict['timestamps'] = local_timestamps_dict[0]

        # Load selected features
        for feature in features + [target_variable]:
            if feature in f:
                data_dict[feature] = f[feature][:, site_indices]
            else:
                print(f"[WARNING] Feature {feature} not found in {h5_file}")

        # Load metadata
        data_dict['time_index'] = time_index
        data_dict['latitude'] = lats
        data_dict['longitude'] = lons
        data_dict['elevation'] = elevations
        data_dict['timezone'] = timezones

        # Check if solar_zenith_angle already exists in the dataset
        if 'solar_zenith_angle' not in data_dict and 'solar_zenith_angle' in f:
            # Load solar zenith angle
            data_dict['solar_zenith_angle'] = f['solar_zenith_angle'][:, site_indices]
            print(f"  Loaded solar_zenith_angle with shape {data_dict['solar_zenith_angle'].shape} and dtype {data_dict['solar_zenith_angle'].dtype}")
            # Print some sample values to debug
            if len(data_dict['solar_zenith_angle']) > 0:
                print(f"  Sample solar_zenith_angle values: {data_dict['solar_zenith_angle'][0, 0]}, {data_dict['solar_zenith_angle'][12, 0]} (raw values)")

        # Load clearsky GHI if available
        if 'clearsky_ghi' in f:
            data_dict['clearsky_ghi'] = f['clearsky_ghi'][:, site_indices]

        # Apply feature scaling if requested - do this BEFORE computing physical constraints
        if apply_scaling:
            # Scale all features and the target variable
            for feature_name in list(data_dict.keys()):
                if feature_name in features + [target_variable, 'solar_zenith_angle', 'clearsky_ghi'] and \
                   isinstance(data_dict[feature_name], np.ndarray):
                    # Apply scaling
                    original_data = data_dict[feature_name].copy()
                    data_dict[feature_name] = scale_feature(data_dict[feature_name], feature_name)
                    # Print info about the scaling for important features
                    if feature_name in ['solar_zenith_angle', 'clearsky_ghi', target_variable]:
                        print(f"  Scaled {feature_name}: range changed from "
                              f"[{np.min(original_data):.2f}, {np.max(original_data):.2f}] to "
                              f"[{np.min(data_dict[feature_name]):.2f}, {np.max(data_dict[feature_name]):.2f}]")

    return data_dict


def load_dataset(files, n_sites=None, features=None, target_variable="ghi", random_state=42, apply_scaling=True, time_interval=None):
    """
    Load data from H5 files with proper handling of multiple files

    Args:
        files: List of H5 file paths
        n_sites: Number of sites to sample per file (None to use all sites)
        features: List of features to load
        target_variable: Name of the target variable
        random_state: Random seed for reproducibility
        apply_scaling: Whether to apply feature scaling (default: True)
        time_interval: Time interval to resample data to (e.g., '1h' for hourly, '30min' for 30-minute)
                      None means use the original data timesteps

    Returns:
        data_dict: Dictionary containing the loaded data
    """
    if not files:
        raise ValueError("No files provided")

    if features is None:
        # Default features if none specified
        features = [
            'air_temperature',
            'wind_speed',
            'relative_humidity',
            'cloud_type'
        ]

    all_data = []

    for file_idx, h5_file in enumerate(files):
        print(f"Loading from {Path(h5_file).name}...")

        if n_sites is None:
            print(f"  Using all available sites in the file")
        else:
            print(f"  Sampling {n_sites} sites from the file")

        # Sample sites from each file
        site_indices = load_sites_sample(h5_file, n_sites, random_state + file_idx)
        print(f"  Selected {len(site_indices)} sites")

        # Pass the apply_scaling parameter to load_data_chunk
        data_chunk = load_data_chunk(h5_file, site_indices, features, target_variable, apply_scaling)
        all_data.append(data_chunk)

    # If only one file, return its data directly
    if len(all_data) == 1:
        combined_data = all_data[0]
    else:
        # For multiple files, we need to combine them properly
        # This assumes time periods or sites are distinct between files
        print("Combining data from multiple files...")

        # Determine if files contain different time periods or different sites
        first_timestamps = all_data[0]['timestamps']
        second_timestamps = all_data[1]['timestamps'] if len(all_data) > 1 else None

        # Check if timestamps are the same across files
        same_time_periods = False
        if second_timestamps is not None:
            if len(first_timestamps) == len(second_timestamps):
                # Compare a few timestamps to determine if they match
                sample_indices = [0, len(first_timestamps)//2, -1]
                matches = sum(first_timestamps[i] == second_timestamps[i] for i in sample_indices)
                same_time_periods = matches >= 2  # If most samples match, assume same time periods

        if same_time_periods:
            # Files contain different sites for the same time periods
            print("Files contain different sites for the same time periods")
            combined_data = combine_sites(all_data)
        else:
            # Files contain different time periods for the same sites
            # or entirely different datasets
            print("Files contain different time periods - concatenating by time")
            combined_data = combine_time_periods(all_data)

    # Resample time series data if a time interval is specified
    if time_interval is not None:
        print(f"Resampling data to {time_interval} intervals...")
        combined_data = resample_time_series(combined_data, time_interval)

    # Print the shapes of the data for diagnostic purposes
    print("Final data shapes:")
    for key in combined_data:
        if isinstance(combined_data[key], np.ndarray):
            print(f"  {key}: {combined_data[key].shape}")

    # Print time series info
    if 'timestamps' in combined_data:
        timestamps = combined_data['timestamps']
        if len(timestamps) > 1:
            time_diff = timestamps[1] - timestamps[0]
            print(f"  Time series interval: {time_diff}")
            steps_per_day = 24 * 60 * 60 / time_diff.total_seconds()
            print(f"  Approximately {steps_per_day:.1f} timesteps per day")

    return combined_data

def combine_sites(data_chunks):
    """
    Combine data from multiple chunks that contain different sites
    for the same time periods

    Args:
        data_chunks: List of data dictionaries to combine

    Returns:
        combined_data: Combined data dictionary
    """
    combined_data = {}
    first_chunk = data_chunks[0]

    # For time-related fields, just use the first chunk
    time_fields = ['time_index', 'timestamps', 'utc_timestamps']
    for field in time_fields:
        if field in first_chunk:
            combined_data[field] = first_chunk[field]

    # For feature arrays, concatenate along the site dimension (axis 1)
    for key in first_chunk:
        if key not in time_fields:
            if isinstance(first_chunk[key], np.ndarray):
                if len(first_chunk[key].shape) > 1:
                    # 2D arrays (time, sites) - concatenate along sites axis
                    arrays_to_concatenate = [chunk[key] for chunk in data_chunks if key in chunk]
                    combined_data[key] = np.concatenate(arrays_to_concatenate, axis=1)
                elif key in ['latitude', 'longitude', 'elevation', 'timezone']:
                    # 1D site metadata - concatenate normally
                    arrays_to_concatenate = [chunk[key] for chunk in data_chunks if key in chunk]
                    combined_data[key] = np.concatenate(arrays_to_concatenate)

    return combined_data

def combine_time_periods(data_chunks):
    """
    Combine data from multiple chunks that contain different time periods

    Args:
        data_chunks: List of data dictionaries to combine

    Returns:
        combined_data: Combined data dictionary
    """
    combined_data = {}
    first_chunk = data_chunks[0]

    # For site metadata, make sure they match across chunks
    # and just use the first chunk's values
    site_fields = ['latitude', 'longitude', 'elevation', 'timezone']
    for field in site_fields:
        if field in first_chunk:
            combined_data[field] = first_chunk[field]

    # For time and feature arrays, concatenate along the time dimension (axis 0)
    time_fields = ['time_index', 'timestamps', 'utc_timestamps']
    for key in first_chunk:
        if isinstance(first_chunk[key], np.ndarray) or key in time_fields:
            if key in time_fields:
                # Special handling for timestamp fields
                if key == 'timestamps' or key == 'utc_timestamps':
                    # Concatenate datetime arrays
                    arrays_to_concatenate = [chunk[key] for chunk in data_chunks if key in chunk]
                    combined_data[key] = np.concatenate(arrays_to_concatenate)
                elif key == 'time_index':
                    # Concatenate byte strings
                    arrays_to_concatenate = [chunk[key] for chunk in data_chunks if key in chunk]
                    combined_data[key] = np.concatenate(arrays_to_concatenate)
            elif len(first_chunk[key].shape) > 1:
                # 2D arrays (time, sites) - concatenate along time axis
                arrays_to_concatenate = [chunk[key] for chunk in data_chunks if key in chunk]
                combined_data[key] = np.concatenate(arrays_to_concatenate, axis=0)

    return combined_data

def scale_feature(feature_data, feature_name):
    """
    Apply appropriate scaling to different feature types

    Args:
        feature_data: Feature array to scale
        feature_name: Name of the feature

    Returns:
        scaled_data: Appropriately scaled data
    """
    # Make a copy to avoid modifying the original
    data = feature_data.copy()

    # Handle different data types and ranges
    if feature_name in ['air_temperature', 'dew_point']:
        # Convert from int16 (°C * 10) to float
        return data.astype(float) / 10.0
    elif feature_name in ['ghi', 'dni', 'dhi', 'clearsky_ghi', 'clearsky_dni', 'clearsky_dhi']:
        # Convert from uint16 (W/m²) to float
        return data.astype(float)
    elif feature_name in ['relative_humidity']:
        # Convert from uint16 (% * 100) to float
        return data.astype(float) / 100.0
    elif feature_name in ['wind_speed']:
        # Convert from uint16 (m/s * 10) to float
        return data.astype(float) / 10.0
    elif feature_name in ['surface_albedo']:
        # Convert from uint16 (mbar * 100) to float
        return data.astype(float) / 100.0
    elif feature_name in ['total_precipitable_water']:
        # Convert from uint8 (cm * 10) to float
        return data.astype(float) / 10.0
    elif feature_name in ['cloud_type', 'cloud_fill_flag', 'nighttime_mask']:
        # Categorical features and binary masks - leave as integers
        return data.astype(float)
    elif feature_name in ['ozone']:
        # Convert from uint16 (DU * 1000) to float
        return data.astype(float) / 1000.0
    elif feature_name in ['solar_zenith_angle', 'solar_azimuth_angle']:
        # Convert from int16 (degrees * 100) to float degrees
        # This is critical for correct nighttime_mask calculation
        return data.astype(float) / 100.0
    elif feature_name in ['aod']:
        # Convert from uint16 (DU * 1000) to float
        return data.astype(float) / 1000.0
    elif feature_name in ['cld_press_acha']:
        # Convert from uint16 (10) to float
        return data.astype(float) / 10.0
    elif feature_name in ['cld_opd_dcomp', 'cld_reff_dcomp']:
        # Convert from uint16 (100) to float
        return data.astype(float) / 100.0
    elif feature_name in ['dew_point']:
        # Convert from int16 (°C * 10) to float
        return data.astype(float) / 10.0
    else:
        # Default scaling for other features
        return data.astype(float)

def resample_time_series(data_dict, time_interval):
    """
    Resample time series data to a specified time interval

    Args:
        data_dict: Dictionary containing time series data
        time_interval: Time interval to resample to (e.g., '1h' for hourly, '30min' for 30-minute)

    Returns:
        resampled_data: Dictionary with resampled time series
    """
    if 'timestamps' not in data_dict:
        print("WARNING: No timestamps found in data, cannot resample")
        return data_dict

    import pandas as pd

    # Create a copy to avoid modifying the original
    resampled_data = {}

    # Get original timestamps
    orig_timestamps = data_dict['timestamps']
    print(f"  Resampling from {len(orig_timestamps)} original timesteps to '{time_interval}' frequency")

    # Create a DataFrame with timestamps as index for resampling
    resampler_df = pd.DataFrame(index=pd.DatetimeIndex(orig_timestamps))
    resampled_index = resampler_df.resample(time_interval).asfreq().index
    print(f"  New timestep count: {len(resampled_index)}")

    # Store new timestamps
    resampled_data['timestamps'] = pd.DatetimeIndex(resampled_index).to_pydatetime()

    # If UTC timestamps exist, resample those too
    if 'utc_timestamps' in data_dict:
        utc_timestamps = data_dict['utc_timestamps']
        utc_df = pd.DataFrame(index=pd.DatetimeIndex(utc_timestamps))
        utc_resampled = utc_df.resample(time_interval).asfreq().index
        resampled_data['utc_timestamps'] = pd.DatetimeIndex(utc_resampled).to_pydatetime()

    # Copy non-time series data directly
    non_time_series_keys = ['latitude', 'longitude', 'elevation', 'timezone', 'time_index']
    for key in non_time_series_keys:
        if key in data_dict:
            resampled_data[key] = data_dict[key]

    # Identify time series features (2D arrays with time as first dimension)
    time_series_keys = []
    categorical_keys = ['nighttime_mask', 'cloud_fill_flag', 'cloud_type']

    for key, value in data_dict.items():
        if isinstance(value, np.ndarray) and len(value.shape) > 1:
            if key not in non_time_series_keys and key not in ['timestamps', 'utc_timestamps', 'time_index']:
                time_series_keys.append(key)

    print(f"  Processing {len(time_series_keys)} time series variables")

    # Process time series features in groups to improve efficiency
    # Create a dummy dataframe with the original timestamps as index
    # This will be used for all features to avoid recreating for each one
    df_template = pd.DataFrame(index=orig_timestamps)

    # Process features in groups of similar type
    for is_categorical in [True, False]:
        current_keys = [k for k in time_series_keys if (k in categorical_keys) == is_categorical]
        if not current_keys:
            continue

        print(f"  Resampling {'categorical' if is_categorical else 'continuous'} features: {current_keys}")

        # Get example array to determine dimensions
        example_arr = data_dict[current_keys[0]]
        time_steps, n_sites = example_arr.shape
        new_time_steps = len(resampled_index)

        for key in current_keys:
            # Create a resampled array
            resampled_array = np.zeros((new_time_steps, n_sites), dtype=data_dict[key].dtype)

            # For large arrays, process in batches of sites to avoid memory issues
            batch_size = 200  # Number of sites to process at once
            for batch_start in range(0, n_sites, batch_size):
                batch_end = min(batch_start + batch_size, n_sites)
                print(f"    Processing {key}: sites {batch_start}-{batch_end} of {n_sites}")

                # Extract batch of sites
                site_batch = data_dict[key][:, batch_start:batch_end]
                batch_size_actual = site_batch.shape[1]

                # Create a DataFrame with original timestamps as index and sites as columns
                df = pd.DataFrame(
                    site_batch,
                    index=orig_timestamps,
                    columns=[f'site_{i}' for i in range(batch_size_actual)]
                )

                # Resample based on the type of data
                if is_categorical:
                    # For categorical or binary data, use mode (most common value)
                    resampled_df = df.resample(time_interval).apply(
                        lambda x: x.mode().iloc[0] if not x.empty else np.nan
                    )
                else:
                    # For continuous data, use mean (faster)
                    resampled_df = df.resample(time_interval).mean()

                # Fill any NaN values that might have been introduced
                resampled_df = resampled_df.fillna(method='ffill').fillna(method='bfill')

                # Store in the resampled array
                resampled_array[:, batch_start:batch_end] = resampled_df.values

            # Store the resampled array
            resampled_data[key] = resampled_array

    return resampled_data
