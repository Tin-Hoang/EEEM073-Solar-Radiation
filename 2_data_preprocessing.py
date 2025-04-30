# %% [markdown]
# # GHI Forecasting using PyTorch Models
#
# This notebook implements deep learning models to forecast Global Horizontal Irradiance (GHI) based on various input features. The notebook is structured as follows:
#
# 1. **Data Loading and Exploration**: Load the pre-split train/validation/test datasets and explore their structure
# 2. **Data Preprocessing**: Prepare the data for model training (normalization, feature engineering)

# %% [markdown]
# ## 1. Setup and Data Loading
#
# First, let's import the necessary libraries and load the data.

# %%
# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import os
import datetime
from pathlib import Path
from tqdm.notebook import tqdm

# PyTorch libraries
import torch

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")



# %%
# Define data loading parameters

# Find all data files
train_files = [
    "data/raw/himawari7_2011_Hồ-Chí-Minh.h5",
    # "data/raw/himawari7_2012_Hồ-Chí-Minh.h5",
    # "data/raw/himawari7_2013_Hồ-Chí-Minh.h5",
    # "data/raw/himawari7_2014_Hồ-Chí-Minh.h5",
    # "data/raw/himawari7_2015_Hồ-Chí-Minh.h5",
    # "data/raw/himawari7_2016_Hồ-Chí-Minh.h5",
    # "data/raw/himawari7_2017_Hồ-Chí-Minh.h5",
    # "data/raw/himawari7_2018_Hồ-Chí-Minh.h5",
]
val_files = [
    "data/raw/himawari7_2019_Hồ-Chí-Minh.h5",
]
test_files = [
    "data/raw/himawari7_2020_Hồ-Chí-Minh.h5",
]

print(f"Found {len(train_files)} training files: {[Path(f).name for f in train_files]}")
print(f"Found {len(val_files)} validation files: {[Path(f).name for f in val_files]}")
print(f"Found {len(test_files)} test files: {[Path(f).name for f in test_files]}")

# Check if the files exist
if not (train_files and val_files and test_files):
    print("Warning: Some data files are missing!")

# List of features to use
AVAILABLE_FEATURES = [
    'ghi',                     # Target variable
    'air_temperature',         # Weather features
    'wind_speed',
    'relative_humidity',
    'dew_point',
    'surface_pressure',
    'total_precipitable_water',
    'cloud_type',              # Cloud features
    'cloud_fill_flag',
    'cld_opd_dcomp',
    'cld_press_acha',
    'cld_reff_dcomp',
    'clearsky_ghi',            # Clear sky estimates
    'clearsky_dni',
    'clearsky_dhi',
    'solar_zenith_angle',      # Solar geometry
    'surface_albedo',          # Surface properties
    'ozone',                   # Atmospheric properties
    'aod',
    'ssa',
    'asymmetry',
    'alpha'
]

# Choose features which could be potentially useful for the modeling
# This is to reduce the number of features to be processed
# by excluding the features which are unrelevant to the target variable
# In the later modelling part, we will continue to select only subset of those features to input into the model
SELECTED_FEATURES = [
    'air_temperature',
    'wind_speed',
    'relative_humidity',
    'cloud_type',
    'cld_opd_dcomp',
    'cld_press_acha',
    'cld_reff_dcomp',
    'solar_zenith_angle',
    'clearsky_ghi',
    'total_precipitable_water',
    'surface_albedo',
    'aod'
]

# Target variable
TARGET_VARIABLE = 'ghi'


# %% [markdown]
# ### 1.1 Exploring the Data Structure
#
# Let's examine the structure of our H5 files.

# %%
def explore_h5_structure(h5_path):
    try:
        with h5py.File(h5_path, 'r') as f:
            print(f"\nExploring {h5_path}")
            print("\nDatasets:")
            for key in f.keys():
                dataset = f[key]
                print(f"  /{key} {dataset.shape}: {dataset.dtype}")
                if key == 'meta':
                    # Explore metadata structure
                    print("  Metadata fields:")
                    for field in dataset.dtype.names:
                        values = dataset[field]
                        unique_count = len(np.unique(values))
                        sample = np.unique(values)[:5]
                        print(f"    {field}: {unique_count} unique values, examples: {sample}")
                elif len(dataset) > 0 and dataset.ndim > 0:
                    # For numerical data, show some statistics
                    try:
                        if dataset.dtype in [np.float32, np.float64, np.int16, np.int32, np.uint8, np.uint16]:
                            # Take a small sample to calculate stats
                            sample = dataset[:10, :100] if dataset.ndim > 1 else dataset[:10]
                            stats = {
                                'min': np.min(sample),
                                'max': np.max(sample),
                                'mean': np.mean(sample),
                                'has_nan': np.isnan(sample).any()
                            }
                            print(f"    Sample stats: {stats}")
                    except Exception as e:
                        print(f"    Error computing stats: {e}")
    except Exception as e:
        print(f"Error exploring {h5_path}: {e}")

# Explore first file from each set
if train_files:
    explore_h5_structure(train_files[0])
if val_files:
    explore_h5_structure(val_files[0])
if test_files:
    explore_h5_structure(test_files[0])


# %% [markdown]
# ### 1.2 Raw Data Loading
#

# %%
from utils.data_loading_utils import load_dataset

# Load data with selected features
print("Loading training data...")
train_data = load_dataset(train_files, n_sites=None, features=SELECTED_FEATURES, target_variable="ghi", apply_scaling=True, time_interval="1h")
print("\nLoading validation data...")
val_data = load_dataset(val_files, n_sites=None, features=SELECTED_FEATURES, target_variable="ghi", apply_scaling=True, time_interval="1h")
print("\nLoading test data...")
test_data = load_dataset(test_files, n_sites=None, features=SELECTED_FEATURES, target_variable="ghi", apply_scaling=True, time_interval="1h")


# %% [markdown]
# ### 1.3 Visualize the Data

# %%
from utils.plot_utils import plot_time_series

# Plot data for a few locations
fig = plot_time_series(train_data,
                       features=SELECTED_FEATURES + ['nighttime_mask', "ghi"],
                       location_idx=0,
                       start_idx=0,
                       n_steps=48)


# %% [markdown]
# ## 2. Data Preprocessing

# %% [markdown]
# ### 2.1 Time Feature Engineering

# %%
from utils.normalize_utils import create_time_features

# Create time features
train_time_features_dict = create_time_features(train_data['timestamps'])
val_time_features_dict = create_time_features(val_data['timestamps'])
test_time_features_dict = create_time_features(test_data['timestamps'])

# Add each time feature to the respective data dictionaries
print("Time features:")
for key, value in train_time_features_dict.items():
    train_data[key] = value
    val_data[key] = val_time_features_dict[key]
    test_data[key] = test_time_features_dict[key]
    print(f"  - {key}: {value.shape}")


# %%
from utils.plot_utils import plot_time_features

_ = plot_time_features(train_data['timestamps'], train_time_features_dict, n_days=14)


# %% [markdown]
# ### 2.2 Create nighttime mask feature

# %%
from utils.features_utils import compute_nighttime_mask

# Compute nighttime mask using solar zenith angle if available
train_nighttime_mask = compute_nighttime_mask(
    train_data['timestamps'], train_data['latitude'], train_data['longitude'], train_data.get('solar_zenith_angle')
)
train_data['nighttime_mask'] = train_nighttime_mask
# Do the same for validation
val_nighttime_mask = compute_nighttime_mask(
    val_data['timestamps'], val_data['latitude'], val_data['longitude'], val_data.get('solar_zenith_angle')
)
val_data['nighttime_mask'] = val_nighttime_mask
# Do the same for test data
test_nighttime_mask = compute_nighttime_mask(
    test_data['timestamps'], test_data['latitude'], test_data['longitude'], test_data.get('solar_zenith_angle')
)
test_data['nighttime_mask'] = test_nighttime_mask
print(f"  Computed nighttime_mask with {np.sum(train_nighttime_mask)} night hours out of {train_nighttime_mask.size} total hours")

# Add nighttime mask to selected features if not already present
if "nighttime_mask" not in SELECTED_FEATURES:
    SELECTED_FEATURES += ["nighttime_mask"]


# %%
from utils.plot_utils import plot_solar_day_night

_ = plot_solar_day_night(train_data, location_idx=0, n_steps=48, start_idx=0)


# %% [markdown]
# ### 2.3 Create Clear Sky model (if not provided)

# %%
from utils.features_utils import compute_clearsky_ghi

# Use pre-computed clear sky GHI if available, otherwise compute it
if 'clearsky_ghi' not in train_data:
    clear_sky_ghi = compute_clearsky_ghi(
        train_data['timestamps'], train_data['latitude'], train_data['longitude'], train_data.get('solar_zenith_angle')
    )
    train_data['clearsky_ghi'] = clear_sky_ghi

if 'clearsky_ghi' not in val_data:
    clear_sky_ghi = compute_clearsky_ghi(
        val_data['timestamps'], val_data['latitude'], val_data['longitude'], val_data.get('solar_zenith_angle')
    )
    val_data['clearsky_ghi'] = clear_sky_ghi

if 'clearsky_ghi' not in test_data:
    clear_sky_ghi = compute_clearsky_ghi(
        test_data['timestamps'], test_data['latitude'], test_data['longitude'], test_data.get('solar_zenith_angle')
    )
    test_data['clearsky_ghi'] = clear_sky_ghi

if "clearsky_ghi" not in SELECTED_FEATURES:
    SELECTED_FEATURES += ["clearsky_ghi"]


# %%
from utils.plot_utils import plot_time_series

# Plot data for a few locations
fig = plot_time_series(train_data,
                       features=["clearsky_ghi", "ghi"],
                       location_idx=0,
                       start_idx=0,
                       n_steps=72)


# %% [markdown]
# ### 2.4 Data Normalization

# %%
from utils.normalize_utils import normalize_data

# Normalize data
norm_train_data, scalers = normalize_data(train_data, SELECTED_FEATURES, TARGET_VARIABLE, scalers=None, fit_scalers=True)
norm_val_data, _ = normalize_data(val_data, SELECTED_FEATURES, TARGET_VARIABLE, scalers=scalers, fit_scalers=False)
norm_test_data, _ = normalize_data(test_data, SELECTED_FEATURES, TARGET_VARIABLE, scalers=scalers, fit_scalers=False)

# Print normalized data shapes
print("Normalized data shapes:")
for key in norm_train_data:
    if isinstance(norm_train_data[key], np.ndarray):
        print(f"  {key}: {norm_train_data[key].shape}")

# %%
from utils.data_persistence import save_normalized_data, save_scalers

# Create metadata
train_metadata = {
    "selected_features": SELECTED_FEATURES,
    "target_variable": TARGET_VARIABLE,
    "selected_feature_org_shape": train_data[SELECTED_FEATURES[0]].shape,
    "target_org_shape": train_data[TARGET_VARIABLE].shape,
    "raw_files": train_files,
}
val_metadata = {
    "selected_features": SELECTED_FEATURES,
    "target_variable": TARGET_VARIABLE,
    "selected_feature_org_shape": val_data[SELECTED_FEATURES[0]].shape,
    "target_org_shape": val_data[TARGET_VARIABLE].shape,
    "raw_files": val_files,
}
test_metadata = {
    "selected_features": SELECTED_FEATURES,
    "target_variable": TARGET_VARIABLE,
    "selected_feature_org_shape": test_data[SELECTED_FEATURES[0]].shape,
    "target_org_shape": test_data[TARGET_VARIABLE].shape,
    "raw_files": test_files,
}

# Save normalized data
save_normalized_data(norm_train_data, output_dir="data/processed", prefix="train", add_timestamp=True)
save_normalized_data(norm_val_data, output_dir="data/processed", prefix="val", add_timestamp=True)
save_normalized_data(norm_test_data, output_dir="data/processed", prefix="test", add_timestamp=True)

# Save scalers
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save_scalers(scalers, filepath=f"data/processed/scalers_{timestamp}.pkl")



