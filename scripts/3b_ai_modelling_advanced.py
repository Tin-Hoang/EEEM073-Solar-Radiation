# %% [markdown]
# # Advanced Solar Radiation Forecasting Models
#
# This notebook implements state-of-the-art deep learning architectures for Global Horizontal Irradiance (GHI) forecasting using time series weather data. Building on the basic models in `3a_ai_modelling_basic.py`, this notebook explores more sophisticated architectures:
#
# 1. **Transformer** - Attention-based architecture, adapted for time series forecasting
# 2. **Informer** - Advanced Transformer variant optimized for long sequence time-series forecasting
# 3. **TSMixer** - Simple yet effective architecture for time series forecasting
# 4. **iTransformer** - Inverted Transformer architecture for time series forecasting
# 5. **Mamba** - Linear-Time Sequence Modeling with Selective State Spaces (SSMs)
#
# ## Prerequisites
#
# **IMPORTANT**: Before running this notebook, you must first run the `2_data_preprocessing.ipynb` script to prepare the normalized data. This script generates the train, validation, and test datasets needed for model training and evaluation.
#
# ## Workflow Overview
#
# 1. **Data Loading** - Load preprocessed time series datasets
# 2. **Model Training Setup** - Configure training parameters and utilities
# 3. **Advanced Model Training** - Train cutting-edge deep learning architectures
# 4. **Performance Evaluation** - Compare state-of-the-art models using various metrics
# 5. **Visualization** - Plot time series predictions and model comparisons

# %% [markdown]
# ## 0. Debug Mode
#
# **IMPORTANT**: Set to True for code debugging mode and False for actual training.
# In debug mode, the code will only run 10 batches/epoch for 10 epochs.

# %%
# Debug mode to test code. Set to False for actual training
DEBUG_MODE = True

# %% [markdown]
# # 1. Data Loading
#
# In this section, we load and prepare the preprocessed time series data for training our advanced models. The data includes various weather features like temperature, wind speed, solar angles, etc., used to predict the Global Horizontal Irradiance (GHI).

# %% [markdown]
# ### 1.1 Import modules and define hyperparameters
#
# Here, we define hyperparameters for model training, including the lookback window, batch size, and selected features.

# %%
# Load autoreload extension
# %load_ext autoreload
# Set autoreload to mode 1
# %autoreload 2

# Import required libraries
import os
import numpy as np
from datetime import datetime
import json

import torch
from torch.utils.data import DataLoader
import wandb
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.dates as mdates
# Local modules
from utils.data_persistence import load_scalers
from utils.plot_utils import plot_training_history, plot_evaluation_metrics
from utils.training_utils import train_model, evaluate_model, evaluate_inference_time
from utils.wandb_utils import is_wandb_enabled, set_wandb_flag, set_keep_run_open
from utils.model_utils import print_model_info, save_model

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# ========== Model training hyperparameters =========
PATIENCE = 5  # Early stopping patience
LR = 0.0001

if DEBUG_MODE:
    # Local debug settings (to check if the code is working)
    # Will only run 10 batches/epoch for 10 epochs
    N_EPOCHS = 10
    BATCH_SIZE = 3000
    NUM_WORKERS = 4
else:
    # Remote server settings (to train the model, recommend using Otter lab machine)
    N_EPOCHS = 30
    BATCH_SIZE = 2 ** 13   # = 8192 samples
    NUM_WORKERS = 16

# ================= Wandb settings =============
USE_WANDB = False if DEBUG_MODE else True
WANDB_USERNAME = "tin-hoang"  # Your wandb username
WANDB_PROJECT = "EEEM073-Solar-Radiation"  # Your wandb project name

# =========== Time series hyperparameters ===========
# Number of timesteps to look back when creating sequences
LOOKBACK = 24

# Choose features to use in modeling
TIME_FEATURES = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'month_sin', 'month_cos', 'dow_sin', 'dow_cos']
SELECTED_FEATURES = [
    'air_temperature',
    'wind_speed',
    'relative_humidity',
    'cloud_type',      # Categorical feature
    'solar_zenith_angle',
    'clearsky_ghi',
    'total_precipitable_water',
    'surface_albedo',
    'nighttime_mask',  # New field from preprocess_data
    'cld_opd_dcomp',
    'aod'
]
STATIC_FEATURES = ['latitude', 'longitude', 'elevation']
# Target variable
TARGET_VARIABLE = 'ghi'

# %% [markdown]
# ### 1.2 Create PyTorch Datasets and DataLoaders
#
# Here, we set up the PyTorch data pipeline by creating custom datasets and DataLoaders.

# %%
# Loading preprocessed data files generated from 2_data_preprocessing.py
# These files contain normalized time series data split into train, validation, and test sets
from utils.data_persistence import load_normalized_data

TRAIN_PREPROCESSED_DATA_PATH = "data/processed/train_normalized_20250430_145157.h5"
VAL_PREPROCESSED_DATA_PATH = "data/processed/val_normalized_20250430_145205.h5"
TEST_PREPROCESSED_DATA_PATH = "data/processed/test_normalized_20250430_145205.h5"

# Load sequences
train_data, metadata = load_normalized_data(TRAIN_PREPROCESSED_DATA_PATH)

SCALER_PATH = "data/processed/scalers_20250430_145206.pkl"
scalers = load_scalers(SCALER_PATH)

# Print metadata
print(f"Train set | Metadata: {metadata}")
# Print created time
print(f"Train set | Created time: {metadata['created_time'] if 'created_time' in metadata else 'No created time'}")
# Print raw files
print(f"Train set | Raw files: {metadata['raw_files'] if 'raw_files' in metadata else 'No raw files'}")

# Print data structure and shape
print(f"Train set | Data structure:")
for key, value in train_data.items():
    print(f"  {key} shape: {value.shape}")


# %%
# Creating PyTorch datasets from preprocessed data
# TimeSeriesDataset is a custom dataset class that formats the data for model training
from utils.timeseriesdataset import TimeSeriesDataset

# Create datasets
train_dataset = TimeSeriesDataset(TRAIN_PREPROCESSED_DATA_PATH, lookback=LOOKBACK, target_field=TARGET_VARIABLE,
                                 selected_features=SELECTED_FEATURES, include_target_history=False,
                                 static_features=STATIC_FEATURES)
val_dataset = TimeSeriesDataset(VAL_PREPROCESSED_DATA_PATH, lookback=LOOKBACK, target_field=TARGET_VARIABLE,
                               selected_features=SELECTED_FEATURES, include_target_history=False,
                               static_features=STATIC_FEATURES)
test_dataset = TimeSeriesDataset(TEST_PREPROCESSED_DATA_PATH, lookback=LOOKBACK, target_field=TARGET_VARIABLE,
                                selected_features=SELECTED_FEATURES, include_target_history=False,
                                static_features=STATIC_FEATURES)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


# %%
# Examining data dimensions to configure model architectures
# Get a batch to determine input dimensions
batch = next(iter(train_loader))

# Check sample batch
sample_batch = next(iter(train_loader))
for key, value in sample_batch.items():
    if isinstance(value, torch.Tensor):
        print(f"{key} shape: {value.shape}")
    elif isinstance(value, list):
        print(f"{key} length: {len(value)}")

# Extract dimensions from a batch (more reliable)
temporal_features = batch['temporal_features']
static_features = batch['static_features']
TEMPORAL_FEATURES_SHAPE = list(temporal_features.shape)
STATIC_FEATURES_SHAPE = list(static_features.shape)

# Check if we have 3D temporal features (batch, seq_len, features)
if len(temporal_features.shape) == 3:
    temporal_dim = temporal_features.shape[2]
else:
    # Handle 2D temporal features (batch, features)
    temporal_dim = temporal_features.shape[1]

static_dim = static_features.shape[1]

print(f"  Input dimensions determined from batch:")
print(f"  - Batch temporal_features shape: {TEMPORAL_FEATURES_SHAPE}")
print(f"  - Batch static_features shape: {STATIC_FEATURES_SHAPE}")
print(f"  - Temporal dimension: {temporal_dim}")
print(f"  - Static dimension: {static_dim}")


# %% [markdown]
# ## 2. Model Training Setup
#
# This section configures the training environment, including setting up experiment tracking, defining the training pipeline, and preparing evaluation metrics.

# %% [markdown]
# ## 2.1 Setting Wandb logging (optional)
#
# Weights & Biases (wandb) is used for experiment tracking. Here we configure whether to use wandb for logging model training progress and results.

# %%
# Enable wandb tracking
set_wandb_flag(USE_WANDB)
# Keep the wandb run open after training to continue logging evaluation plots
set_keep_run_open(True)


# %% [markdown]
# ## 2.2 Setup Experiment Pipeline
#
# We define a standardized pipeline for training and evaluating models. This function handles the entire workflow:
# 1. Model training with early stopping (use train and val set)
# 2. Evaluation on test data
# 3. Saving model checkpoints
# 4. Logging results to wandb (if enabled)

# %%
def run_experiment_pipeline(model, train_loader, val_loader, test_loader, model_name, epochs=30, patience=5, lr=0.001):
    """
    Run the experiment pipeline for a given model.

    Args:
        model: The model to train.
        train_loader: The training data loader.
        val_loader: The validation data loader.
        test_loader: The test data loader.
        model_name: The name of the model.
        epochs: The number of epochs to train the model.
        patience: The number of epochs to wait before early stopping.
        lr: The learning rate for the model.
    """
    history, val_metrics, test_metrics = None, None, None

    # Get the current config
    CONFIG = {}
    cur_globals = globals().copy()
    for x in cur_globals:
        # Only get the variables that are uppercase and not digits
        if x.upper() == x and not x.startswith('_') and not x == "CONFIG":
            CONFIG[x] = cur_globals[x]

    try:
        print(f"Training {model_name} model...")
        history = train_model(
            model,
            train_loader,
            val_loader,
            model_name=model_name,
            epochs=epochs,
            patience=patience,
            lr=lr,
            target_scaler=scalers[f'{TARGET_VARIABLE}_scaler'],
            config=CONFIG,
            device=device,
            debug_mode=DEBUG_MODE,
        )
        training_plot = plot_training_history(history, model_name=model_name)

        print(f"\nEvaluating {model_name} model on test set...")
        test_metrics = evaluate_model(
            model,
            test_loader,
            scalers[f'{TARGET_VARIABLE}_scaler'],
            model_name=f"{model_name} - Test",
            device=device,
            debug_mode=DEBUG_MODE,
        )
        # Evaluate inference time
        timing_metrics = evaluate_inference_time(model,
                                                 test_loader,
                                                 model_name=f"{model_name} - Test",
                                                 timing_iterations=3,
                                                 debug_mode=DEBUG_MODE)
        test_metrics.update(timing_metrics)
        test_plot = plot_evaluation_metrics(test_metrics, model_name=f"{model_name} - Test")

        # ========== Save Best Model Checkpoint ===========
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Construct filename with timestamp and directory
        model_filename = f"{model_name}_best_{timestamp}.pt"
        model_path = os.path.join(checkpoint_dir, model_filename)

        # Combine time keys and selected features for the complete temporal feature set
        all_temporal_features = TIME_FEATURES + SELECTED_FEATURES

        # Save the model with metadata using the new save_model function
        save_model(
            model=model,
            filepath=model_path,
            metadata={
                "model_name": model_name,
                "timestamp": timestamp,
                "train_metrics": {
                    "final_train_loss": history["train_loss"][-1] if history and "train_loss" in history else None,
                    "final_train_mae": history["train_mae"][-1] if history and "train_mae" in history else None,
                    "final_val_loss": history["val_loss"][-1] if history and "val_loss" in history else None,
                    "final_val_mae": history["val_mae"][-1] if history and "val_mae" in history else None,
                },
                "test_metrics": {
                    "mse": test_metrics["mse"] if test_metrics else None,
                    "rmse": test_metrics["rmse"] if test_metrics else None,
                    "mae": test_metrics["mae"] if test_metrics else None,
                    "r2": test_metrics["r2"] if test_metrics else None,
                    "mase": test_metrics["mase"] if test_metrics else None,
                }
            },
            temporal_features=all_temporal_features,
            static_features=STATIC_FEATURES,
            time_feature_keys=TIME_FEATURES,
            config=CONFIG
        )

        print(f"Best model saved to {model_path}")

        # Log saved model path to wandb if enabled
        if is_wandb_enabled():
            wandb.save(model_path)
            print(f"Saved model checkpoint logged to wandb: {model_path}")
            wandb.log({"plots/history_plot": wandb.Image(training_plot)})
            wandb.log({"plots/predictions_plot": wandb.Image(test_plot)})

    finally:
        # Finish wandb run if it's still open
        if is_wandb_enabled():
            wandb.finish()

        # Clear GPU memory
        torch.cuda.empty_cache()

    return history, val_metrics, test_metrics


# %% [markdown]
# # 3. Advanced Model Experiments
#
# This section implements and trains state-of-the-art neural network architectures for GHI forecasting.
# These advanced models can capture more complex temporal patterns compared to basic architectures.

# %% [markdown]
# ### 3.1 Transformer Model
#
# The Transformer model, state-of-the-art for sequence modeling, using self-attention mechanisms to capture long-range dependencies.
#
# For detailed model code: `models/transformer.py`

# %%
from models.transformer import TransformerModel

# Create Transformer model
transformer_model = TransformerModel(
    input_dim=temporal_dim,           # Dimension of input features
    static_dim=static_dim,            # Dimension of static features
    d_model=128,                      # Model dimension
    n_heads=4,                        # Number of attention heads
    e_layers=2,                       # Number of encoder layers
    d_ff=256,                         # Dimension of feedforward network
    dropout=0.1,                      # Dropout rate
    activation='gelu'                 # Activation function
).to(device)

# Print the model
print_model_info(transformer_model, temporal_features.shape, static_features.shape)

# %%
model_name = "Transformer"

# Train the Transformer model
transformer_history, transformer_val_metrics, transformer_test_metrics = run_experiment_pipeline(
    transformer_model,
    train_loader,
    val_loader,
    test_loader,
    model_name=model_name,
    epochs=N_EPOCHS,
    patience=PATIENCE,
    lr=LR
)

# %% [markdown]
# ### 3.2 Informer Model
#
# The Informer model is a recent advancement in time series forecasting that addresses the computational limitations of standard Transformer models for long sequence prediction. Key innovations include:
#
# - **ProbSparse Self-attention**: Reduces complexity from O(L²) to O(L log L) where L is sequence length.
# - **Self-attention Distilling**: Progressive downsampling of hidden states along the encoder.
# - **Generative Decoder**: Enables long sequence prediction with minimal compute.
#
# For solar radiation forecasting, Informer can efficiently capture daily, weekly, and seasonal patterns while focusing computational resources on the most informative timestamps.
#
# For detailed model code: `models/informer.py`

# %%
from models.informer import InformerModel

# Create Informer model
informer_model = InformerModel(
    input_dim=temporal_dim,
    static_dim=static_dim,
    d_model=128,
    n_heads=4,
    e_layers=2,
    d_ff=256,
    dropout=0.1,
    activation='gelu',
).to(device)

# Print the model
print_model_info(informer_model, temporal_features.shape, static_features.shape)


# %%
model_name = "Informer"

# Train the Informer model with a lower learning rate
informer_history, informer_val_metrics, informer_test_metrics = run_experiment_pipeline(
    informer_model,
    train_loader,
    val_loader,
    test_loader,
    model_name=model_name,
    epochs=N_EPOCHS,
    patience=PATIENCE,
    lr=LR
)

# %% [markdown]
# ### 3.3 TSMixer Model
#
# TSMixer is a simple yet effective architecture for time series forecasting that applies the ideas of MLP-Mixer to time series data:
#
# - **Separate time and feature mixing**: Processes temporal and feature dimensions separately
# - **Parameter efficiency**: Uses simple MLP blocks to mix information across dimensions
# - **Fast training**: Simple architecture allows for efficient training
#
# For detailed model code: `models/tsmixer.py`

# %%
from models.tsmixer import TSMixerModel

# Create TSMixer model
tsmixer_model = TSMixerModel(
    input_dim=temporal_dim,
    static_dim=static_dim,
    lookback=LOOKBACK,
    horizon=1,             # Single-step prediction
    ff_dim=256,            # Feed-forward dimension
    num_blocks=10,          # Number of TSMixer blocks
    norm_type='batch',
    activation='relu',
    dropout=0.1
).to(device)

# Print the model
print_model_info(tsmixer_model, temporal_features.shape, static_features.shape)


# %%
model_name = "TSMixer"

# Train the TSMixer model
tsmixer_history, tsmixer_val_metrics, tsmixer_test_metrics = run_experiment_pipeline(
    tsmixer_model,
    train_loader,
    val_loader,
    test_loader,
    model_name=model_name,
    epochs=N_EPOCHS,
    patience=PATIENCE,
    lr=LR
)

# %% [markdown]
# ### 3.4 iTransformer Model
#
# iTransformer is an innovative approach to time series forecasting that inverts the traditional Transformer architecture:
#
# - **Inverted Attention Mechanism**: Applies self-attention across the feature dimension rather than the time dimension
# - **Feature as Tokens**: Treats each feature as a token (rather than each timestamp)
# - **Improved Feature Interactions**: Better captures correlations between different variables
#
# For detailed model code: `models/itransformer.py`

# %%
from models.itransformer import iTransformerModel

# Create iTransformer model
itransformer_model = iTransformerModel(
    input_dim=temporal_dim,           # Number of input features
    static_dim=static_dim,            # Number of static features
    d_model=128,                      # Model dimension
    n_heads=4,                        # Number of attention heads
    e_layers=2,                       # Number of encoder layers
    d_ff=256,                         # Dimension of feedforward network
    dropout=0.1,                      # Dropout rate
    lookback=LOOKBACK,                # Historical sequence length
    pred_len=1                        # Prediction length
).to(device)

# Print the model
print_model_info(itransformer_model, temporal_features.shape, static_features.shape)


# %%
model_name = "iTransformer"

# Train the iTransformer model
itransformer_history, itransformer_val_metrics, itransformer_test_metrics = run_experiment_pipeline(
    itransformer_model,
    train_loader,
    val_loader,
    test_loader,
    model_name=model_name,
    epochs=N_EPOCHS,
    patience=PATIENCE,
    lr=LR
)

# %% [markdown]
# ### 3.5 Mamba Model
#
# Mamba is a state-of-the-art architecture that uses State Space Models (SSMs) instead of attention mechanisms:
#
# - **Selective State Space Modeling**: Captures long-range dependencies with linear scaling to sequence length
# - **Data-dependent Parameters**: Adapts model parameters based on input data
# - **Local Convolution**: Enhances local pattern recognition with a convolutional layer
# - **Efficient Processing**: Linear time complexity O(L) compared to quadratic complexity O(L²) of Transformers
#
# For detailed model code: `models/mamba.py`

# %%
from models.mamba import MambaModel

# Create Mamba model
mamba_model = MambaModel(
    input_dim=temporal_dim,           # Number of input features
    static_dim=static_dim,            # Number of static features
    d_model=128,                      # Model dimension
    d_state=16,                       # State dimension for SSM
    n_layers=2,                       # Number of Mamba blocks
    dt_rank=32,                       # Rank for delta (Δ) projection
    d_conv=4,                         # Kernel size for local convolution
    expand_factor=2,                  # Expansion factor for inner dimension
    dt_min=0.001,                     # Minimum delta value
    dt_max=0.1,                       # Maximum delta value
    dropout=0.1,                      # Dropout rate
).to(device)

# Print the model
print_model_info(mamba_model, temporal_features.shape, static_features.shape)


# %%
model_name = "Mamba"

# Train the Mamba model
mamba_history, mamba_val_metrics, mamba_test_metrics = run_experiment_pipeline(
    mamba_model,
    train_loader,
    val_loader,
    test_loader,
    model_name=model_name,
    epochs=N_EPOCHS,
    patience=PATIENCE,
    lr=LR
)

# %% [markdown]
# ## 4. Model Comparison
#
# After training all models, we compare their performance to determine which advanced architecture works best for GHI forecasting.

# %% [markdown]
# ## 4.1 Compare Advanced Models' Performance
#
# This section compares the overall performance metrics (MSE, RMSE, MAE, MASE, R²) of all trained advanced models
# on the test dataset. These metrics help us understand which state-of-the-art model provides the most
# accurate predictions across the entire test set.

# %%
from utils.plot_utils import compare_models

# Create a dictionary of model metrics
model_metrics = {
    'Transformer': transformer_test_metrics,
    'Informer': informer_test_metrics,
    'TSMixer': tsmixer_test_metrics,
    'iTransformer': itransformer_test_metrics,
    'Mamba': mamba_test_metrics
}
# Drop the 'y_pred' and 'y_true' keys from the model metrics
for model in model_metrics:
    model_metrics[model].pop('y_pred', None)
    model_metrics[model].pop('y_true', None)
    model_metrics[model].pop('nighttime_mask', None)

# Save model metrics to a json file for later use
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
json_file_path = f'plots/advanced_model_metrics_{timestamp}.json'
# Fix TypeError: Object of type float32 is not JSON serializable
for model in model_metrics:
    for key, value in model_metrics[model].items():
        if isinstance(value, np.float32):
            model_metrics[model][key] = float(value)
with open(json_file_path, 'w') as f:
    json.dump(model_metrics, f)

# Compare model performance on test set
fig = compare_models(model_metrics, dataset_name='Test')

# %% [markdown]
# ## 4.2 Model Comparison on Daytime/Nighttime/Overall
#
# Here we analyze model performance separately for daytime and nighttime periods. This is crucial for solar forecasting
# as prediction requirements and patterns differ significantly between day and night. The comparison helps identify
# which advanced models perform better under different lighting conditions.

# %%
from utils.plot_utils import compare_models_daytime_nighttime

# Generate the comparison plot
comparison_fig = compare_models_daytime_nighttime(model_metrics, dataset_name='Test')


# %% [markdown]
# ## 5. Visualization and Analysis
#
# This section provides visual analysis of advanced model predictions to better understand model performance.

# %% [markdown]
# ### 5.1 Time Series Predictions
#
# Visualize predictions over time to compare how each advanced model tracks the actual GHI values. This visualization includes:
# - Actual GHI values (ground truth)
# - Predictions from each advanced model architecture
# - Nighttime periods shaded for context
# - Error metrics for the visualized time period

# %%
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


# %%
# Plot time series predictions for advanced models
_ = plot_predictions_over_time(
    models=[transformer_model, informer_model, tsmixer_model, itransformer_model, mamba_model],
    model_names=['Transformer', 'Informer', 'TSMixer', 'iTransformer', 'Mamba'],
    data_loader=test_loader,
    target_scaler=scalers[f'{TARGET_VARIABLE}_scaler'],
    num_samples=72,
    start_idx=40
)
