"""
Utilities package for the GHI forecasting project.
"""

# Import main functions from utility modules for convenient access
from .data_loading_utils import (
    load_sites_sample,
    convert_to_local_time,
    compute_physical_constraints,
    load_data_chunk,
    load_dataset,
    combine_sites,
    combine_time_periods,
)

# from .model_utils import (
#     create_sequences,
#     create_multisite_sequences,
#     normalize_data,
#     split_train_test,
#     build_lstm_model,
#     train_model_with_wandb,
#     evaluate_model,
#     plot_training_history,
#     plot_predictions
# )

from .plot_utils import (
    plot_time_series,
    plot_predictions,
    plot_predictions_over_time,
    plot_solar_day_night,
    plot_time_features
)

from .features_utils import (
    compute_nighttime_mask,
    compute_clearsky_ghi,
    compute_physical_constraints
)

from .normalize_utils import (
    create_time_features,
    normalize_data,
    create_sequences
)

# Try to import wandb_utils if it exists
try:
    from .wandb_utils import *
except ImportError:
    pass
