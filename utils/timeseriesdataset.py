import torch
from torch.utils.data import Dataset
import numpy as np
import gc
import os


class TimeSeriesDataset(Dataset):
    def __init__(self, preprocessed_data_path=None, seq_data=None, targets=None,
                 lazy_loading=False, dtype=torch.float32, lookback=24,
                 target_field='ghi', selected_features=None,
                 include_target_history=True, time_feature_keys=None,
                 static_features=None):
        """
        Dataset for time series forecasting with memory optimization and customizable field names.

        Can be initialized either from preprocessed data files or directly from data objects.

        Args:
            preprocessed_data_path: Path to preprocessed data file(s) (string or list of strings)
                                   If provided, seq_data and targets parameters are ignored
            seq_data: Dictionary of sequence arrays (used only if preprocessed_data_path is None)
            targets: Target values (used only if preprocessed_data_path is None)
            lazy_loading: If True, tensors are only created when accessed (reduces memory)
            dtype: Tensor data type (torch.float32 or torch.float16 for half precision)
            lookback: Number of timesteps to look back when creating sequences (default: 24)
            target_field: Name of the field to use as target (default: 'ghi')
                         Can be a string for single target or list of strings for multiple targets
            selected_features: List of feature names to include in the prediction input.
                              If None, all available features are used.
            include_target_history: Whether to include past values of target field(s) for autoregressive
                                   modeling (default: True)
            time_feature_keys: List of individual time feature keys (default: standard cyclical time features)
                              If None, uses the standard set from normalize_utils.create_time_features
            static_features: List of static feature field names (required)
        """
        if static_features is None:
            raise ValueError("static_features parameter is required and must contain the list of static feature fields")

        self.lazy_loading = lazy_loading
        self.dtype = dtype
        self.lookback = lookback
        self.static_features = static_features

        # Define the standard time feature keys used in normalize_utils.create_time_features
        if time_feature_keys is None:
            self.time_feature_keys = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                                     'month_sin', 'month_cos', 'dow_sin', 'dow_cos']
        else:
            self.time_feature_keys = time_feature_keys

        print(f"Using time feature keys: {self.time_feature_keys}")
        print(f"Using static features: {self.static_features}")

        # Handle single target field or multiple target fields
        self.target_field = target_field
        self.target_fields = [target_field] if isinstance(target_field, str) else list(target_field)

        self.selected_features = selected_features
        self.include_target_history = include_target_history

        # Initialize tensor containers
        self.tensors = {}
        self.temporal_features = None
        self.static_features_tensor = None
        self.targets_tensor = None

        # Print autoregressive settings
        if self.include_target_history:
            print(f"Using autoregressive mode: excluding current timestep")
        else:
            print(f"Not using target field history for autoregression")

        # Load data if path is provided
        if preprocessed_data_path is not None:
            self._seq_data, self._targets = self._load_from_files(preprocessed_data_path)
        else:
            # Use directly provided data
            if seq_data is None or targets is None:
                raise ValueError("Either preprocessed_data_path or both seq_data and targets must be provided")
            self._seq_data = seq_data
            self._targets = targets

        # Filter features based on selected_features if provided
        if self.selected_features is not None:
            # Create a filtered version of seq_data
            self._filtered_seq_data = {}

            # Always include certain core features
            core_features = self.static_features.copy()

            # Include time feature keys
            core_features.extend(self.time_feature_keys)

            # Always include the target field(s) in the data
            for field in self.target_fields:
                if field not in core_features:
                    core_features.append(field)

            # Copy only selected features and core features
            for key, value in self._seq_data.items():
                if key in self.selected_features or key in core_features:
                    self._filtered_seq_data[key] = value

            # Print which features will be used
            feature_keys = [k for k in self._filtered_seq_data.keys()
                          if k not in core_features or (k in self.target_fields and self.include_target_history)]
            print(f"Using {len(feature_keys)} selected features: {feature_keys}")

            # Use filtered data for feature identification
            feature_data = self._filtered_seq_data
        else:
            # Use all data for feature identification
            feature_data = self._seq_data
            print(f"Using all available features")

        # Identify temporal and static features using the filtered or complete data
        self._identify_features(feature_data)

        # Get number of timesteps and locations from data
        self._get_data_dimensions()

        # Get dataset size - number of possible windows
        self.size = self._get_dataset_size()

        if not lazy_loading:
            # Convert data to tensors immediately (uses more memory)
            self._initialize_tensors()
        else:
            # Calculate and print memory estimate
            self._estimate_memory_usage()

            # Initialize tensor placeholders
            self._init_tensor_placeholders()

    def _load_from_files(self, data_paths):
        """
        Load data from file(s) and extract target data

        Args:
            data_paths: String path or list of paths to data files

        Returns:
            seq_data: Combined data dictionary
            targets: Target values array
        """
        from utils.data_persistence import load_normalized_data, load_sequences

        # Convert single path to list for uniform processing
        if isinstance(data_paths, str):
            data_paths = [data_paths]

        # Load and combine data from all specified files
        combined_data = {}
        for path_idx, path in enumerate(data_paths):
            try:
                # First try to load as normalized data (newer format)
                try:
                    data_dict, metadata = load_normalized_data(path)
                    print(f"Loaded normalized data file ({path_idx+1}/{len(data_paths)}): {path}")
                except (KeyError, ValueError):
                    # If that fails, try to load as sequence data (legacy format)
                    data_dict, _, metadata = load_sequences(path)
                    print(f"Loaded legacy sequence file ({path_idx+1}/{len(data_paths)}): {path}")

                # For the first file, initialize the combined data
                if not combined_data:
                    combined_data = data_dict.copy()
                else:
                    # For subsequent files, append time series data along the time dimension
                    for key, value in data_dict.items():
                        if key in combined_data:
                            # Check if it's a time series (has first dimension matching timesteps)
                            if (len(value.shape) == 2 and len(combined_data[key].shape) == 2 and
                                value.shape[1] == combined_data[key].shape[1]):
                                # Append along time dimension (axis 0)
                                combined_data[key] = np.concatenate([combined_data[key], value], axis=0)
                            elif key not in ['latitude', 'longitude', 'elevation'] and value.shape != combined_data[key].shape:
                                print(f"Warning: Shape mismatch for {key}, skipping append")
                        else:
                            # New key, just add it
                            combined_data[key] = value

            except Exception as e:
                print(f"Error loading file {path}: {e}")

        # Check if target field exists
        if self.target_field not in combined_data:
            raise ValueError(f"Target field '{self.target_field}' not found in data")

        # Extract targets
        targets = combined_data[self.target_field]

        # Report loaded data size
        print(f"Loaded data with {len(combined_data)} features")

        return combined_data, targets

    def _get_data_dimensions(self):
        """Get the dimensions of the dataset (timesteps, locations)"""
        # Check if we can infer dimensions from time series features
        for key in self.time_series_keys:
            if key in self._seq_data:
                self.n_timesteps, self.n_locations = self._seq_data[key].shape
                break

        # If we couldn't infer from time series, try other approaches
        if not hasattr(self, 'n_timesteps') or not hasattr(self, 'n_locations'):
            # Try to infer from time_features
            if 'time_features' in self._seq_data:
                self.n_timesteps = self._seq_data['time_features'].shape[0]

            # Try to infer from latitude/longitude
            if 'latitude' in self._seq_data:
                self.n_locations = self._seq_data['latitude'].shape[0]
            elif 'longitude' in self._seq_data:
                self.n_locations = self._seq_data['longitude'].shape[0]

            # If still missing dimensions, use targets
            if not hasattr(self, 'n_timesteps') or not hasattr(self, 'n_locations'):
                if isinstance(self._targets, np.ndarray) and len(self._targets.shape) == 2:
                    self.n_timesteps, self.n_locations = self._targets.shape
                else:
                    # Last resort - assume targets are for the final timestep
                    self.n_locations = len(self._targets)
                    self.n_timesteps = 1

        print(f"Dataset dimensions: {self.n_timesteps} timesteps, {self.n_locations} locations")

    def _get_dataset_size(self):
        """Calculate the number of possible sample windows"""
        # If using lookback, size is (timesteps - lookback) * locations
        if self.lookback > 0:
            effective_timesteps = max(0, self.n_timesteps - self.lookback)
            size = effective_timesteps * self.n_locations
        else:
            # Without lookback, size is just timesteps * locations
            size = self.n_timesteps * self.n_locations

        print(f"Dataset contains {size} possible samples")
        return size

    def _identify_features(self, data_dict=None):
        """
        Identify temporal and static features in the dataset

        For the common structure with (time, locations) features:
        - Individual time features (hour_sin, hour_cos, etc.) are treated as temporal features
        - Static features are defined by the static_features parameter
        - Most features have shape (time, locations)

        Args:
            data_dict: Dictionary containing data to identify features from.
                      If None, uses self._seq_data
        """
        if data_dict is None:
            data_dict = self._seq_data

        self.temporal_feature_keys = []
        self.static_feature_keys = self.static_features.copy()  # Use the explicitly provided static features
        self.time_series_keys = []
        self.other_keys = []

        # Check each key in the seq_data dictionary
        for key, value in data_dict.items():
            # Skip target fields to avoid mixing with features
            if key in self.target_fields:
                continue

            if not isinstance(value, np.ndarray):
                continue

            # Check shape and identify type
            shape = value.shape

            # Individual time features with shape (time,)
            if key in self.time_feature_keys and len(shape) == 1:
                self.temporal_feature_keys.append(key)
            # Skip static features as they are already defined by the static_features parameter
            elif key in self.static_feature_keys:
                continue
            # Time series data with shape (time, locations)
            elif len(shape) == 2 and key not in self.time_feature_keys + self.static_feature_keys:
                self.time_series_keys.append(key)
            # Traditional 3D temporal features (samples, timesteps, features)
            elif len(shape) == 3:
                self.temporal_feature_keys.append(key)
            # Other features
            else:
                self.other_keys.append(key)

        print(f"Temporal features: {self.temporal_feature_keys}")
        print(f"Static features: {self.static_feature_keys}")
        print(f"Time series features: {self.time_series_keys}")
        if self.other_keys:
            print(f"Other data: {self.other_keys}")

    def _init_tensor_placeholders(self):
        """Initialize all tensor placeholders to None"""
        # Placeholders for individual tensors
        self.tensors = {key: None for key in self._seq_data.keys()}

    def _initialize_tensors(self):
        """Initialize all tensors upfront (high memory usage)"""
        self.preload_all(during_init=True)

    def preload_all(self, during_init=False):
        """
        Explicitly load all tensors into memory at once

        Args:
            during_init: Whether this is being called during initialization
        """
        if during_init or self.lazy_loading:
            print("Loading all tensors into memory...")

            # Convert all data to tensors
            self.tensors = {}

            # Process all features in seq_data
            for key, value in self._seq_data.items():
                if isinstance(value, np.ndarray):
                    # Handle numpy.bytes_ arrays by converting to strings
                    if value.dtype.type == np.bytes_:
                        # Convert bytes to strings
                        string_array = np.array([s.decode('utf-8') for s in value.flatten()]).reshape(value.shape)
                        # Store as Python list instead of tensor
                        self.tensors[key] = string_array
                    else:
                        # Default tensor conversion for supported numeric types
                        self.tensors[key] = torch.tensor(value, dtype=self.dtype)

            # Convert targets to tensor
            self._get_targets()

            # Initialize temporal and static features
            self._get_temporal_features()
            self._get_static_features()

            # Load any remaining tensors
            for key in self._seq_data.keys():
                if key not in self.tensors or self.tensors[key] is None:
                    self._get_tensor(key)

            # Free original data to save memory unless in strict mode
            if not self.lazy_loading or (self.lazy_loading != "strict"):
                del self._seq_data
                del self._targets
                self._seq_data = None
                self._targets = None
                gc.collect()

                # Update lazy_loading flag if we're no longer lazy
                if self.lazy_loading:
                    self.lazy_loading = False
                    print("All tensors loaded, original data cleared")
            else:
                print("All tensors loaded, keeping original data for strict mode")

    def _get_temporal_features(self, idx=None):
        """Get or create temporal features tensor"""
        if self.temporal_features is None:
            # Initialize lists to store tensors
            temporal_tensors = []

            # Process individual time features
            individual_time_features = []
            for key in self.time_feature_keys:
                if key in self.tensors and self.tensors[key] is not None:
                    # Get the tensor with shape (time,)
                    time_tensor = self.tensors[key]

                    # Reshape to (time, 1, 1) then expand to (time, locations, 1)
                    time_feature = time_tensor.unsqueeze(1).unsqueeze(2).repeat(1, self.n_locations, 1)
                    individual_time_features.append(time_feature)

            # Combine individual time features if any
            if individual_time_features:
                # Combine along feature dimension
                combined_time = torch.cat(individual_time_features, dim=2)  # (time, locations, n_features)
                temporal_tensors.append(combined_time)

            # Process time series features
            time_series_features = []
            for key in self.time_series_keys:
                if key in self.tensors and self.tensors[key] is not None:
                    # Get the tensor with shape (time, locations)
                    ts_tensor = self.tensors[key]

                    # Reshape to (time, locations, 1) for concatenation
                    ts_feature = ts_tensor.unsqueeze(2)  # (time, locations, 1)
                    time_series_features.append(ts_feature)

            # Concatenate time series features if any
            if time_series_features:
                # Combine along feature dimension
                combined_ts = torch.cat(time_series_features, dim=2)  # (time, locations, n_features)
                temporal_tensors.append(combined_ts)

            # Combine all temporal features
            if len(temporal_tensors) > 0:
                self.temporal_features = torch.cat(temporal_tensors, dim=2)
            else:
                self.temporal_features = None

            # Free individual tensors if we're in strict memory saving mode
            if self.lazy_loading == "strict":
                for key in self.temporal_feature_keys + self.time_series_keys:
                    if key in self.tensors:
                        del self.tensors[key]
                        self.tensors[key] = None
                gc.collect()

        if idx is not None and self.temporal_features is not None:
            return self.temporal_features[idx]
        return self.temporal_features

    def _get_static_features(self, idx=None):
        """Get or create static features tensor"""
        if self.static_features_tensor is None:
            static_features_list = []

            # Add static features based on the explicitly provided list
            for key in self.static_feature_keys:
                if key in self._seq_data:
                    feature_tensor = self._get_tensor(key)
                    # Ensure it's 2D by adding feature dimension if needed
                    if len(feature_tensor.shape) == 1:
                        feature_tensor = feature_tensor.unsqueeze(1)
                    static_features_list.append(feature_tensor)

            # Combine static features if we have any
            if static_features_list:
                if len(static_features_list) > 1:
                    # Concatenate along feature dimension
                    self.static_features_tensor = torch.cat(static_features_list, dim=1)
                else:
                    # Just use the single static feature
                    self.static_features_tensor = static_features_list[0]
            else:
                # No static features available - create a placeholder
                self.static_features_tensor = torch.zeros((self.n_locations, 1), dtype=self.dtype)
                print("Warning: No static features available, using zero tensor as placeholder")

            # Free tensors if we're in strict memory saving mode
            if self.lazy_loading == "strict":
                for key in self.static_feature_keys:
                    if key in self.tensors:
                        del self.tensors[key]
                        self.tensors[key] = None
                gc.collect()

        if idx is not None:
            return self.static_features_tensor[idx]
        return self.static_features_tensor

    def _get_targets(self, idx=None):
        """Get or create targets tensor"""
        if self.targets_tensor is None:
            # Check if targets are numpy.bytes_ type
            if isinstance(self._targets, np.ndarray) and self._targets.dtype.type == np.bytes_:
                # Convert bytes to strings
                string_array = np.array([s.decode('utf-8') for s in self._targets.flatten()]).reshape(self._targets.shape)
                # Store as Python list instead of tensor
                self.targets_tensor = string_array
            else:
                # Convert targets to tensor
                self.targets_tensor = torch.tensor(self._targets, dtype=self.dtype)
                if len(self.targets_tensor.shape) == 1:
                    self.targets_tensor = self.targets_tensor.unsqueeze(1)

        if idx is not None:
            return self.targets_tensor[idx]
        return self.targets_tensor

    def _get_other_tensor(self, key, idx=None):
        """Get or create any other tensor"""
        if key in self.tensors and self.tensors[key] is None:
            # Convert to tensor
            tensor = torch.tensor(self._seq_data[key], dtype=self.dtype)

            # Ensure 2D tensor for scalar values
            if len(tensor.shape) == 1:
                tensor = tensor.unsqueeze(1)

            self.tensors[key] = tensor

        if idx is not None and key in self.tensors:
            return self.tensors[key][idx]
        return self.tensors.get(key, None)

    def _extract_window(self, global_idx):
        """
        Extract sequence data for a specific index.

        Args:
            global_idx: Global index (0 to self.size-1)

        Returns:
            time_idx: Time index for the end of the window
            loc_idx: Location index
            window_start: Start index of the window
        """
        # Convert global index to (time, location) coordinates
        effective_timesteps = max(1, self.n_timesteps - self.lookback)
        loc_idx = global_idx // effective_timesteps
        rel_time_idx = global_idx % effective_timesteps

        # Calculate actual time index (after lookback window)
        time_idx = rel_time_idx + self.lookback

        # Window start is the beginning of the lookback window
        window_start = time_idx - self.lookback

        return time_idx, loc_idx, window_start

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        Get a specific item with lookback window

        If lookback > 0, this returns a sequence window for the location
        Otherwise, it returns a single timestep
        """
        # Convert global index to time and location indices
        time_idx, loc_idx, window_start = self._extract_window(idx)

        if not self.lazy_loading:
            # All tensors already created upfront
            result = {}

            # Add static features
            if self.static_features_tensor is not None:
                result['static_features'] = self.static_features_tensor[loc_idx]

            # Add target
            if isinstance(self.targets_tensor, (list, np.ndarray)) and not isinstance(self.targets_tensor, torch.Tensor):
                # Handle string arrays
                if len(np.array(self.targets_tensor).shape) > 1:
                    result['target'] = self.targets_tensor[time_idx][loc_idx]
                else:
                    result['target'] = self.targets_tensor[loc_idx]
            else:
                # Handle tensor targets
                if len(self.targets_tensor.shape) > 1:
                    result['target'] = self.targets_tensor[time_idx, loc_idx]
                else:
                    result['target'] = self.targets_tensor[loc_idx]

            # Add temporal features (with lookback window if applicable)
            if self.temporal_features is not None:
                if self.lookback > 0:
                    # Get sequence window
                    if len(self.temporal_features.shape) == 3:  # (time, locations, features)
                        result['temporal_features'] = self.temporal_features[window_start:time_idx, loc_idx, :]
                    else:  # Time features (time, features)
                        result['temporal_features'] = self.temporal_features[window_start:time_idx, :]
                else:
                    # For single timestep
                    if len(self.temporal_features.shape) == 3:  # (time, locations, features)
                        result['temporal_features'] = self.temporal_features[time_idx, loc_idx, :]
                    else:  # Time features (time, features)
                        result['temporal_features'] = self.temporal_features[time_idx, :]

            # Handle target history separately for autoregressive modeling
            if self.include_target_history and self.lookback > 0:
                target_histories = []
                for target_field in self.target_fields:
                    if target_field in self.tensors:
                        # Get target history window for lookback steps
                        # We want data from (time_idx - lookback) to (time_idx - 1)
                        history_start = time_idx - self.lookback
                        history_end = time_idx  # exclusive

                        # Handle case where history_start is negative (at the beginning of dataset)
                        if history_start >= 0:
                            # Normal case - full history available
                            target_history = self.tensors[target_field][history_start:history_end, loc_idx]
                        else:
                            # Need to pad the history with earliest available values
                            padding_needed = abs(history_start)
                            available_history = self.tensors[target_field][0:history_end, loc_idx]

                            # Create padding using the earliest available value
                            earliest_value = self.tensors[target_field][0, loc_idx]
                            padding = earliest_value.repeat(padding_needed)

                            # Combine padding with available history
                            target_history = torch.cat([padding, available_history])

                        target_histories.append(target_history.unsqueeze(1))  # Add feature dimension

                if target_histories:
                    # Combine along feature dimension if we have multiple targets
                    if len(target_histories) > 1:
                        combined_history = torch.cat(target_histories, dim=1)
                    else:
                        combined_history = target_histories[0]

                    # If there are existing temporal features, combine with them
                    if 'temporal_features' in result:
                        # Check dimension to decide how to concatenate
                        if len(result['temporal_features'].shape) == 2:
                            # (seq_len, features) - combine along feature dimension
                            if len(combined_history.shape) == 1:
                                # Need to add feature dimension
                                combined_history = combined_history.unsqueeze(1)
                            result['temporal_features'] = torch.cat([result['temporal_features'], combined_history], dim=1)
                        else:
                            # Handle other cases if needed
                            pass
                    else:
                        # No existing temporal features, just use the target history
                        result['temporal_features'] = combined_history

            # Add time series features (all shape time, locations)
            for key in self.time_series_keys:
                if key in self.tensors:
                    # Check if tensor is a string array
                    if isinstance(self.tensors[key], (list, np.ndarray)) and not isinstance(self.tensors[key], torch.Tensor):
                        # Handle string arrays
                        if self.lookback > 0:
                            # Get sequence window
                            string_data = self.tensors[key][window_start:time_idx]
                            if len(np.array(string_data).shape) > 1:
                                result[key] = [row[loc_idx] for row in string_data]
                            else:
                                result[key] = string_data
                        else:
                            # Get single timestep
                            if len(np.array(self.tensors[key]).shape) > 1:
                                result[key] = self.tensors[key][time_idx][loc_idx]
                            else:
                                result[key] = self.tensors[key][time_idx]
                    else:
                        # Non-target fields are included as separate entries for easier model access
                        if self.lookback > 0:
                            # Get sequence window
                            result[key] = self.tensors[key][window_start:time_idx, loc_idx]
                        else:
                            # Get single timestep
                            result[key] = self.tensors[key][time_idx, loc_idx]

            # Add time_index_local (timestamp as string)
            if 'time_index_local' in self.tensors:
                # Add the timestamp for the current time_idx
                result['time_index_local'] = self.tensors['time_index_local'][time_idx]

            # Add individual time features for current time step (for reference/debugging)
            for key in self.time_feature_keys:
                if key in self.tensors:
                    result[f'current_{key}'] = self.tensors[key][time_idx]

            return result
        else:
            # Lazy loading mode
            result = {}

            # Add static features
            static_features = self._get_static_features(loc_idx)
            if static_features is not None:
                result['static_features'] = static_features

            # Add target value
            if isinstance(self._targets, np.ndarray):
                if len(self._targets.shape) > 1:
                    # 2D targets array (time, locations)
                    target_val = self._targets[time_idx, loc_idx]
                else:
                    # 1D targets array (locations)
                    target_val = self._targets[loc_idx]
            else:
                # Scalar target
                target_val = self._targets

            result['target'] = torch.tensor(target_val, dtype=self.dtype)
            if len(result['target'].shape) == 0:  # Scalar
                result['target'] = result['target'].unsqueeze(0)

            # Add time features and other temporal features
            temporal_features_list = []

            # Add individual time features
            individual_time_features = []
            for key in self.time_feature_keys:
                if key in self._seq_data:
                    if self.lookback > 0:
                        # Get sequence window
                        time_feature = torch.tensor(
                            self._seq_data[key][window_start:time_idx],
                            dtype=self.dtype
                        )
                        # Add to current time features for reference
                        result[f'current_{key}'] = torch.tensor(self._seq_data[key][time_idx], dtype=self.dtype)
                        # Reshape to add feature dimension
                        individual_time_features.append(time_feature.unsqueeze(1))
                    else:
                        # Get single timestep
                        time_feature = torch.tensor(
                            self._seq_data[key][time_idx],
                            dtype=self.dtype
                        )
                        # Add to result directly for single timestep case
                        result[f'current_{key}'] = time_feature
                        individual_time_features.append(time_feature.unsqueeze(0))

            # Combine individual time features if any
            if individual_time_features:
                combined_time_features = torch.cat(individual_time_features, dim=-1)
                temporal_features_list.append(combined_time_features)

            # Add time series features
            for key in self.time_series_keys:
                if key in self._seq_data:
                    # For non-target fields, include all timesteps
                    if self.lookback > 0:
                        # Get sequence window for this location
                        feature_val = self._seq_data[key][window_start:time_idx, loc_idx]
                    else:
                        # Get single timestep for this location
                        feature_val = self._seq_data[key][time_idx, loc_idx]
                    result[key] = torch.tensor(feature_val, dtype=self.dtype)

            # Add time_index_local (timestamp as string)
            if 'time_index_local' in self._seq_data:
                # Check if it's bytes data
                if isinstance(self._seq_data['time_index_local'], np.ndarray) and self._seq_data['time_index_local'].dtype.type == np.bytes_:
                    # Convert from bytes to string
                    result['time_index_local'] = self._seq_data['time_index_local'][time_idx].decode('utf-8')
                else:
                    # Already a string or other format
                    result['time_index_local'] = self._seq_data['time_index_local'][time_idx]

            # Add target history for autoregressive modeling
            if self.include_target_history and self.lookback > 0:
                target_histories = []
                for target_field in self.target_fields:
                    if target_field in self._seq_data:
                        # Get target history window for lookback steps
                        # We want data from (time_idx - lookback) to (time_idx - 1)
                        history_start = time_idx - self.lookback
                        history_end = time_idx  # exclusive

                        # Handle case where history_start is negative (at the beginning of dataset)
                        if history_start >= 0:
                            # Normal case - full history available
                            target_history = self._seq_data[target_field][history_start:history_end, loc_idx]
                        else:
                            # Need to pad the history with earliest available values
                            padding_needed = abs(history_start)
                            available_history = self._seq_data[target_field][0:history_end, loc_idx]

                            # Create padding using the earliest available value
                            earliest_value = self._seq_data[target_field][0, loc_idx]
                            padding = np.repeat(earliest_value, padding_needed)

                            # Combine padding with available history
                            target_history = np.concatenate([padding, available_history])

                        # Convert to tensor
                        target_history_tensor = torch.tensor(target_history, dtype=self.dtype)
                        target_histories.append(target_history_tensor.unsqueeze(1))  # Add feature dimension

                if target_histories:
                    # Combine along feature dimension if we have multiple targets
                    if len(target_histories) > 1:
                        combined_history = torch.cat(target_histories, dim=1)
                    else:
                        combined_history = target_histories[0]

                    # Add to temporal features list
                    temporal_features_list.append(combined_history)

            # Combine all temporal features
            if temporal_features_list:
                if len(temporal_features_list) == 1:
                    result['temporal_features'] = temporal_features_list[0]
                else:
                    # Need to concatenate - this depends on the shape of each tensor
                    # For now, we assume they all have the same structure and concatenate on last dimension
                    result['temporal_features'] = torch.cat(temporal_features_list, dim=-1)

            return result

    @classmethod
    def from_file(cls, filepath, lazy_loading=False, dtype=torch.float32,
                 lookback=24, target_field='ghi', selected_features=None,
                 include_target_history=True, time_feature_keys=None,
                 static_features=None):
        """
        Create dataset directly from a saved file

        Alias for the constructor with preprocessed_data_path parameter

        Args:
            filepath: Path to the saved file
            lazy_loading: If True, tensors are only created when accessed
            dtype: Tensor data type
            lookback: Number of timesteps to look back
            target_field: Name of the field to use as target (default: 'ghi')
            selected_features: List of feature names to include in the prediction input
            include_target_history: Whether to include past values of target field(s) (default: True)
            time_feature_keys: List of individual time feature keys (default: standard cyclical time features)
            static_features: List of static feature field names (required)

        Returns:
            dataset: TimeSeriesDataset instance
        """
        return cls(
            preprocessed_data_path=filepath,
            lazy_loading=lazy_loading,
            dtype=dtype,
            lookback=lookback,
            target_field=target_field,
            selected_features=selected_features,
            include_target_history=include_target_history,
            time_feature_keys=time_feature_keys,
            static_features=static_features
        )

    def _estimate_memory_usage(self):
        """Estimate memory usage of the dataset"""
        total_bytes = 0

        # Calculate sizes of arrays
        for key, value in self._seq_data.items():
            if isinstance(value, np.ndarray):
                bytes_per_element = value.itemsize
                total_bytes += bytes_per_element * value.size

        # Add targets
        if isinstance(self._targets, np.ndarray):
            bytes_per_element = self._targets.itemsize
            total_bytes += bytes_per_element * self._targets.size

        # Convert to MB
        total_mb = total_bytes / (1024 * 1024)
        print(f"Estimated memory for raw data: {total_mb:.2f} MB")
        print(f"Approximate memory if converted to tensors: {total_mb * 1.5:.2f} MB")

    def _get_tensor(self, key, unsqueeze_dim=None):
        """Get tensor if already created, or create it on demand"""
        if self.tensors[key] is None:
            if key in self._seq_data:
                # Check if data is numpy.bytes_ type
                if isinstance(self._seq_data[key], np.ndarray) and self._seq_data[key].dtype.type == np.bytes_:
                    # Convert bytes to strings
                    string_array = np.array([s.decode('utf-8') for s in self._seq_data[key].flatten()]).reshape(self._seq_data[key].shape)
                    # Store as Python list instead of tensor
                    self.tensors[key] = string_array
                else:
                    # Convert to tensor
                    tensor = torch.tensor(self._seq_data[key], dtype=self.dtype)
                    if unsqueeze_dim is not None:
                        tensor = tensor.unsqueeze(unsqueeze_dim)
                    self.tensors[key] = tensor
            else:
                raise KeyError(f"Key {key} not found in seq_data")

        return self.tensors[key]
