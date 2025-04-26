import torch
from torch.utils.data import Dataset
import numpy as np
import gc
import os


class TimeSeriesDataset(Dataset):
    def __init__(self, preprocessed_data_path=None, seq_data=None, targets=None,
                 lazy_loading=False, dtype=torch.float32, field_names=None,
                 lookback=24, target_field='ghi'):
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
            field_names: Dictionary mapping standard field types to custom field names
                         e.g., {'time_field': 'time_features', 'coordinates_field': 'coordinates'}
            lookback: Number of timesteps to look back when creating sequences (default: 24)
            target_field: Name of the field to use as target (default: 'ghi')
        """
        self.lazy_loading = lazy_loading
        self.dtype = dtype
        self.lookback = lookback
        self.target_field = target_field

        # Define default field name mappings and update with user-provided ones
        self.field_names = {
            'time_field': 'time_features',
            'coordinates_field': 'coordinates',
            'elevation_field': 'elevation',
        }
        if field_names is not None:
            self.field_names.update(field_names)

        # Load data if path is provided
        if preprocessed_data_path is not None:
            self._seq_data, self._targets = self._load_from_files(preprocessed_data_path)
        else:
            # Use directly provided data
            if seq_data is None or targets is None:
                raise ValueError("Either preprocessed_data_path or both seq_data and targets must be provided")
            self._seq_data = seq_data
            self._targets = targets

        # Identify temporal and static features
        self._identify_features()

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
                            elif key not in ['coordinates', 'elevation'] and value.shape != combined_data[key].shape:
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
            if self.field_names['time_field'] in self._seq_data:
                self.n_timesteps = self._seq_data[self.field_names['time_field']].shape[0]

            # Try to infer from coordinates
            if self.field_names['coordinates_field'] in self._seq_data:
                self.n_locations = self._seq_data[self.field_names['coordinates_field']].shape[0]

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

    def _identify_features(self):
        """
        Identify temporal and static features in the dataset

        For the common structure with (time, locations) features:
        - time_features is treated as a special case with shape (time, features)
        - coordinates has shape (locations, 2)
        - elevation has shape (locations,)
        - Most features have shape (time, locations)
        """
        self.temporal_feature_keys = []
        self.static_feature_keys = []
        self.time_series_keys = []
        self.other_keys = []

        # Check each key in the seq_data dictionary
        for key, value in self._seq_data.items():
            if not isinstance(value, np.ndarray):
                continue

            # Check shape and identify type
            shape = value.shape

            # Special case for time_features
            if key == self.field_names['time_field'] and len(shape) == 2:
                # time_features with shape (time, features)
                self.temporal_feature_keys.append(key)
            # Coordinates with shape (locations, features)
            elif key == self.field_names['coordinates_field'] and len(shape) == 2 and shape[1] == 2:
                self.static_feature_keys.append(key)
            # Elevation with shape (locations,)
            elif key == self.field_names['elevation_field'] and len(shape) == 1:
                self.static_feature_keys.append(key)
            # Time series data with shape (time, locations)
            elif len(shape) == 2 and (
                key not in [self.field_names['time_field'], self.field_names['coordinates_field']]
            ):
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
        self.targets_tensor = None

        # Placeholders for combined features
        self.temporal_features = None
        self.static_features = None

    def _initialize_tensors(self):
        """Initialize all tensors upfront (high memory usage)"""
        # Convert all data to tensors
        self.tensors = {}

        # Process all features in seq_data
        for key, value in self._seq_data.items():
            if isinstance(value, np.ndarray):
                # Handle different feature types
                if key in self.temporal_feature_keys:
                    # No special handling for temporal features
                    self.tensors[key] = torch.tensor(value, dtype=self.dtype)
                elif key in self.static_feature_keys:
                    # Static features like coordinates
                    self.tensors[key] = torch.tensor(value, dtype=self.dtype)
                elif key in self.time_series_keys:
                    # Time series features with shape (time, locations)
                    self.tensors[key] = torch.tensor(value, dtype=self.dtype)
                else:
                    # Default tensor conversion
                    self.tensors[key] = torch.tensor(value, dtype=self.dtype)

        # Convert targets to tensor
        self.targets_tensor = torch.tensor(self._targets, dtype=self.dtype)
        if len(self.targets_tensor.shape) == 1:
            self.targets_tensor = self.targets_tensor.unsqueeze(1)

        # Create combined feature tensors
        self._create_combined_features()

        # Clear references to original data to free memory
        if hasattr(self, '_seq_data'):
            del self._seq_data
        if hasattr(self, '_targets'):
            del self._targets
        gc.collect()

    def _create_combined_features(self):
        """
        Create combined feature tensors for models that expect this format

        For the common structure:
        1. Extract time features from time_features array
        2. Reshape static features like elevation for each time step
        3. Combine with time series features for temporal data
        """
        # Combine temporal features if we have any
        temporal_tensors = []

        # Process time_features if available (shape: time, features)
        time_field = self.field_names['time_field']
        if time_field in self.tensors and self.tensors[time_field] is not None:
            # Get time features tensor
            time_features = self.tensors[time_field]

            # For 2D time features (time, features), we need to expand it
            # to match the dimensionality of other time series features
            if len(time_features.shape) == 2:
                # We'll reshape it when combining with other features
                temporal_tensors.append(time_features)

        # Get dimensions for reshaping
        n_timesteps = None
        n_locations = None

        # Try to get dimensions from a time series feature
        for key in self.time_series_keys:
            if key in self.tensors and self.tensors[key] is not None:
                n_timesteps, n_locations = self.tensors[key].shape
                break

        # Process elevation if available (shape: locations)
        elevation_field = self.field_names['elevation_field']
        if elevation_field in self.tensors and self.tensors[elevation_field] is not None:
            elevation = self.tensors[elevation_field]

            # If elevation is 1D (locations), we need to expand it to match
            # time series features (time, locations)
            if len(elevation.shape) == 1 and n_timesteps is not None:
                # Reshape to (1, locations) then repeat for each timestep
                elevation_expanded = elevation.unsqueeze(0).repeat(n_timesteps, 1)
                # Now elevation_expanded has shape (time, locations)
                elevation_feature = elevation_expanded.unsqueeze(2)  # (time, locations, 1)
                temporal_tensors.append(elevation_feature)

        # Process time series features (shape: time, locations)
        # We need to reshape them to (time*locations, 1) for sequence creation
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

        # Combine all temporal features if we have any
        if temporal_tensors:
            # For now, just use the first temporal tensor
            # Advanced combination would require reshaping to match dimensions
            self.temporal_features = temporal_tensors[0]
        else:
            self.temporal_features = None

        # Extract static features from coordinates
        coordinates_field = self.field_names['coordinates_field']
        if coordinates_field in self.tensors:
            # Get coordinates - shape (locations, 2)
            coords = self.tensors[coordinates_field]
            self.static_features = coords
        elif self.static_feature_keys:
            # If we have other static features, use the first one
            first_key = self.static_feature_keys[0]
            self.static_features = self.tensors[first_key]
        else:
            self.static_features = None

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
                # Convert to tensor
                tensor = torch.tensor(self._seq_data[key], dtype=self.dtype)
                if unsqueeze_dim is not None:
                    tensor = tensor.unsqueeze(unsqueeze_dim)
                self.tensors[key] = tensor
            else:
                raise KeyError(f"Key {key} not found in seq_data")

        return self.tensors[key]

    def _get_temporal_features(self, idx=None):
        """Get or create temporal features tensor"""
        if self.temporal_features is None:
            # Initialize lists to store tensors
            temporal_tensors = []

            # Get time features if available
            time_field = self.field_names['time_field']
            if time_field in self._seq_data:
                time_features = self._get_tensor(time_field)
                temporal_tensors.append(time_features)

            # Get elevation if available
            elevation_field = self.field_names['elevation_field']
            if elevation_field in self._seq_data:
                elevation = self._get_tensor(elevation_field)
                # Handle reshaping for elevation similar to _create_combined_features
                if len(elevation.shape) == 1:
                    # Get dimensions for reshaping
                    n_timesteps = None
                    n_locations = len(elevation)

                    # Try to get timesteps from a time series feature
                    for key in self.time_series_keys:
                        if key in self._seq_data:
                            n_timesteps = self._seq_data[key].shape[0]
                            break

                    if n_timesteps is not None:
                        # Reshape to (1, locations) then repeat for each timestep
                        elevation_expanded = elevation.unsqueeze(0).repeat(n_timesteps, 1)
                        # Add feature dimension
                        elevation_feature = elevation_expanded.unsqueeze(2)  # (time, locations, 1)
                        temporal_tensors.append(elevation_feature)

            # Process time series features
            time_series_features = []
            for key in self.time_series_keys:
                if key in self._seq_data:
                    # Get tensor with shape (time, locations)
                    ts_tensor = self._get_tensor(key)

                    # Add feature dimension
                    ts_feature = ts_tensor.unsqueeze(2)  # (time, locations, 1)
                    time_series_features.append(ts_feature)

            # Combine time series features
            if time_series_features:
                combined_ts = torch.cat(time_series_features, dim=2)  # (time, locations, n_features)
                temporal_tensors.append(combined_ts)

            # Combine all temporal features
            if temporal_tensors:
                # For now, just use the first tensor
                self.temporal_features = temporal_tensors[0]
            else:
                self.temporal_features = None
                return None

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
        if self.static_features is None:
            coordinates_field = self.field_names['coordinates_field']
            if coordinates_field in self._seq_data:
                # Get coordinates tensor
                coords = self._get_tensor(coordinates_field)
                self.static_features = coords
            elif self.static_feature_keys:
                # Use another static feature if coordinates not available
                key = self.static_feature_keys[0]
                self.static_features = self._get_tensor(key)
            else:
                # No static features available
                self.static_features = torch.zeros((self.size, 1), dtype=self.dtype)

            # Free tensors if we're in strict memory saving mode
            if self.lazy_loading == "strict":
                for key in self.static_feature_keys:
                    if key in self.tensors and key != coordinates_field:
                        del self.tensors[key]
                        self.tensors[key] = None
                gc.collect()

        if idx is not None:
            return self.static_features[idx]
        return self.static_features

    def _get_targets(self, idx=None):
        """Get or create targets tensor"""
        if self.targets_tensor is None:
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
            result = {
                'static_features': self.static_features[loc_idx] if self.static_features is not None else None,
                'target': self.targets_tensor[time_idx, loc_idx] if len(self.targets_tensor.shape) > 1 else self.targets_tensor[loc_idx]
            }

            # Add temporal features (with lookback window if applicable)
            if self.temporal_features is not None:
                if self.lookback > 0:
                    # Get sequence window
                    if len(self.temporal_features.shape) == 3:  # (time, locations, features)
                        result['temporal_features'] = self.temporal_features[window_start:time_idx, loc_idx, :]
                    else:  # Time features (time, features)
                        result['temporal_features'] = self.temporal_features[window_start:time_idx, :]
                else:
                    # Get single timestep
                    if len(self.temporal_features.shape) == 3:  # (time, locations, features)
                        result['temporal_features'] = self.temporal_features[time_idx, loc_idx, :]
                    else:  # Time features (time, features)
                        result['temporal_features'] = self.temporal_features[time_idx, :]

            # Add time series features (all shape time, locations)
            for key in self.time_series_keys:
                if key in self.tensors:
                    if self.lookback > 0:
                        # Get sequence window
                        result[key] = self.tensors[key][window_start:time_idx, loc_idx]
                    else:
                        # Get single timestep
                        result[key] = self.tensors[key][time_idx, loc_idx]

            # Add nighttime field
            if 'nighttime_mask' in self.tensors:
                if self.lookback > 0:
                    result['nighttime'] = self.tensors['nighttime_mask'][window_start:time_idx, loc_idx]
                else:
                    result['nighttime'] = self.tensors['nighttime_mask'][time_idx, loc_idx]
            elif 'ghi' in self.tensors:
                # Infer from GHI value (GHI=0 means nighttime)
                if self.lookback > 0:
                    ghi_values = self.tensors['ghi'][window_start:time_idx, loc_idx]
                    result['nighttime'] = (ghi_values == 0).float()
                else:
                    ghi_value = self.tensors['ghi'][time_idx, loc_idx]
                    result['nighttime'] = torch.tensor(float(ghi_value == 0), dtype=self.dtype)

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

            # Add time features
            time_field = self.field_names['time_field']
            if time_field in self._seq_data:
                if self.lookback > 0:
                    # Get sequence window
                    time_features = torch.tensor(
                        self._seq_data[time_field][window_start:time_idx],
                        dtype=self.dtype
                    )
                else:
                    # Get single timestep
                    time_features = torch.tensor(
                        self._seq_data[time_field][time_idx],
                        dtype=self.dtype
                    )
                result['temporal_features'] = time_features

            # Add time series features (all with shape time, locations)
            for key in self.time_series_keys:
                if key in self._seq_data:
                    if self.lookback > 0:
                        # Get sequence window for this location
                        feature_val = self._seq_data[key][window_start:time_idx, loc_idx]
                    else:
                        # Get single timestep for this location
                        feature_val = self._seq_data[key][time_idx, loc_idx]

                    result[key] = torch.tensor(feature_val, dtype=self.dtype)

            # Add nighttime field
            if 'nighttime_mask' in self._seq_data:
                if self.lookback > 0:
                    nighttime = self._seq_data['nighttime_mask'][window_start:time_idx, loc_idx]
                else:
                    nighttime = self._seq_data['nighttime_mask'][time_idx, loc_idx]
                result['nighttime'] = torch.tensor(nighttime, dtype=self.dtype)
            elif 'ghi' in self._seq_data:
                # Infer from GHI value (GHI=0 means nighttime)
                if self.lookback > 0:
                    ghi_values = self._seq_data['ghi'][window_start:time_idx, loc_idx]
                    result['nighttime'] = torch.tensor(ghi_values == 0, dtype=self.dtype)
                else:
                    ghi_value = self._seq_data['ghi'][time_idx, loc_idx]
                    result['nighttime'] = torch.tensor(float(ghi_value == 0), dtype=self.dtype)

            return result

    def preload_all(self):
        """Explicitly load all tensors into memory at once"""
        if self.lazy_loading:
            print("Preloading all tensors into memory...")

            # Load temporal and static features
            self._get_temporal_features()
            self._get_static_features()
            self._get_targets()

            # Load all remaining tensors
            for key in self._seq_data.keys():
                if key not in self.tensors or self.tensors[key] is None:
                    self._get_other_tensor(key)

            # Free original data to save memory
            if self.lazy_loading != "strict":
                del self._seq_data
                del self._targets
                self._seq_data = None
                self._targets = None
                gc.collect()
                self.lazy_loading = False
                print("All tensors loaded, original data cleared")

    @classmethod
    def from_file(cls, filepath, lazy_loading=False, dtype=torch.float32, field_names=None, lookback=24, target_field='ghi'):
        """
        Create dataset directly from a saved file

        Alias for the constructor with preprocessed_data_path parameter

        Args:
            filepath: Path to the saved file
            lazy_loading: If True, tensors are only created when accessed
            dtype: Tensor data type
            field_names: Dictionary mapping standard field types to custom field names
            lookback: Number of timesteps to look back
            target_field: Name of the field to use as target (default: 'ghi')

        Returns:
            dataset: TimeSeriesDataset instance
        """
        return cls(
            preprocessed_data_path=filepath,
            lazy_loading=lazy_loading,
            dtype=dtype,
            field_names=field_names,
            lookback=lookback,
            target_field=target_field
        )
