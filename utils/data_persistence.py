"""
Utilities for saving and loading intermediate processing data
"""

import os
import numpy as np
import h5py
import json
import pickle
from datetime import datetime


def save_sequences(sequences, targets, output_dir, prefix='', add_timestamp=True, metadata=None):
    """
    [DEPRECATED]
    Save sequence data to disk for later reuse

    Args:
        sequences: Dictionary of sequence arrays as returned by create_sequences
        targets: Target values array
        output_dir: Directory to save the files
        prefix: Optional prefix for the filename (e.g., 'train', 'val', 'test')
        add_timestamp: Whether to add a timestamp to the filename
        metadata: Optional dictionary with metadata about the sequences

    Returns:
        filepath: Path to the saved file
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp for the filename
    if add_timestamp:
        f_timestamp = datetime.now().strftime("_%Y%m%d_%H%M%S")
    else:
        f_timestamp = ""

    # Create filename
    if prefix:
        filename = f"{prefix}_sequences{f_timestamp}.h5"
    else:
        filename = f"sequences{f_timestamp}.h5"

    filepath = os.path.join(output_dir, filename)

    if metadata is None:
        metadata = {}
    # Add created time to metadata
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata['created_time'] = timestamp

    # Save arrays to HDF5 file
    with h5py.File(filepath, 'w') as f:
        # Create a sequences group
        seq_group = f.create_group('sequences')

        # Save each sequence array
        for key, value in sequences.items():
            seq_group.create_dataset(key, data=value, compression='gzip')

        # Save targets
        f.create_dataset('targets', data=targets, compression='gzip')

        # Save metadata if provided
        if metadata:
            # Convert any non-serializable objects to strings
            meta_json = json.dumps(metadata, default=str)
            f.attrs['metadata'] = meta_json

        # Save timestamp
        f.attrs['created'] = timestamp

    print(f"Saved sequences to {filepath}")
    return filepath


def save_normalized_data(data_dict, output_dir, prefix='', add_timestamp=True, metadata=None):
    """
    Save normalized data to disk for direct loading with TimeSeriesDataset

    Target field should be included in data_dict and will be handled by TimeSeriesDataset

    Args:
        data_dict: Dictionary of normalized data arrays (including target variable)
        output_dir: Directory to save the files
        prefix: Optional prefix for the filename (e.g., 'train', 'val', 'test')
        add_timestamp: Whether to add a timestamp to the filename
        metadata: Optional dictionary with metadata about the data

    Returns:
        filepath: Path to the saved file
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp for the filename
    if add_timestamp:
        f_timestamp = datetime.now().strftime("_%Y%m%d_%H%M%S")
    else:
        f_timestamp = ""

    # Create filename
    if prefix:
        filename = f"{prefix}_normalized{f_timestamp}.h5"
    else:
        filename = f"normalized{f_timestamp}.h5"

    filepath = os.path.join(output_dir, filename)

    if metadata is None:
        metadata = {}
    # Add created time to metadata
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata['created_time'] = timestamp

    # Save arrays to HDF5 file
    with h5py.File(filepath, 'w') as f:
        # Save each data array directly (no group needed)
        for key, value in data_dict.items():
            f.create_dataset(key, data=value, compression='gzip')

        # Save metadata if provided
        if metadata:
            # Convert any non-serializable objects to strings
            meta_json = json.dumps(metadata, default=str)
            f.attrs['metadata'] = meta_json

        # Save timestamp
        f.attrs['created'] = timestamp

    print(f"Saved normalized data to {filepath}")
    return filepath


def load_sequences(filepath):
    """
    [DEPRECATED]
    Load sequence data from disk

    Args:
        filepath: Path to the saved file

    Returns:
        sequences: Dictionary of sequence arrays
        targets: Target values array
        metadata: Metadata dictionary if it exists, otherwise None
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Sequence file not found at {filepath}")

    # Load data from HDF5 file
    with h5py.File(filepath, 'r') as f:
        # Load sequences
        sequences = {}
        seq_group = f['sequences']
        for key in seq_group.keys():
            sequences[key] = np.array(seq_group[key])

        # Load targets
        targets = np.array(f['targets'])

        # Load metadata if it exists
        metadata = None
        if 'metadata' in f.attrs:
            try:
                metadata = json.loads(f.attrs['metadata'])
            except json.JSONDecodeError:
                print("Warning: Could not parse metadata JSON")

    print(f"Loaded sequences from {filepath}")
    return sequences, targets, metadata


def load_normalized_data(filepath):
    """
    Load normalized data from disk

    All data is loaded into a single dictionary, including target variable

    Args:
        filepath: Path to the saved file

    Returns:
        data_dict: Dictionary of normalized data arrays
        metadata: Metadata dictionary if it exists, otherwise None
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Normalized data file not found at {filepath}")

    # Load data from HDF5 file
    with h5py.File(filepath, 'r') as f:
        # Load all arrays directly
        data_dict = {}
        for key in f.keys():
            data_dict[key] = np.array(f[key])

        # Load metadata if it exists
        metadata = None
        if 'metadata' in f.attrs:
            try:
                metadata = json.loads(f.attrs['metadata'])
            except json.JSONDecodeError:
                print("Warning: Could not parse metadata JSON")

    print(f"Loaded normalized data from {filepath}")
    return data_dict, metadata


def list_sequence_files(directory, prefix=None):
    """
    List all sequence files in a directory

    Args:
        directory: Directory to search
        prefix: Optional prefix to filter by (e.g., 'train', 'val', 'test')

    Returns:
        files: List of sequence files with full paths
    """
    if not os.path.exists(directory):
        return []

    files = []
    for filename in os.listdir(directory):
        if filename.endswith('.h5'):
            if prefix is None or filename.startswith(prefix):
                files.append(os.path.join(directory, filename))

    return sorted(files)


def get_latest_sequence_file(directory, prefix=None):
    """
    Get the most recent sequence file in a directory

    Args:
        directory: Directory to search
        prefix: Optional prefix to filter by (e.g., 'train', 'val', 'test')

    Returns:
        filepath: Path to the most recent file, or None if no files found
    """
    files = list_sequence_files(directory, prefix)

    if not files:
        return None

    # Sort by file modification time (most recent first)
    files.sort(key=os.path.getmtime, reverse=True)

    return files[0]


def list_normalized_data_files(directory, prefix=None):
    """
    List all normalized data files in a directory

    Args:
        directory: Directory to search
        prefix: Optional prefix to filter by (e.g., 'train', 'val', 'test')

    Returns:
        files: List of normalized data files with full paths
    """
    if not os.path.exists(directory):
        return []

    files = []
    for filename in os.listdir(directory):
        if filename.endswith('.h5') and 'normalized' in filename:
            if prefix is None or filename.startswith(prefix):
                files.append(os.path.join(directory, filename))

    return sorted(files)


def get_latest_normalized_data_file(directory, prefix=None):
    """
    Get the most recent normalized data file in a directory

    Args:
        directory: Directory to search
        prefix: Optional prefix to filter by (e.g., 'train', 'val', 'test')

    Returns:
        filepath: Path to the most recent file, or None if no files found
    """
    files = list_normalized_data_files(directory, prefix)

    if not files:
        return None

    # Sort by file modification time (most recent first)
    files.sort(key=os.path.getmtime, reverse=True)

    return files[0]


def save_scalers(scalers, filepath, overwrite=False):
    """
    Save a dictionary of scalers to a file for later use in inference.

    Args:
        scalers: Dictionary of scalers output from normalize_data
        filepath: Path where scalers will be saved
        overwrite: Whether to overwrite existing file

    Returns:
        bool: True if successful
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    if os.path.exists(filepath) and not overwrite:
        raise FileExistsError(f"File {filepath} already exists. Set overwrite=True to overwrite.")

    # Check if we have any scalers
    if not scalers:
        print("Warning: No scalers found.")
        return False

    # Save scalers to file
    with open(filepath, 'wb') as f:
        pickle.dump(scalers, f)

    print(f"Saved {len(scalers)} scalers to {filepath}")
    return True


def load_scalers(filepath):
    """
    Load scalers from a file.

    Args:
        filepath: Path to the scalers file

    Returns:
        dict: Dictionary of scalers
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Scaler file {filepath} not found")

    with open(filepath, 'rb') as f:
        scalers = pickle.load(f)

    print(f"Loaded {len(scalers)} scalers from {filepath}")
    return scalers


def list_scaler_files(directory, prefix=None):
    """
    List all scaler files in a directory

    Args:
        directory: Directory to search
        prefix: Optional prefix to filter by (e.g., 'train', 'model')

    Returns:
        files: List of scaler files with full paths
    """
    if not os.path.exists(directory):
        return []

    files = []
    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            if prefix is None or filename.startswith(prefix):
                files.append(os.path.join(directory, filename))

    return sorted(files)


def get_latest_scaler_file(directory, prefix=None):
    """
    Get the most recent scaler file in a directory

    Args:
        directory: Directory to search
        prefix: Optional prefix to filter by (e.g., 'train', 'model')

    Returns:
        filepath: Path to the most recent file, or None if no files found
    """
    files = list_scaler_files(directory, prefix)

    if not files:
        return None

    # Sort by file modification time (most recent first)
    files.sort(key=os.path.getmtime, reverse=True)

    return files[0]
