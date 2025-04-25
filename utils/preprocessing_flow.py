"""
Preprocessing flow utilities that incorporate sequence saving and loading
"""

import os
import numpy as np
from utils.normalize_utils import create_sequences
from utils.timeseriesdataset import TimeSeriesDataset
from utils.data_persistence import save_sequences, load_sequences, get_latest_sequence_file
from torch.utils.data import DataLoader


def process_data_with_cache(norm_data, lookback=24, selected_features=None, target_variable=None,
                           output_dir='cached_sequences', prefix='', force_regenerate=False):
    """
    Process data through create_sequences with caching support

    Args:
        norm_data: Dictionary of normalized data
        lookback: Number of timesteps to look back
        selected_features: List of selected features
        target_variable: Target variable name (e.g., 'ghi')
        output_dir: Directory to save cached sequences
        prefix: Prefix for the cached file (e.g., 'train', 'val', 'test')
        force_regenerate: If True, regenerate sequences even if cached data exists

    Returns:
        seq_data: Dictionary of sequence arrays
        seq_targets: Target values
    """
    # Check if we have cached data
    if not force_regenerate:
        latest_file = get_latest_sequence_file(output_dir, prefix)
        if latest_file:
            print(f"Found cached sequence data: {latest_file}")
            seq_data, seq_targets, metadata = load_sequences(latest_file)
            return seq_data, seq_targets

    # No cached data or forced regeneration, create sequences
    print(f"Creating new sequences with lookback={lookback}...")
    seq_data, seq_targets = create_sequences(
        norm_data,
        lookback=lookback,
        selected_features=selected_features,
        target_variable=target_variable
    )

    # Save the sequences
    metadata = {
        'lookback': lookback,
        'selected_features': selected_features,
        'target_variable': target_variable,
    }
    save_sequences(seq_data, seq_targets, output_dir, prefix, metadata)

    return seq_data, seq_targets


def create_datasets_and_loaders(train_seq=None, train_targets=None,
                               val_seq=None, val_targets=None,
                               test_seq=None, test_targets=None,
                               train_cache_path=None, val_cache_path=None, test_cache_path=None,
                               batch_size=64):
    """
    Create PyTorch datasets and data loaders, loading from cache if paths are provided

    Args:
        train_seq, val_seq, test_seq: Sequence data dictionaries
        train_targets, val_targets, test_targets: Target arrays
        train_cache_path, val_cache_path, test_cache_path: Paths to cached sequence files
        batch_size: Batch size for data loaders

    Returns:
        train_loader, val_loader, test_loader: PyTorch DataLoader objects
    """
    # Load from cache if paths are provided
    if train_cache_path and (train_seq is None or train_targets is None):
        train_seq, train_targets, _ = load_sequences(train_cache_path)

    if val_cache_path and (val_seq is None or val_targets is None):
        val_seq, val_targets, _ = load_sequences(val_cache_path)

    if test_cache_path and (test_seq is None or test_targets is None):
        test_seq, test_targets, _ = load_sequences(test_cache_path)

    # Create datasets
    train_dataset = TimeSeriesDataset(train_seq, train_targets) if train_seq is not None else None
    val_dataset = TimeSeriesDataset(val_seq, val_targets) if val_seq is not None else None
    test_dataset = TimeSeriesDataset(test_seq, test_targets) if test_seq is not None else None

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) if train_dataset else None
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0) if test_dataset else None

    return train_loader, val_loader, test_loader


def full_preprocessing_flow(norm_train_data, norm_val_data, norm_test_data,
                           lookback=24, selected_features=None, target_variable=None,
                           cache_dir='cached_sequences', batch_size=64, force_regenerate=False):
    """
    Full preprocessing flow with sequence caching

    Args:
        norm_train_data, norm_val_data, norm_test_data: Normalized data dictionaries
        lookback: Number of timesteps to look back
        selected_features: List of selected features
        target_variable: Target variable name (e.g., 'ghi')
        cache_dir: Directory to save cached sequences
        batch_size: Batch size for data loaders
        force_regenerate: If True, regenerate sequences even if cached data exists

    Returns:
        train_loader, val_loader, test_loader: PyTorch DataLoader objects
        train_seq_path, val_seq_path, test_seq_path: Paths to the cached sequence files
    """
    # Create sequences with caching
    train_seq, train_targets = process_data_with_cache(
        norm_train_data, lookback, selected_features, target_variable,
        cache_dir, 'train', force_regenerate
    )

    val_seq, val_targets = process_data_with_cache(
        norm_val_data, lookback, selected_features, target_variable,
        cache_dir, 'val', force_regenerate
    )

    test_seq, test_targets = process_data_with_cache(
        norm_test_data, lookback, selected_features, target_variable,
        cache_dir, 'test', force_regenerate
    )

    # Get paths to the cached files
    train_seq_path = get_latest_sequence_file(cache_dir, 'train')
    val_seq_path = get_latest_sequence_file(cache_dir, 'val')
    test_seq_path = get_latest_sequence_file(cache_dir, 'test')

    # Create datasets and loaders
    train_loader, val_loader, test_loader = create_datasets_and_loaders(
        train_seq, train_targets, val_seq, val_targets, test_seq, test_targets,
        batch_size=batch_size
    )

    return train_loader, val_loader, test_loader, train_seq_path, val_seq_path, test_seq_path
