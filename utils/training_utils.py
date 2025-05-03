import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

from utils.wandb_utils import track_experiment, is_wandb_enabled

# Device configuration
default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@track_experiment
def train_model(
    model,
    train_loader,
    val_loader,
    model_name="Model",
    epochs=50,
    patience=10,
    lr=0.001,
    debug_mode=False,
    target_scaler=None,
    config=None,
    device=default_device
):
    """
    Train a model and validate it

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        model_name: Name of the model for logging
        epochs: Maximum number of epochs
        patience: Early stopping patience
        lr: Learning rate
        debug_mode: Whether to run as debug mode (only run 10 batches per epoch)
        target_scaler: Scaler for the target variable (required for evaluate_model)
        config: Configuration dictionary (optional)

    Returns:
        history: Dictionary of training history
        best_model: Model with best validation performance
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    if config is not None:
        print(f"Training config: {config}")

    history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': [], 'train_samples': [], 'val_samples': []}
    best_val_loss = float('inf')
    counter = 0
    best_model_state = None

    if debug_mode:
        print("Debug mode is enabled. Only running 10 batches per epoch.")

    # Instead of wrapping epochs in tqdm, we'll manually print epoch progress
    for epoch in range(epochs):
        # Training phase
        model.train()
        all_train_outputs = []
        all_train_targets = []
        debug_counter = 0
        train_samples = 0  # Initialize sample counter for training

        # Use tqdm for batch-level progress
        train_loop = tqdm(train_loader, desc=f"Training {model_name}", leave=False)
        for batch in train_loop:
            # Check for required fields
            if 'temporal_features' not in batch or 'static_features' not in batch or 'target' not in batch:
                raise ValueError("Batch missing required fields: 'temporal_features', 'static_features', or 'target'")

            temporal_features = batch['temporal_features'].to(device)
            static_features = batch['static_features'].to(device)
            target = batch['target'].to(device)
            batch_size = temporal_features.size(0)
            train_samples += batch_size  # Count samples

            # Ensure target has the right shape for broadcasting
            if len(target.shape) == 1 and target.shape[0] > 1:
                # If target is [batch_size], reshape to [batch_size, 1]
                target = target.view(-1, 1)

            optimizer.zero_grad()
            output = model(temporal_features, static_features)

            # Ensure shapes match for loss calculation
            if output.shape != target.shape:
                if len(output.shape) > len(target.shape):
                    # If output has more dimensions than target, reshape target
                    target = target.view(*output.shape)
                else:
                    # If target has more dimensions, reshape output
                    output = output.view(*target.shape)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Calculate batch metrics for progress display only
            batch_loss = loss.item()
            batch_mae = F.l1_loss(output, target, reduction='mean').item()

            # Store outputs and targets for later computation
            all_train_outputs.append(output.detach().cpu().numpy())
            all_train_targets.append(target.detach().cpu().numpy())

            # Update the progress bar with current batch metrics
            train_loop.set_postfix(loss=batch_loss, mae=batch_mae)

            debug_counter += 1
            if debug_mode and debug_counter > 10:
                break

        # Compute training metrics using sklearn (same as validation)
        all_train_outputs = np.vstack(all_train_outputs)
        all_train_targets = np.vstack(all_train_targets)

        # Calculate metrics using the same functions as in evaluate_model
        if target_scaler is not None:
            # Apply inverse transform to get predictions in original scale
            all_train_outputs_orig = target_scaler.inverse_transform(all_train_outputs)
            all_train_targets_orig = target_scaler.inverse_transform(all_train_targets)
            train_loss = mean_squared_error(all_train_targets_orig, all_train_outputs_orig)
            train_mae = mean_absolute_error(all_train_targets_orig, all_train_outputs_orig)
        else:
            train_loss = mean_squared_error(all_train_targets, all_train_outputs)
            train_mae = mean_absolute_error(all_train_targets, all_train_outputs)

        # Validation phase - using evaluate_model
        # Note: evaluate_model handles model.eval() and torch.no_grad() internally
        val_metrics = evaluate_model(
            model=model,
            data_loader=val_loader,
            target_scaler=target_scaler,
            model_name=f"Validation {model_name} (Epoch {epoch+1})",
            log_to_wandb=False,  # We'll handle wandb logging separately for training
            debug_mode=debug_mode
        )

        # Extract metrics needed for training loop
        val_loss = val_metrics['mse']  # MSE is equivalent to the criterion we use (nn.MSELoss)
        val_mae = val_metrics['mae']
        val_samples = val_metrics['total_samples']  # Get number of validation samples

        # Update learning rate
        scheduler.step(val_loss)

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        history['train_samples'].append(train_samples)
        history['val_samples'].append(val_samples)

        # Log metrics to wandb
        if is_wandb_enabled():
            wandb.log({
                'train/epoch': epoch,
                'train/loss': train_loss,
                'train/mae': train_mae,
                'train/learning_rate': optimizer.param_groups[0]['lr'],
                'val/epoch': epoch,
                'val/loss': val_loss,
                'val/mae': val_mae,
            })
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            best_model_state = model.state_dict().copy()

        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    print("Loading best model from training session. The model object now can be used to make predictions.")
    model.load_state_dict(best_model_state)
    return history


def evaluate_model(model, data_loader, target_scaler, model_name="", log_to_wandb=True, device=default_device, debug_mode=False):
    """
    Evaluate a model on a dataset and compute metrics.

    Args:
        model: PyTorch model
        data_loader: Data loader for evaluation
        target_scaler: Scaler for the target variable
        model_name: Name of the model for logging
        log_to_wandb: Whether to log to wandb
        device: Device to run the evaluation on
        debug_mode: Whether to run in debug mode (only run 10 batches)

    Returns:
        metrics: Dictionary of evaluation metrics including inference speed
    """
    model.eval()
    all_outputs = []
    all_targets = []
    all_nighttime = []
    has_nighttime_data = False

    # Track inference time
    total_inference_time = 0
    total_samples = 0

    if debug_mode:
        print("Debug mode is enabled for evaluation. Only running 10 batches.")

    with torch.no_grad():
        debug_counter = 0
        # Add tqdm progress bar
        eval_loop = tqdm(data_loader, desc=f"Evaluating {model_name}", leave=False)
        for batch in eval_loop:
            # Check for required fields
            if 'temporal_features' not in batch or 'static_features' not in batch or 'target' not in batch:
                raise ValueError("Batch missing required fields: 'temporal_features', 'static_features', or 'target'")

            temporal_features = batch['temporal_features'].to(device)
            static_features = batch['static_features'].to(device)
            target = batch['target'].to(device)
            batch_size = temporal_features.size(0)
            total_samples += batch_size

            # Ensure target has the right shape for broadcasting
            if len(target.shape) == 1 and target.shape[0] > 1:
                target = target.view(-1, 1)

            # Check if we have nighttime data
            if 'nighttime_mask' in batch:
                has_nighttime_data = True
                nighttime = batch['nighttime_mask']
                # Ensure nighttime has same shape as target
                if nighttime.shape != target.cpu().shape:
                    # Handle different shapes - if nighttime is a sequence, take the last value
                    # or if it's a different size, reshape it appropriately
                    if len(nighttime.shape) > 1 and nighttime.shape[0] == target.shape[0]:
                        # If nighttime has a sequence dimension but same batch size
                        if len(nighttime.shape) == 3:  # [batch, seq, 1]
                            # Take the last timestep from each sequence
                            nighttime = nighttime[:, -1, :]
                        elif len(nighttime.shape) == 2 and nighttime.shape[1] > target.shape[1]:
                            # Take just what we need if it's wider
                            nighttime = nighttime[:, :target.shape[1]]
                    # Now try to reshape to match target
                    try:
                        nighttime = nighttime.view(*target.cpu().shape)
                    except RuntimeError:
                        # If reshaping fails, create a new tensor matching target's shape
                        # For binary nighttime indicator, use the mode (most common value)
                        if len(nighttime.shape) == 1:
                            # Single value per sample case
                            nighttime = nighttime.view(-1, 1)
                        else:
                            # For sequence data, try to use the most common value
                            # or the last value as fallback
                            print(f"Warning: Nighttime shape {nighttime.shape} incompatible with target shape {target.shape}")
                            nighttime = torch.zeros_like(target.cpu())
            else:
                print("No nighttime data found in batch")
                # Create a placeholder (all zeros) for nighttime
                nighttime = torch.zeros_like(target).cpu()

            # Time the inference
            start_time = time.time()
            output = model(temporal_features, static_features)
            end_time = time.time()

            # Calculate inference time for this batch
            batch_inference_time = end_time - start_time
            total_inference_time += batch_inference_time

            # Ensure shapes match for comparisons
            if output.shape != target.shape:
                if len(output.shape) > len(target.shape):
                    # If output has more dimensions than target, reshape target
                    target = target.view(*output.shape)
                else:
                    # If target has more dimensions, reshape output
                    output = output.view(*target.shape)

            # Store results
            all_outputs.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_nighttime.append(nighttime.cpu().numpy())

            # Update the progress bar with batch size and inference time
            eval_loop.set_postfix(samples=batch_size, time_per_batch=f"{batch_inference_time:.4f}s")

            debug_counter += 1
            if debug_mode and debug_counter > 10:
                break

    # Calculate inference speed metrics
    if total_samples > 0:
        avg_time_per_sample = total_inference_time / total_samples
        samples_per_second = total_samples / total_inference_time
    else:
        avg_time_per_sample = float('nan')
        samples_per_second = float('nan')

    # Concatenate batches
    all_outputs = np.vstack(all_outputs)
    all_targets = np.vstack(all_targets)
    all_nighttime = np.vstack(all_nighttime)

    # Inverse transform to original scale
    y_pred_orig = target_scaler.inverse_transform(all_outputs)
    y_true_orig = target_scaler.inverse_transform(all_targets)

    # Calculate overall metrics
    mse = mean_squared_error(y_true_orig, y_pred_orig)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    r2 = r2_score(y_true_orig, y_pred_orig)

    # Calculate MASE (Mean Absolute Scaled Error) with 24-hour seasonal naive forecast
    # For seasonal data, the naive forecast is the value from the same time in the previous season (24 hours ago)
    # MASE = MAE / MAE_naive where MAE_naive is the MAE of the naive forecast

    # For simplicity, we'll use a naive approach to calculate MASE assuming data is ordered
    # Create a naive forecast that's the value 24 steps ago (one day for hourly data)
    n_samples = len(y_true_orig)
    naive_forecast = np.roll(y_true_orig, 24)  # Shift by 24 hours

    # The first 24 values don't have valid naive forecasts, exclude them from calculation
    if n_samples > 24:
        # Calculate MAE of the naive forecast (excluding the first 24 samples)
        mae_naive = mean_absolute_error(y_true_orig[24:], naive_forecast[24:])
        # Calculate MASE (avoid division by zero)
        mase = mae / mae_naive if mae_naive > 0 else float('nan')
    else:
        mase = float('nan')

    # Calculate daytime/nighttime metrics if we have nighttime data
    if has_nighttime_data:
        night_mask = all_nighttime.flatten() > 0.5
        day_mask = ~night_mask
    else:
        # If no nighttime data, treat all as daytime
        day_mask = np.ones(len(y_true_orig), dtype=bool)
        night_mask = np.zeros(len(y_true_orig), dtype=bool)

    if np.sum(day_mask) > 0:
        day_mse = mean_squared_error(y_true_orig[day_mask], y_pred_orig[day_mask])
        day_rmse = np.sqrt(day_mse)
        day_mae = mean_absolute_error(y_true_orig[day_mask], y_pred_orig[day_mask])
        day_r2 = r2_score(y_true_orig[day_mask], y_pred_orig[day_mask])

        # Calculate daytime MASE
        day_mask_array = np.where(day_mask)[0]
        if len(day_mask_array) > 24:
            # Create a mask for day samples that have valid naive forecasts (24 hours ago was also day)
            valid_day_indices = day_mask_array[np.isin(day_mask_array - 24, day_mask_array)]
            if len(valid_day_indices) > 0:
                # Get actual values and naive forecast for these indices
                actual_day_values = y_true_orig[valid_day_indices]
                naive_day_forecast = y_true_orig[valid_day_indices - 24]
                day_mae_naive = mean_absolute_error(actual_day_values, naive_day_forecast)
                day_mase = day_mae / day_mae_naive if day_mae_naive > 0 else float('nan')
            else:
                day_mase = float('nan')
        else:
            day_mase = float('nan')
    else:
        day_mse = day_rmse = day_mae = day_r2 = day_mase = float('nan')

    if np.sum(night_mask) > 0:
        night_mse = mean_squared_error(y_true_orig[night_mask], y_pred_orig[night_mask])
        night_rmse = np.sqrt(night_mse)
        night_mae = mean_absolute_error(y_true_orig[night_mask], y_pred_orig[night_mask])
        night_r2 = r2_score(y_true_orig[night_mask], y_pred_orig[night_mask]) if np.unique(y_true_orig[night_mask]).size > 1 else float('nan')

        # Calculate nighttime MASE
        night_mask_array = np.where(night_mask)[0]
        if len(night_mask_array) > 24:
            # Create a mask for night samples that have valid naive forecasts (24 hours ago was also night)
            valid_night_indices = night_mask_array[np.isin(night_mask_array - 24, night_mask_array)]
            if len(valid_night_indices) > 0:
                # Get actual values and naive forecast for these indices
                actual_night_values = y_true_orig[valid_night_indices]
                naive_night_forecast = y_true_orig[valid_night_indices - 24]
                night_mae_naive = mean_absolute_error(actual_night_values, naive_night_forecast)
                night_mase = night_mae / night_mae_naive if night_mae_naive > 0 else float('nan')
            else:
                night_mase = float('nan')
        else:
            night_mase = float('nan')
    else:
        night_mse = night_rmse = night_mae = night_r2 = night_mase = float('nan')

    # Create evaluation metrics dictionary
    metrics = {
        'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'mase': mase,
        'day_mse': day_mse, 'day_rmse': day_rmse, 'day_mae': day_mae, 'day_r2': day_r2, 'day_mase': day_mase,
        'night_mse': night_mse, 'night_rmse': night_rmse, 'night_mae': night_mae, 'night_r2': night_r2, 'night_mase': night_mase,
        'y_pred': y_pred_orig, 'y_true': y_true_orig, 'nighttime_mask': all_nighttime,
        'total_inference_time': total_inference_time,
        'total_samples': total_samples,
        'avg_time_per_sample': avg_time_per_sample,
        'samples_per_second': samples_per_second
    }

    # Print metrics
    print(f"\n{model_name} Evaluation Metrics:")
    print(f"  Overall:  MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}, MASE: {mase:.2f}")
    print(f"  Daytime:  MSE: {day_mse:.2f}, RMSE: {day_rmse:.2f}, MAE: {day_mae:.2f}, R²: {day_r2:.4f}, MASE: {day_mase:.2f}")

    if has_nighttime_data:
        # Fix the f-string formatting - move conditional outside format specifier
        r2_str = f"{night_r2:.4f}" if not np.isnan(night_r2) else "N/A"
        mase_str = f"{night_mase:.2f}" if not np.isnan(night_mase) else "N/A"
        print(f"  Nighttime: MSE: {night_mse:.2f}, RMSE: {night_rmse:.2f}, MAE: {night_mae:.2f}, R²: {r2_str}, MASE: {mase_str}")
    else:
        print("  Nighttime metrics: Not available (no nighttime data)")

    # Print inference speed metrics
    print(f"  Inference Speed: {samples_per_second:.2f} samples/sec, {avg_time_per_sample*1000:.4f} ms/sample")
    print(f"  Total time: {total_inference_time:.4f} sec for {total_samples} samples")

    # Log to wandb if enabled
    if log_to_wandb and is_wandb_enabled():
        # Create a metrics table instead of logging as timeseries
        eval_prefix = 'val/' if 'Validation' in model_name else 'test/' if 'Test' in model_name else ''

        # Create a table with metrics
        metrics_table = wandb.Table(
            columns=["Metric", "Overall", "Daytime", "Nighttime"],
            data=[
                ["MSE", float(mse), float(day_mse), float(night_mse)],
                ["RMSE", float(rmse), float(day_rmse), float(night_rmse)],
                ["MAE", float(mae), float(day_mae), float(night_mae)],
                ["MASE", float(mase), float(day_mase), float(night_mase)],
                ["R²", float(r2), float(day_r2), float(night_r2)],
            ]
        )

        # Create inference speed metrics table
        speed_table = wandb.Table(
            columns=["Metric", "Value"],
            data=[
                ["Total Inference Time (s)", float(total_inference_time)],
                ["Total Samples", int(total_samples)],
                ["Avg Time per Sample (ms)", float(avg_time_per_sample * 1000)],
                ["Samples per Second", float(samples_per_second)],
            ]
        )

        # Create a summary dictionary for key metrics
        summary_metrics = {
            f"{eval_prefix}mse": mse,
            f"{eval_prefix}rmse": rmse,
            f"{eval_prefix}mae": mae,
            f"{eval_prefix}mase": mase,
            f"{eval_prefix}r2": r2,
            f"{eval_prefix}inference_speed_samples_per_sec": samples_per_second,
            f"{eval_prefix}inference_time_ms_per_sample": avg_time_per_sample * 1000
        }

        # Create a sample predictions table
        sample_size = min(100, len(y_true_orig))
        indices = np.random.choice(len(y_true_orig), sample_size, replace=False)

        # Create the table with predictions data
        pred_table = wandb.Table(
            columns=["True GHI", "Predicted GHI", "Residual", "Is Nighttime", "Error %"]
        )

        for i in indices:
            true_val = float(y_true_orig[i][0])
            pred_val = float(y_pred_orig[i][0])
            residual = true_val - pred_val
            is_night = bool(all_nighttime[i][0] > 0.5)

            # Calculate error percentage, handling division by zero
            if abs(true_val) > 1e-6:  # Avoid division by near-zero
                error_pct = (residual / true_val) * 100
            else:
                error_pct = float('nan')

            pred_table.add_data(
                true_val,
                pred_val,
                residual,
                is_night,
                error_pct
            )

        # Log both the tables and the summary metrics
        wandb.log({
            f"{eval_prefix}metrics_table": metrics_table,
            f"{eval_prefix}inference_speed_table": speed_table,
            f"{eval_prefix}predictions_sample": pred_table,
            **summary_metrics
        })

    return metrics
