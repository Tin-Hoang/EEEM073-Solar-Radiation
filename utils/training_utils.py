import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

from utils.wandb_utils import track_experiment, is_wandb_enabled
from utils.model_utils import get_model_summary

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@track_experiment
def train_model(model, train_loader, val_loader, model_name="Model", epochs=50, patience=10, lr=0.001, debug_mode=False):
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

    Returns:
        history: Dictionary of training history
        best_model: Model with best validation performance
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}
    best_val_loss = float('inf')
    counter = 0
    best_model_state = None

    # Log model architecture if using wandb
    if is_wandb_enabled():
        # wandb.watch(model, log="all", log_freq=2)
        try:
            model_summary = get_model_summary(model)
        except:
            model_summary = repr(model)

        # Log model architecture
        wandb.log({
            "model_architecture": model_summary
        })

    # Instead of wrapping epochs in tqdm, we'll manually print epoch progress
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        # Training phase
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        debug_counter = 0
        # Use tqdm for batch-level progress
        train_loop = tqdm(train_loader, desc=f"Training {model_name}", leave=False)
        for batch in train_loop:
            # Check for required fields
            if 'temporal_features' not in batch or 'static_features' not in batch or 'target' not in batch:
                raise ValueError("Batch missing required fields: 'temporal_features', 'static_features', or 'target'")

            temporal_features = batch['temporal_features'].to(device)
            static_features = batch['static_features'].to(device)
            target = batch['target'].to(device)

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

            batch_loss = loss.item()
            batch_mae = F.l1_loss(output, target, reduction='mean').item()

            train_loss += batch_loss * temporal_features.size(0)
            train_mae += F.l1_loss(output, target, reduction='sum').item()

            # Update the progress bar with current batch metrics
            train_loop.set_postfix(loss=batch_loss, mae=batch_mae)

            debug_counter += 1
            if debug_mode and debug_counter > 10:
                break

        train_loss /= len(train_loader.dataset)
        train_mae /= len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0

        with torch.no_grad():
            # Use tqdm for validation batches as well
            val_loop = tqdm(val_loader, desc=f"Validating {model_name}", leave=False)
            debug_counter = 0
            for batch in val_loop:
                # Check for required fields
                if 'temporal_features' not in batch or 'static_features' not in batch or 'target' not in batch:
                    raise ValueError("Batch missing required fields: 'temporal_features', 'static_features', or 'target'")

                temporal_features = batch['temporal_features'].to(device)
                static_features = batch['static_features'].to(device)
                target = batch['target'].to(device)

                # Ensure target has the right shape for broadcasting
                if len(target.shape) == 1 and target.shape[0] > 1:
                    # If target is [batch_size], reshape to [batch_size, 1]
                    target = target.view(-1, 1)

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
                batch_loss = loss.item()
                batch_mae = F.l1_loss(output, target, reduction='mean').item()

                val_loss += loss.item() * temporal_features.size(0)
                val_mae += F.l1_loss(output, target, reduction='sum').item()

                # Update validation progress bar
                val_loop.set_postfix(loss=batch_loss, mae=batch_mae)

                debug_counter += 1
                if debug_mode and debug_counter > 10:
                    break

        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)

        # Update learning rate
        scheduler.step(val_loss)

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)

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

            # Save best model checkpoint in wandb
            if is_wandb_enabled():
                model_path = f"{model_name}_best.pt"
                torch.save(model.state_dict(), model_path)
                wandb.save(model_path)
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    print("LOADING BEST MODEL FROM TRAINING SESSION")
    model.load_state_dict(best_model_state)
    return history

@track_experiment
def train_pinn_model(model, train_loader, val_loader, model_name="PINN-Model", epochs=50, patience=10, lr=0.001,
                     lambda_night=1.0, lambda_neg=0.5, lambda_clear=0.1):
    """
    Train a physics-informed model with custom loss functions

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        model_name: Name of the model for logging
        epochs: Maximum number of epochs
        patience: Early stopping patience
        lr: Learning rate
        lambda_night: Weight for nighttime constraint loss
        lambda_neg: Weight for non-negativity constraint loss
        lambda_clear: Weight for clear-sky alignment loss

    Returns:
        history: Dictionary of training history
        best_model: Model with best validation performance
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}
    best_val_loss = float('inf')
    counter = 0
    best_model_state = None

    # Log model architecture if using wandb
    if USE_WANDB:
        wandb.watch(model, log="all", log_freq=100)

    # Instead of wrapping epochs in tqdm, we'll manually print epoch progress
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        # Training phase
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        train_loss_components = {'mse': 0.0, 'night': 0.0, 'neg': 0.0, 'clear': 0.0}

        # Use tqdm for batch-level progress
        train_loop = tqdm(train_loader, desc=f"Training {model_name}", leave=False)
        for batch in train_loop:
            temporal_features = batch['temporal_features'].to(device)
            static_features = batch['static_features'].to(device)
            target = batch['target'].to(device)

            # Ensure target has the right shape for broadcasting
            if len(target.shape) == 1 and target.shape[0] > 1:
                # If target is [batch_size], reshape to [batch_size, 1]
                target = target.view(-1, 1)

            # Check if nighttime and clear_sky fields are present
            has_nighttime = 'nighttime' in batch
            has_clear_sky = 'clear_sky' in batch

            if has_nighttime:
                nighttime = batch['nighttime'].to(device)
                # Ensure nighttime has same shape as target
                if nighttime.shape != target.shape:
                    nighttime = nighttime.view(*target.shape)
            else:
                # Default: assume all data is daytime (no nighttime constraint)
                nighttime = torch.zeros_like(target).to(device)

            if has_clear_sky:
                clear_sky = batch['clear_sky'].to(device)
                # Ensure clear_sky has same shape as target
                if clear_sky.shape != target.shape:
                    clear_sky = clear_sky.view(*target.shape)
            else:
                # Default: use the target as a proxy for clear sky (no clear-sky constraint)
                clear_sky = target.clone().to(device)

            optimizer.zero_grad()
            output = model(temporal_features, static_features)

            # Ensure shapes match for loss calculation
            if output.shape != target.shape:
                if len(output.shape) > len(target.shape):
                    # If output has more dimensions than target, reshape target
                    target = target.view(*output.shape)
                    nighttime = nighttime.view(*output.shape)
                    clear_sky = clear_sky.view(*output.shape)
                else:
                    # If target has more dimensions, reshape output
                    output = output.view(*target.shape)

            # Data loss
            mse_loss = criterion(output, target)

            # Nighttime loss (GHI = 0 when nighttime = 1)
            if has_nighttime:
                night_loss = torch.mean((output * nighttime) ** 2)
            else:
                night_loss = torch.tensor(0.0).to(device)

            # Non-negativity loss (penalize GHI < 0)
            neg_loss = torch.mean(torch.relu(-output) ** 2)

            # Clear-sky loss (align with clear-sky GHI during daytime)
            if has_nighttime and has_clear_sky:
                daytime_mask = 1 - nighttime
                clear_loss = torch.mean((daytime_mask * (output - clear_sky)) ** 2)
            else:
                clear_loss = torch.tensor(0.0).to(device)

            # Adjust lambda weights if needed features aren't present
            effective_lambda_night = lambda_night if has_nighttime else 0.0
            effective_lambda_clear = lambda_clear if has_nighttime and has_clear_sky else 0.0

            # Total loss
            total_loss = mse_loss + effective_lambda_night * night_loss + lambda_neg * neg_loss + effective_lambda_clear * clear_loss

            total_loss.backward()
            optimizer.step()

            # Track loss components
            batch_size = temporal_features.size(0)
            batch_total_loss = total_loss.item()
            batch_mae = F.l1_loss(output, target, reduction='mean').item()

            train_loss += total_loss.item() * batch_size
            train_mae += F.l1_loss(output, target, reduction='sum').item()
            train_loss_components['mse'] += mse_loss.item() * batch_size
            train_loss_components['night'] += night_loss.item() * batch_size
            train_loss_components['neg'] += neg_loss.item() * batch_size
            train_loss_components['clear'] += clear_loss.item() * batch_size

            # Update progress bar with current batch metrics
            train_loop.set_postfix(
                loss=batch_total_loss,
                mae=batch_mae,
                mse=mse_loss.item(),
                night=night_loss.item() if has_nighttime else 0
            )

        # Normalize losses
        train_loss /= len(train_loader.dataset)
        train_mae /= len(train_loader.dataset)
        for key in train_loss_components:
            train_loss_components[key] /= len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_loss_components = {'mse': 0.0, 'night': 0.0, 'neg': 0.0, 'clear': 0.0}

        with torch.no_grad():
            # Use tqdm for validation batches
            val_loop = tqdm(val_loader, desc=f"Validating {model_name}", leave=False)
            for batch in val_loop:
                temporal_features = batch['temporal_features'].to(device)
                static_features = batch['static_features'].to(device)
                target = batch['target'].to(device)

                # Ensure target has the right shape for broadcasting
                if len(target.shape) == 1 and target.shape[0] > 1:
                    # If target is [batch_size], reshape to [batch_size, 1]
                    target = target.view(-1, 1)

                # Check if nighttime and clear_sky fields are present
                has_nighttime = 'nighttime' in batch
                has_clear_sky = 'clear_sky' in batch

                if has_nighttime:
                    nighttime = batch['nighttime'].to(device)
                    # Ensure nighttime has same shape as target
                    if nighttime.shape != target.shape:
                        nighttime = nighttime.view(*target.shape)
                else:
                    # Default: assume all data is daytime (no nighttime constraint)
                    nighttime = torch.zeros_like(target).to(device)

                if has_clear_sky:
                    clear_sky = batch['clear_sky'].to(device)
                    # Ensure clear_sky has same shape as target
                    if clear_sky.shape != target.shape:
                        clear_sky = clear_sky.view(*target.shape)
                else:
                    # Default: use the target as a proxy for clear sky (no clear-sky constraint)
                    clear_sky = target.clone().to(device)

                output = model(temporal_features, static_features)

                # Ensure shapes match for loss calculation
                if output.shape != target.shape:
                    if len(output.shape) > len(target.shape):
                        # If output has more dimensions than target, reshape target
                        target = target.view(*output.shape)
                        nighttime = nighttime.view(*output.shape)
                        clear_sky = clear_sky.view(*output.shape)
                    else:
                        # If target has more dimensions, reshape output
                        output = output.view(*target.shape)

                # Data loss
                mse_loss = criterion(output, target)

                # Nighttime loss
                if has_nighttime:
                    night_loss = torch.mean((output * nighttime) ** 2)
                else:
                    night_loss = torch.tensor(0.0).to(device)

                # Non-negativity loss
                neg_loss = torch.mean(torch.relu(-output) ** 2)

                # Clear-sky loss
                if has_nighttime and has_clear_sky:
                    daytime_mask = 1 - nighttime
                    clear_loss = torch.mean((daytime_mask * (output - clear_sky)) ** 2)
                else:
                    clear_loss = torch.tensor(0.0).to(device)

                # Adjust lambda weights if needed features aren't present
                effective_lambda_night = lambda_night if has_nighttime else 0.0
                effective_lambda_clear = lambda_clear if has_nighttime and has_clear_sky else 0.0

                # Total loss
                total_loss = mse_loss + effective_lambda_night * night_loss + lambda_neg * neg_loss + effective_lambda_clear * clear_loss

                # Track loss components
                batch_size = temporal_features.size(0)
                batch_total_loss = total_loss.item()
                batch_mae = F.l1_loss(output, target, reduction='mean').item()

                val_loss += total_loss.item() * batch_size
                val_mae += F.l1_loss(output, target, reduction='sum').item()
                val_loss_components['mse'] += mse_loss.item() * batch_size
                val_loss_components['night'] += night_loss.item() * batch_size
                val_loss_components['neg'] += neg_loss.item() * batch_size
                val_loss_components['clear'] += clear_loss.item() * batch_size

                # Update validation progress bar
                val_loop.set_postfix(
                    loss=batch_total_loss,
                    mae=batch_mae,
                    mse=mse_loss.item(),
                    night=night_loss.item() if has_nighttime else 0
                )

        # Normalize validation metrics
        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)
        for key in val_loss_components:
            val_loss_components[key] /= len(val_loader.dataset)

        # Update learning rate
        scheduler.step(val_loss)

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)

        # Log metrics to wandb
        if USE_WANDB:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_mae': train_mae,
                'val_mae': val_mae,
                'train_mse_loss': train_loss_components['mse'],
                'train_night_loss': train_loss_components['night'],
                'train_neg_loss': train_loss_components['neg'],
                'train_clear_loss': train_loss_components['clear'],
                'val_mse_loss': val_loss_components['mse'],
                'val_night_loss': val_loss_components['night'],
                'val_neg_loss': val_loss_components['neg'],
                'val_clear_loss': val_loss_components['clear'],
                'learning_rate': optimizer.param_groups[0]['lr']
            })

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")
        print(f"Train Loss Components: MSE: {train_loss_components['mse']:.4f}, "
              f"Night: {train_loss_components['night']:.4f}, "
              f"Neg: {train_loss_components['neg']:.4f}, "
              f"Clear: {train_loss_components['clear']:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            best_model_state = model.state_dict().copy()

            # Save best model checkpoint in wandb
            if USE_WANDB:
                model_path = f"{model_name}_best.pt"
                torch.save(model.state_dict(), model_path)
                wandb.save(model_path)
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(best_model_state)
    return history

def evaluate_model(model, data_loader, target_scaler, model_name="", log_to_wandb=True):
    """
    Evaluate a model on a dataset and compute metrics.

    Args:
        model: PyTorch model
        data_loader: Data loader for evaluation
        target_scaler: Scaler for the target variable
        model_name: Name of the model for logging
        log_to_wandb: Whether to log to wandb

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    all_outputs = []
    all_targets = []
    all_nighttime = []
    has_nighttime_data = False

    with torch.no_grad():
        debug_counter = 0
        for batch in data_loader:
            # Check for required fields
            if 'temporal_features' not in batch or 'static_features' not in batch or 'target' not in batch:
                raise ValueError("Batch missing required fields: 'temporal_features', 'static_features', or 'target'")

            temporal_features = batch['temporal_features'].to(device)
            static_features = batch['static_features'].to(device)
            target = batch['target'].to(device)

            # Ensure target has the right shape for broadcasting
            if len(target.shape) == 1 and target.shape[0] > 1:
                target = target.view(-1, 1)

            # Check if we have nighttime data
            if 'nighttime' in batch:
                has_nighttime_data = True
                nighttime = batch['nighttime']
                # Ensure nighttime has same shape as target
                if nighttime.shape != target.cpu().shape:
                    nighttime = nighttime.view(*target.cpu().shape)
            else:
                # Create a placeholder (all zeros) for nighttime
                nighttime = torch.zeros_like(target).cpu()

            output = model(temporal_features, static_features)

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

            debug_counter += 1
            if debug_counter > 10:
                break

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
    else:
        day_mse = day_rmse = day_mae = day_r2 = np.nan

    if np.sum(night_mask) > 0:
        night_mse = mean_squared_error(y_true_orig[night_mask], y_pred_orig[night_mask])
        night_rmse = np.sqrt(night_mse)
        night_mae = mean_absolute_error(y_true_orig[night_mask], y_pred_orig[night_mask])
        night_r2 = r2_score(y_true_orig[night_mask], y_pred_orig[night_mask]) if np.unique(y_true_orig[night_mask]).size > 1 else np.nan
    else:
        night_mse = night_rmse = night_mae = night_r2 = np.nan

    # Create evaluation metrics dictionary
    metrics = {
        'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2,
        'day_mse': day_mse, 'day_rmse': day_rmse, 'day_mae': day_mae, 'day_r2': day_r2,
        'night_mse': night_mse, 'night_rmse': night_rmse, 'night_mae': night_mae, 'night_r2': night_r2,
        'y_pred': y_pred_orig, 'y_true': y_true_orig, 'nighttime': all_nighttime
    }

    # Print metrics
    print(f"\n{model_name} Evaluation Metrics:")
    print(f"  Overall:  MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
    print(f"  Daytime:  MSE: {day_mse:.2f}, RMSE: {day_rmse:.2f}, MAE: {day_mae:.2f}, R²: {day_r2:.4f}")

    if has_nighttime_data:
        print(f"  Nighttime: MSE: {night_mse:.2f}, RMSE: {night_rmse:.2f}, MAE: {night_mae:.2f}, R²: {night_r2:.4f if not np.isnan(night_r2) else 'N/A'}")
    else:
        print("  Nighttime metrics: Not available (no nighttime data)")

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
                ["R²", float(r2), float(day_r2), float(night_r2)]
            ]
        )

        # Create a summary dictionary for key metrics
        summary_metrics = {
            f"{eval_prefix}mse": mse,
            f"{eval_prefix}rmse": rmse,
            f"{eval_prefix}mae": mae,
            f"{eval_prefix}r2": r2,
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
            f"{eval_prefix}predictions_sample": pred_table,
            **summary_metrics
        })

        # Create and log visualization plots
        fig = create_evaluation_plots(metrics, model_name)
        wandb.log({f"{eval_prefix}evaluation_plot": wandb.Image(fig)})
        plt.close(fig)

    return metrics


def plot_training_history(history, model_name=""):
    """
    Plot training and validation loss history

    Args:
        history: Dictionary of training history
        model_name: Name of the model for the plot title
    """
    fig = plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title(f'{model_name} - Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_mae'], label='Train')
    plt.plot(history['val_mae'], label='Validation')
    plt.title(f'{model_name} - MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    return fig


def create_evaluation_plots(metrics, model_name=''):
    """
    Create evaluation plots for wandb logging

    Args:
        metrics: Dictionary of evaluation metrics
        model_name: Name of the model

    Returns:
        fig: Matplotlib figure object
    """
    y_true = metrics['y_true']
    y_pred = metrics['y_pred']

    # Check if we have meaningful nighttime data (any non-zero values)
    has_nighttime = metrics['nighttime'] is not None and np.any(metrics['nighttime'] > 0.5)

    if has_nighttime:
        nighttime = metrics['nighttime'].flatten() > 0.5
    else:
        # Create all daytime mask
        nighttime = np.zeros(len(y_true), dtype=bool)

    # Calculate residuals
    residuals = y_true - y_pred

    # Sample a subset for visualization
    max_samples = 1000
    if len(y_true) > max_samples:
        sample_indices = np.random.choice(len(y_true), max_samples, replace=False)
        y_true_sample = y_true[sample_indices]
        y_pred_sample = y_pred[sample_indices]
        residuals_sample = residuals[sample_indices]
        nighttime_sample = nighttime[sample_indices]
    else:
        y_true_sample = y_true
        y_pred_sample = y_pred
        residuals_sample = residuals
        nighttime_sample = nighttime

    fig = plt.figure(figsize=(15, 12))

    # Actual vs Predicted (colored by time of day)
    ax1 = fig.add_subplot(2, 2, 1)
    day_indices = ~nighttime_sample
    night_indices = nighttime_sample

    ax1.scatter(y_true_sample[day_indices], y_pred_sample[day_indices],
                alpha=0.5, c='skyblue', label='Daytime')

    if has_nighttime and np.any(night_indices):
        ax1.scatter(y_true_sample[night_indices], y_pred_sample[night_indices],
                    alpha=0.5, c='navy', label='Nighttime')

    max_val = max(np.max(y_true_sample), np.max(y_pred_sample))
    ax1.plot([0, max_val], [0, max_val], 'r--')
    ax1.set_title(f'{model_name} - Actual vs Predicted GHI')
    ax1.set_xlabel('Actual GHI (W/m²)')
    ax1.set_ylabel('Predicted GHI (W/m²)')
    ax1.legend()
    ax1.grid(True)

    # Histogram of residuals
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.hist(residuals_sample, bins=50, alpha=0.7, color='green')
    ax2.axvline(x=0, color='r', linestyle='--')
    ax2.set_title(f'{model_name} - Residuals Distribution')
    ax2.set_xlabel('Residual (W/m²)')
    ax2.set_ylabel('Frequency')
    ax2.grid(True)

    # Residuals vs Predicted
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(y_pred_sample, residuals_sample, alpha=0.5, color='purple')
    ax3.axhline(y=0, color='r', linestyle='--')
    ax3.set_title(f'{model_name} - Residuals vs Predicted')
    ax3.set_xlabel('Predicted GHI (W/m²)')
    ax3.set_ylabel('Residual (W/m²)')
    ax3.grid(True)

    # Metrics summary
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    metrics_text = f"Overall Metrics:\n"
    metrics_text += f"  MSE: {metrics['mse']:.2f}\n"
    metrics_text += f"  RMSE: {metrics['rmse']:.2f}\n"
    metrics_text += f"  MAE: {metrics['mae']:.2f}\n"
    metrics_text += f"  R²: {metrics['r2']:.4f}\n\n"

    metrics_text += f"Daytime Metrics:\n"
    metrics_text += f"  MSE: {metrics['day_mse']:.2f}\n"
    metrics_text += f"  RMSE: {metrics['day_rmse']:.2f}\n"
    metrics_text += f"  MAE: {metrics['day_mae']:.2f}\n"
    metrics_text += f"  R²: {metrics['day_r2']:.4f}\n\n"

    if has_nighttime:
        metrics_text += f"Nighttime Metrics:\n"
        metrics_text += f"  MSE: {metrics['night_mse']:.2f}\n"
        metrics_text += f"  RMSE: {metrics['night_rmse']:.2f}\n"
        metrics_text += f"  MAE: {metrics['night_mae']:.2f}\n"
        metrics_text += f"  R²: {metrics['night_r2']:.4f if not np.isnan(metrics['night_r2']) else 'N/A'}\n\n"
    else:
        metrics_text += f"Nighttime Metrics: Not available\n\n"

    metrics_text += f"Residual Stats:\n"
    metrics_text += f"  Mean: {np.mean(residuals):.2f}\n"
    metrics_text += f"  StdDev: {np.std(residuals):.2f}\n"

    ax4.text(0.05, 0.95, metrics_text, fontsize=10, va='top')

    plt.tight_layout()
    return fig


def plot_predictions(metrics, model_name=''):
    """
    Plot model predictions and evaluation metrics

    Args:
        metrics: Dictionary of evaluation metrics
        model_name: Name of the model for the plot title
    """
    fig = create_evaluation_plots(metrics, model_name)
    plt.show()
    plt.close(fig)
    return fig
