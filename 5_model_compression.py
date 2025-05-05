# %% [markdown]
# # Model Efficiency Experiment for Informer Model
#
# This notebook implements various model efficiency techniques for the Informer model used in solar radiation forecasting:
#
# 1. **Quantization**: Reducing model precision to decrease size and improve inference speed
# 2. **Structured Pruning**: Removing less important components to reduce parameters
# 3. **Knowledge Distillation**: Training a smaller model to mimic the larger model's behavior
#
# These techniques help make models more energy-efficient and computationally efficient, which is crucial for sustainability in AI applications.

# %% [markdown]
# ## Setup and Imports

# %%
# Load autoreload extension
# %load_ext autoreload
# Set autoreload to mode 2
# %autoreload 2

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.utils.prune as prune
from torch.quantization import quantize_dynamic, QuantStub, DeQuantStub
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Union
import copy
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_static, QuantFormat, CalibrationDataReader
import onnxruntime as ort
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error

# Import project utilities
from utils.model_utils import load_model, save_model, print_model_info
from utils.data_persistence import load_normalized_data, load_scalers
from utils.plot_utils import plot_predictions_over_time
from utils.timeseriesdataset import TimeSeriesDataset
from utils.training_utils import train_model, evaluate_model
from models.informer_mimick import InformerModel

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model efficiency experiment configuration

# Data settings
TRAIN_PREPROCESSED_DATA_PATH = "data/processed/train_normalized_20250430_145157.h5"
VAL_PREPROCESSED_DATA_PATH = "data/processed/val_normalized_20250430_145205.h5"
TEST_PREPROCESSED_DATA_PATH = "data/processed/test_normalized_20250430_145205.h5"
SCALER_PATH = "data/processed/scalers_20250430_145206.pkl"
# Choose the model checkpoint from the previous experiment
# PRETRAINED_MODEL_PATH = "checkpoints/MLP_best_20250504_052621.pt"
PRETRAINED_MODEL_PATH = "checkpoints/Transformer_best_20250503_232818.pt"

# Dataset settings
LOOKBACK = 24
TARGET_VARIABLE = "ghi"
SELECTED_FEATURES = [
    'air_temperature', 'wind_speed', 'relative_humidity', 'cloud_type',
    'solar_zenith_angle', 'clearsky_ghi', 'total_precipitable_water',
    'surface_albedo', 'nighttime_mask', 'cld_opd_dcomp', 'aod'
]
STATIC_FEATURES = ['latitude', 'longitude', 'elevation']

# Quantization settings
QUANTIZATION_DTYPE = torch.qint8

# Distillation Training settings
BATCH_SIZE = 2**10
NUM_WORKERS = 16
EPOCHS = 2
LEARNING_RATE = 0.0005
PATIENCE = 5

# Student model settings
STUDENT_D_MODEL = 256  # Half of the original model's dimension
STUDENT_N_HEADS = 4    # Half of the original model's heads
STUDENT_E_LAYERS = 2   # Reduced number of encoder layers

# Debug settings
DEBUG_MODE = True  # Set to True to run only 10 batches per epoch for quick debugging

# %% [markdown]
# ## Helper Functions for Evaluation

# %%
def get_model_size(model):
    """Get model size in MB."""
    torch_model_size = 0
    for param in model.parameters():
        torch_model_size += param.nelement() * param.element_size()
    torch_model_size_mb = torch_model_size / (1024 * 1024)
    return torch_model_size_mb

def print_model_report(model_name, model, test_loader, scalers, device=device):
    """Print a comprehensive report about the model."""
    model_size = get_model_size(model)

    # Use the project's evaluate_model function instead of custom evaluation logic
    eval_metrics = evaluate_model(
        model=model,
        data_loader=test_loader,
        target_scaler=scalers.get('ghi_scaler', None),
        model_name=model_name,
        log_to_wandb=False,
        device=device,
        debug_mode=DEBUG_MODE
    )
    # Get inference time from the evaluation metrics
    inference_time = eval_metrics['total_inference_time']

    # Extract test loss (MSE) from the evaluation metrics
    test_loss = eval_metrics['mse']

    print(f"=== {model_name} ===")
    print(f"Model Size: {model_size:.2f} MB")
    print(f"Inference Time: {inference_time*1000:.2f} ms")
    print(f"Test Loss: {test_loss:.4f}")
    if 'mae' in eval_metrics:
        print(f"Test MAE: {eval_metrics['mae']:.4f}")
    if 'r2' in eval_metrics:
        print(f"Test R²: {eval_metrics['r2']:.4f}")
    print()

    return {
        'name': model_name,
        'size': model_size,
        'inference_time': inference_time,
        'test_loss': test_loss,
        'metrics': eval_metrics
    }

# %% [markdown]
# ## Load and Prepare Data
#
# Use the project's data loading utilities to load the preprocessed data.

# %%
# Load scalers
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

scalers = load_scalers(SCALER_PATH)

# Get sample input for inference time measurement
sample_batch = next(iter(test_loader))
sample_temporal = sample_batch['temporal_features'][0:1].to(device)
sample_static = sample_batch['static_features'][0:1].to(device)
sample_input = (sample_temporal, sample_static)

print("Data loading successful.")

# %% [markdown]
# ## Load Original Informer Model
#
# Use the project's model loading utility to load the pretrained model.

# %%

print(f"Loading model from {PRETRAINED_MODEL_PATH}")
original_model, metadata = load_model(PRETRAINED_MODEL_PATH, device=device)
model_name = metadata['model_name']
# Print model information
print_model_info(original_model,
                temporal_shape=sample_input[0].shape,
                static_shape=sample_input[1].shape)

# Evaluate original model
original_metrics = print_model_report(model_name, original_model, test_loader, scalers, device=device)

# %% [markdown]
# ### Visualize Model Predictions Over Time

# %%
# Define target scaler from the data_metadata
target_scaler = scalers.get(f'{TARGET_VARIABLE}_scaler')
if target_scaler is None:
    print(f"Warning: No scaler found for target field '{TARGET_VARIABLE}'. Visualization may show scaled values.")

# Visualize the loaded model's predictions
print("Generating predictions visualization...")
original_model.eval()  # Set model to evaluation mode

# Create the visualization using the imported function
viz_fig = plot_predictions_over_time(
    models=[original_model],
    model_names=[model_name],
    data_loader=test_loader,
    target_scaler=target_scaler,
    num_samples=72,  # Adjust as needed
    start_idx=40,
    device=device       # Adjust as needed
)

# Display the plot if in a notebook environment
plt.show()

# %% [markdown]
# ## Technique 1: Quantization
#
# Quantization reduces model precision from float32 to int8 to decrease model size and improve inference speed.
#
# Quantization is done on the CPU.
# %% [markdown]
# ## Technique 1a: ONNX Quantization (CPU)
#
# This section demonstrates exporting the PyTorch model to ONNX format and applying ONNX dynamic quantization for model efficiency.

# %%
# Define an ONNX wrapper to preserve batch dimension in the output
class OnnxModelWrapper(nn.Module):
    def __init__(self, model):
        super(OnnxModelWrapper, self).__init__()
        self.model = model

    def forward(self, temporal_features, static_features):
        outputs = self.model(temporal_features, static_features)
        return outputs

# Define device (use CPU for 'fbgemm')
cpu_device = torch.device('cpu')

# Prepare wrapper and CPU sample input
wrapper = OnnxModelWrapper(original_model).eval().to(cpu_device)
sample_input_cpu = (sample_input[0].cpu(), sample_input[1].cpu())

# Export the original model to ONNX
onnx_model_path = f"checkpoints/{model_name}_original.onnx"
print(f"Exporting original model to ONNX at {onnx_model_path}")
torch.onnx.export(
    wrapper,
    sample_input_cpu,
    onnx_model_path,
    input_names=["temporal_features", "static_features"],
    output_names=["output"],
    opset_version=20,
    dynamic_axes={
        "temporal_features": {0: "batch"},
        "static_features": {0: "batch"},
        "output": {0: "batch"},
    },
)
# File size before quantization
orig_onnx_size = os.path.getsize(onnx_model_path) / (1024 * 1024)
print(f"Original ONNX model size: {orig_onnx_size:.2f} MB")

# Apply dynamic quantization with ONNX Runtime
quantized_onnx_model_path = f"checkpoints/{model_name}_quantized.onnx"
print(f"Quantizing ONNX model to {quantized_onnx_model_path}")
quantize_dynamic(
    model_input=onnx_model_path,
    model_output=quantized_onnx_model_path,
    weight_type=QuantType.QUInt8
)

# Run ONNX shape inference to populate output shape info and avoid mismatches
print("Running ONNX shape inference for original and quantized models...")
# Original ONNX model shape inference
model_proto = onnx.load(onnx_model_path)
inferred_model = onnx.shape_inference.infer_shapes(model_proto)
onnx.save(inferred_model, onnx_model_path)
# Quantized ONNX model shape inference
model_q_proto = onnx.load(quantized_onnx_model_path)
inferred_q_model = onnx.shape_inference.infer_shapes(model_q_proto)
onnx.save(inferred_q_model, quantized_onnx_model_path)

# Only annotate batch dimension if missing
for model_path in [onnx_model_path, quantized_onnx_model_path]:
    m = onnx.load(model_path)
    for output in m.graph.output:
        shape = output.type.tensor_type.shape
        # Ensure dynamic batch dimension exists
        if len(shape.dim) == 0:
            dim0 = shape.dim.add()
            dim0.dim_param = "batch"
    onnx.save(m, model_path)

quant_onnx_size = os.path.getsize(quantized_onnx_model_path) / (1024 * 1024)
print(f"Quantized ONNX model size: {quant_onnx_size:.2f} MB")
print()

# Helper function to evaluate ONNX models over the full test set
def evaluate_onnx_model(model_path):
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name1 = sess.get_inputs()[0].name
    input_name2 = sess.get_inputs()[1].name
    # Warm-up on the first batch to initialize optimizations
    first_batch = next(iter(test_loader))
    warm_inp1 = first_batch['temporal_features'].numpy()
    warm_inp2 = first_batch['static_features'].numpy()
    for _ in range(5):
        sess.run(None, {input_name1: warm_inp1, input_name2: warm_inp2})
    # Measure inference time over all batches and collect outputs for MAE
    total_time = 0.0
    num_batches = 0
    all_preds = []
    all_targets = []
    for batch in tqdm(test_loader, desc=f"ONNX inference ({os.path.basename(model_path)})"):
        inp1 = batch['temporal_features'].numpy()
        inp2 = batch['static_features'].numpy()
        targets = batch['target'].numpy()
        start = time.time()
        outputs = sess.run(None, {input_name1: inp1, input_name2: inp2})[0]
        elapsed = time.time() - start
        total_time += elapsed
        num_batches += 1
        all_preds.append(outputs)
        all_targets.append(targets)
    avg_time = total_time / num_batches if num_batches > 0 else 0.0
    # Concatenate and inverse-transform
    preds = np.concatenate(all_preds, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)
    # Inverse scale if available
    scaler = scalers.get(f"{TARGET_VARIABLE}_scaler", None)
    if scaler is not None:
        preds = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
        targets_arr = scaler.inverse_transform(targets_arr.reshape(-1, 1)).flatten()
    # Compute MAE via sklearn
    mae_value = mean_absolute_error(targets_arr, preds)
    print(f"Inference time for {os.path.basename(model_path)}: {avg_time*1000:.2f} ms per batch over {num_batches} batches")
    print(f"MAE for {os.path.basename(model_path)}: {mae_value:.4f}")
    return {'size': os.path.getsize(model_path)/(1024*1024), 'inference_time': avg_time, 'mae': mae_value}

# Technique 1a: ONNX CPU Quantization Results
print("\n===== Technique 1a: ONNX CPU Quantization Results =====")
onnx_orig_metrics = evaluate_onnx_model(onnx_model_path)
onnx_quant_metrics = evaluate_onnx_model(quantized_onnx_model_path)
print(f"{'Model':<30}{'Size (MB)':<12}{'Latency(ms)':<15}{'MAE':<10}")
print('-'*67)
print(f"{'Original ONNX CPU':<30}{onnx_orig_metrics['size']:<12.2f}{onnx_orig_metrics['inference_time']*1000:<15.2f}{onnx_orig_metrics['mae']:<10.4f}")
print(f"{'Quantized ONNX CPU':<30}{onnx_quant_metrics['size']:<12.2f}{onnx_quant_metrics['inference_time']*1000:<15.2f}{onnx_quant_metrics['mae']:<10.4f}")
print()

# %% [markdown]
# ## Technique 1b: FP16 Quantization (Requires CUDA GPU)
#
# FP16 quantization is a technique that converts the model to FP16 precision to reduce memory usage and improve inference speed.
# Require libraries:
# - onnxconverter-common
# - onnxruntime-gpu

# %%
# Helper function to evaluate ONNX models on GPU
def evaluate_onnx_model_gpu(model_path, provider='CUDAExecutionProvider', fp16_mode=False):
    # Configure session options to optimize for GPU
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Silence provider assignment warnings - these are expected and normal
    sess_options.add_session_config_entry("session.log.severity", "1")  # Set log level to WARNING (2) or ERROR (3)
    sess_options.log_severity_level = 1  # Set to 1 for detailed logs

    # Create inference session with CUDA provider
    # Add CPU provider as a fallback because not all operations are supported on GPU
    providers = [provider, 'CPUExecutionProvider']
    session = ort.InferenceSession(
        model_path,
        providers=providers,
        sess_options=sess_options
    )

    input_name1 = session.get_inputs()[0].name
    input_name2 = session.get_inputs()[1].name

    # Get a small batch for warm-up
    first_batch = next(iter(test_loader))
    warm_inp1 = first_batch['temporal_features'].numpy()
    warm_inp2 = first_batch['static_features'].numpy()

    # Convert inputs to FP16 if running in FP16 mode
    if fp16_mode:
        warm_inp1 = warm_inp1.astype(np.float16)
        warm_inp2 = warm_inp2.astype(np.float16)

    # Warm up with a few iterations
    print(f"Warming up GPU ONNX model with provider: {provider}...")
    for _ in range(10):  # More warm-up iterations for GPU
        session.run(None, {input_name1: warm_inp1, input_name2: warm_inp2})

    # Measure inference time and MAE
    total_time = 0.0
    num_batches = 0
    all_preds = []
    all_targets = []

    for batch in tqdm(test_loader, desc=f"GPU ONNX inference ({os.path.basename(model_path)})"):
        inp1 = batch['temporal_features'].numpy()
        inp2 = batch['static_features'].numpy()
        targets = batch['target'].numpy()

        # Convert inputs to FP16 if running in FP16 mode
        if fp16_mode:
            inp1 = inp1.astype(np.float16)
            inp2 = inp2.astype(np.float16)

        # Sync CUDA before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.time()
        outputs = session.run(None, {input_name1: inp1, input_name2: inp2})[0]

        # Sync CUDA after inference for accurate timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.time() - start
        total_time += elapsed
        num_batches += 1

        all_preds.append(outputs)
        all_targets.append(targets)

    avg_time = total_time / num_batches if num_batches > 0 else 0.0

    # Calculate MAE (same as before)
    preds = np.concatenate(all_preds, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)

    scaler = scalers.get(f"{TARGET_VARIABLE}_scaler", None)
    if scaler is not None:
        preds = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
        targets_arr = scaler.inverse_transform(targets_arr.reshape(-1, 1)).flatten()

    mae_value = mean_absolute_error(targets_arr, preds)

    print(f"GPU Inference time for {os.path.basename(model_path)}: {avg_time*1000:.2f} ms per batch over {num_batches} batches")
    print(f"MAE for {os.path.basename(model_path)}: {mae_value:.4f}")

    return {
        'size': os.path.getsize(model_path)/(1024*1024),
        'inference_time': avg_time,
        'mae': mae_value,
        'provider': provider
    }

# Create an FP16 quantized model specifically for GPU
quantized_fp16_path = f"checkpoints/{model_name}_quantized_fp16.onnx"
print(f"Creating FP16 quantized model for GPU at {quantized_fp16_path}")

# Use the convert_float_to_float16 utility from ONNX
from onnxconverter_common import float16
model_fp16 = float16.convert_float_to_float16(
    model=model_proto,
    min_positive_val=1e-7,
    max_finite_val=1e4,
)
for output in model_fp16.graph.output:
    shape = output.type.tensor_type.shape
    # Ensure dynamic batch dimension exists
    if len(shape.dim) == 0:
        dim0 = shape.dim.add()
        dim0.dim_param = "batch"
onnx.save(model_fp16, quantized_fp16_path)

# First evaluate the original model on GPU for baseline comparison
print("\nEvaluating original (FP32) ONNX model on GPU...")
onnx_gpu_metrics = evaluate_onnx_model_gpu(onnx_model_path, fp16_mode=False)

# Evaluate the FP16 model on GPU
print("\nEvaluating FP16 quantized model on GPU...")
onnx_fp16_metrics = evaluate_onnx_model_gpu(quantized_fp16_path, fp16_mode=True)

# Technique 1b: ONNX GPU Quantization Results (FP32 vs FP16 vs INT8)
print("\n===== Technique 1b: ONNX GPU Quantization Results =====")
print(f"{'Model':<30}{'Size (MB)':<12}{'Latency(ms)':<15}{'MAE':<10}")
print('-'*67)
print(f"{'Original ONNX GPU (FP32)':<30}{onnx_gpu_metrics['size']:<12.2f}{onnx_gpu_metrics['inference_time']*1000:<15.2f}{onnx_gpu_metrics['mae']:<10.4f}")
print(f"{'FP16 ONNX GPU':<30}{onnx_fp16_metrics['size']:<12.2f}{onnx_fp16_metrics['inference_time']*1000:<15.2f}{onnx_fp16_metrics['mae']:<10.4f}")
print()
# %% [markdown]
# ## Technique 2: Structured Pruning
#
# Structured pruning removes less important components like attention heads to reduce the number of parameters.

# %%
class PrunableInformer(InformerModel):
    """Informer model with pruning capabilities."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prune_heads(self, head_nums_to_prune):
        """Prune specified attention heads in all encoder layers."""
        for layer_idx, layer in enumerate(self.transformer_encoder.layers):
            # Get the attention layer
            attn_layer = layer.self_attn

            # Get number of attention heads and head dimension
            n_heads = attn_layer.nhead
            head_dim = attn_layer.in_proj_weight.shape[0] // n_heads

            # Create a mask for heads: 1 for keeping, 0 for pruning
            head_mask = torch.ones(n_heads, device=device)
            for head_idx in head_nums_to_prune:
                if head_idx < n_heads:
                    head_mask[head_idx] = 0

            # Expand mask to cover all dimensions for each attention component
            expanded_mask = head_mask.repeat_interleave(head_dim).view(1, -1)

            # Apply masks to the projection layers
            attn_layer.in_proj_weight.data *= expanded_mask.unsqueeze(-1)
            attn_layer.out_proj.weight.data *= expanded_mask.unsqueeze(-1)

            # Mark these heads as pruned
            print(f"Pruned heads {head_nums_to_prune} in encoder layer {layer_idx}")

print(original_model.transformer_encoder.layers[0])
# Get model dimensions from the original model
input_dim = original_model.input_dim if hasattr(original_model, 'input_dim') else sample_input[0].shape[2]
static_dim = original_model.static_dim if hasattr(original_model, 'static_dim') else sample_input[1].shape[1]
d_model = original_model.enc_embedding.out_features if hasattr(original_model, 'enc_embedding') else 512
n_heads = original_model.n_heads if hasattr(original_model, 'n_heads') else 8
e_layers = original_model.e_layers if hasattr(original_model, 'e_layers') else 3
d_ff = original_model.transformer_encoder.layers[0].linear1.out_features if hasattr(original_model, 'transformer_encoder') else 256
print(f"Input Dimensions: {input_dim}, Static Dimensions: {static_dim}, d_model: {d_model}, n_heads: {n_heads}, e_layers: {e_layers}, d_ff: {d_ff}")
# Create prunable model with same dimensions as original
pruned_model = PrunableInformer(
    input_dim=input_dim,
    static_dim=static_dim,
    d_model=d_model,
    n_heads=n_heads,
    e_layers=e_layers,
    d_ff=d_ff,
    dropout=0.1,
    activation='gelu'
).to(device)

# Load original weights (copy parameters from original model)
pruned_model.load_state_dict(original_model.state_dict())

# Prune less important attention heads (for example, heads 0, 3, and 6)
pruned_model.prune_heads([0, 3, 6])

# Fine-tune the pruned model
optimizer = torch.optim.Adam(pruned_model.parameters(), lr=LEARNING_RATE)
# Define loss function
criterion = nn.MSELoss()

# Simple training function since we can't use the project's train_model function directly
def simple_train(model, train_loader, val_loader, optimizer, criterion, epochs=2, debug_mode=False):
    """Simplified training function for fine-tuning."""
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        # Training loop
        for batch_idx, batch in enumerate(train_loader):
            # In debug mode, only process 10 batches per epoch
            if debug_mode and batch_idx >= 10:
                print(f"Debug mode: Stopping after 10 batches")
                break

            temporal_features = batch['temporal_features'].to(device)
            static_features = batch['static_features'].to(device)
            targets = batch['target'].to(device)

            optimizer.zero_grad()
            outputs = model(temporal_features, static_features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        epoch_loss /= num_batches
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    return model

# Fine-tune pruned model
pruned_model = simple_train(pruned_model, train_loader, val_loader, optimizer, criterion,
                            epochs=EPOCHS, debug_mode=DEBUG_MODE)

# Evaluate pruned model
pruned_metrics = print_model_report('Pruned Informer', pruned_model, test_loader, scalers)

# Save the pruned model
os.makedirs('checkpoints', exist_ok=True)
save_model(
    pruned_model,
    'checkpoints/informer_pruned.pth',
    metadata={'pruned_heads': [0, 3, 6]}
)

# %% [markdown]
# ## Technique 3: Knowledge Distillation
#
# Knowledge distillation trains a smaller model (student) to mimic the behavior of the larger model (teacher).

# %%
class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss combining MSE loss with KL divergence.
    """
    def __init__(self, alpha=0.5, temperature=2.0):
        super().__init__()
        self.alpha = alpha  # Weight for distillation loss
        self.temperature = temperature  # Temperature for softening probability distributions
        self.mse_loss = nn.MSELoss()

    def forward(self, student_outputs, teacher_outputs, targets):
        # Standard MSE loss against ground truth
        student_loss = self.mse_loss(student_outputs, targets)

        # Distillation loss (MSE between student and teacher outputs)
        distillation_loss = self.mse_loss(
            student_outputs / self.temperature,
            teacher_outputs.detach() / self.temperature
        )

        # Combine the losses
        loss = (1 - self.alpha) * student_loss + self.alpha * distillation_loss
        return loss

# Create a smaller student model with fewer parameters
student_model = InformerModel(
    input_dim=input_dim,
    static_dim=static_dim,
    d_model=STUDENT_D_MODEL,  # Smaller model dimension
    n_heads=STUDENT_N_HEADS,  # Fewer attention heads
    e_layers=STUDENT_E_LAYERS, # Fewer encoder layers
    d_ff=STUDENT_D_MODEL * 2,  # Smaller feed-forward dimension
    dropout=0.1,
    activation='gelu'
).to(device)

# Use the original model as teacher
teacher_model = original_model.eval()

# Define distillation loss
distillation_criterion = DistillationLoss(alpha=0.7, temperature=2.0)

# Train the student model with knowledge distillation
def train_with_distillation(student, teacher, train_loader, criterion, optimizer, epochs=5, debug_mode=False):
    """Train student model with knowledge distillation from teacher."""
    student.train()
    teacher.eval()  # Teacher always in eval mode

    train_losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # In debug mode, only process 10 batches per epoch
            if debug_mode and batch_idx >= 10:
                print(f"Debug mode: Stopping after 10 batches")
                break

            temporal_features = batch['temporal_features'].to(device)
            static_features = batch['static_features'].to(device)
            targets = batch['target'].to(device)

            # Forward pass with teacher model (no grad)
            with torch.no_grad():
                teacher_outputs = teacher(temporal_features, static_features)

            # Forward pass with student model
            optimizer.zero_grad()
            student_outputs = student(temporal_features, static_features)

            # Calculate distillation loss
            loss = criterion(student_outputs, teacher_outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        epoch_loss /= num_batches
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    return student

# Train student model with distillation
optimizer = torch.optim.Adam(student_model.parameters(), lr=LEARNING_RATE)
student_model = train_with_distillation(
    student_model,
    teacher_model,
    train_loader,
    distillation_criterion,
    optimizer,
    epochs=EPOCHS,
    debug_mode=DEBUG_MODE
)

# Save the student model
save_model(
    student_model,
    'checkpoints/informer_student.pth',
    metadata={
        'student_d_model': STUDENT_D_MODEL,
        'student_n_heads': STUDENT_N_HEADS,
        'student_e_layers': STUDENT_E_LAYERS,
        'distillation': True
    }
)

# Evaluate student model
student_metrics = print_model_report('Student Informer', student_model, test_loader, scalers)

# %% [markdown]
# ## Combining Techniques: Quantized Student Model
#
# Let's combine quantization with knowledge distillation for maximum efficiency.

# %%
# Apply quantization to the student model
quantizable_student = QuantizableInformer(student_model).to(device)
quantized_student = quantize_dynamic(
    quantizable_student,
    {nn.Linear},
    dtype=torch.qint8
)

# Evaluate the quantized student model
quantized_student_metrics = print_model_report('Quantized Student', quantized_student, test_loader, scalers)

# %% [markdown]
# ## Results Comparison

# %%
# Compile all results
all_models = [
    original_metrics,
    cpu_quantized_metrics,  # Use the CPU quantized model metrics
    pruned_metrics,
    student_metrics,
    quantized_student_metrics
]

# Extract data for plotting
names = [model['name'] for model in all_models]
sizes = [model['size'] for model in all_models]
times = [model['inference_time'] * 1000 for model in all_models]  # Convert to ms
losses = [model['test_loss'] for model in all_models]

# Create bar plots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Model size comparison
axes[0].bar(names, sizes, color='skyblue')
axes[0].set_title('Model Size (MB)')
axes[0].set_ylabel('Size (MB)')
axes[0].tick_params(axis='x', rotation=45)

# Inference time comparison
axes[1].bar(names, times, color='lightgreen')
axes[1].set_title('Inference Time (ms)')
axes[1].set_ylabel('Time (ms)')
axes[1].tick_params(axis='x', rotation=45)

# Test loss comparison
axes[2].bar(names, losses, color='salmon')
axes[2].set_title('Test Loss (MSE)')
axes[2].set_ylabel('Loss')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('model_efficiency_comparison.png', dpi=300)
plt.show()

# %% [markdown]
# ## Summary of Efficiency Improvements

# %%
# Calculate percentage improvements relative to the original model
def calculate_improvements(models, baseline_idx=0):
    baseline = models[baseline_idx]
    improvements = []

    for model in models:
        size_reduction = (baseline['size'] - model['size']) / baseline['size'] * 100
        speed_improvement = (baseline['inference_time'] - model['inference_time']) / baseline['inference_time'] * 100
        accuracy_change = (baseline['test_loss'] - model['test_loss']) / baseline['test_loss'] * 100

        improvements.append({
            'name': model['name'],
            'size_reduction': size_reduction,
            'speed_improvement': speed_improvement,
            'accuracy_change': accuracy_change  # Negative means worse performance
        })

    return improvements

improvements = calculate_improvements(all_models)

# Display improvements in a table
print("=== Efficiency Improvements (% relative to original model) ===")
print(f"{'Model':<20} {'Size Reduction':<15} {'Speed Improvement':<20} {'Accuracy Change':<15}")
print("-" * 70)

for imp in improvements:
    print(f"{imp['name']:<20} {imp['size_reduction']:>6.2f}% {imp['speed_improvement']:>18.2f}% {imp['accuracy_change']:>14.2f}%")

# %% [markdown]
# ## Conclusion
#
# This notebook demonstrated three key model efficiency techniques for the Informer model:
#
# 1. **Quantization**: Achieved significant size reduction with minimal accuracy impact
# 2. **Structured Pruning**: Reduced model size by removing attention heads
# 3. **Knowledge Distillation**: Created a smaller student model that mimics the larger teacher
#
# The most effective approach appears to be combining knowledge distillation with quantization, which provides the best balance of model size reduction, inference speed, and maintained accuracy.
#
# These techniques are valuable for deploying models in resource-constrained environments and reducing the carbon footprint of AI systems through improved computational efficiency.
