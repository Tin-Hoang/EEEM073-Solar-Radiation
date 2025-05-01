"""
Module for explainability of machine learning models for solar radiation forecasting.

This module provides tools to explain model predictions using methods like:
- SHAP (SHapley Additive exPlanations)
- Sensitivity Analysis

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import shap
from typing import List, Tuple, Dict, Union, Optional, Any, Callable
import pandas as pd
from abc import ABC, abstractmethod


class BaseExplainer(ABC):
    """Base class for all explainers."""

    def __init__(self, model: torch.nn.Module, feature_names: List[str], static_feature_names: Optional[List[str]] = None):
        """Initialize the base explainer.

        Args:
            model: Trained PyTorch model
            feature_names: List of names for temporal features
            static_feature_names: List of names for static features (if any)
        """
        self.model = model
        self.feature_names = feature_names
        self.static_feature_names = static_feature_names
        self.device = next(model.parameters()).device

    @abstractmethod
    def explain_batch(self, batch_data: Tuple[np.ndarray, Optional[np.ndarray]]) -> np.ndarray:
        """Explain a batch of samples.

        Args:
            batch_data: Tuple of (X_temporal, X_static) numpy arrays

        Returns:
            Explanation values (implementation-specific)
        """
        pass


class ShapExplainer(BaseExplainer):
    """Explainer using SHAP (SHapley Additive exPlanations) for model interpretability."""

    def __init__(self, model: torch.nn.Module, feature_names: List[str], static_feature_names: Optional[List[str]] = None):
        """Initialize the SHAP explainer.

        Args:
            model: Trained PyTorch model
            feature_names: List of names for temporal features
            static_feature_names: List of names for static features (if any)
        """
        super().__init__(model, feature_names, static_feature_names)
        self.explainer = None
        self.algorithm = None
        self.custom_model_wrapper = None  # Added for custom wrapper function

    def set_custom_model_wrapper(self, custom_wrapper: Callable):
        """Set a custom model wrapper function to use in place of the default.

        Args:
            custom_wrapper: A callable function that takes in input data and returns model predictions
        """
        self.custom_model_wrapper = custom_wrapper
        print("Custom model wrapper set successfully.")

    def initialize_explainer(self, background_data: Tuple[np.ndarray, Optional[np.ndarray]], algorithm: str = "kernel"):
        """Initialize the SHAP explainer with background data.

        Args:
            background_data: Tuple of (X_temporal, X_static) numpy arrays to use as background data
            algorithm: SHAP algorithm to use ('kernel' or 'gradient')
        """
        self.algorithm = algorithm
        X_temporal_bg, X_static_bg = background_data

        # Use custom model wrapper if provided, otherwise use the default one
        if self.custom_model_wrapper is not None:
            model_wrapper = self.custom_model_wrapper
            print("Using custom model wrapper for SHAP explainer")
        else:
            # Default model wrapper function
            def model_wrapper(x):
                if len(x.shape) == 2 and len(self.feature_names) != x.shape[1]:
                    # Input is flattened, reshape it for the model
                    # Assuming x shape is (batch_size, lookback * n_features)
                    n_features = len(self.feature_names)
                    lookback = x.shape[1] // n_features
                    try:
                        x_reshaped = x.reshape(x.shape[0], lookback, n_features)
                        x_tensor = torch.tensor(x_reshaped, dtype=torch.float32).to(self.device)
                    except ValueError as e:
                        print(f"Reshape error: Cannot reshape array of size {x.size} into shape ({x.shape[0]},{lookback},{n_features})")
                        print(f"Features: {len(self.feature_names)}, calculated lookback: {lookback}")
                        raise e

                    # Set static features to None if not provided
                    static_input = None

                    with torch.no_grad():
                        return self.model(x_tensor, static_input).cpu().numpy()
                else:
                    # Input is already in the correct shape
                    x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

                    # Set static features to None if not provided
                    static_input = None
                    if X_static_bg is not None:
                        static_input = torch.tensor(X_static_bg, dtype=torch.float32).to(self.device)

                    with torch.no_grad():
                        return self.model(x_tensor, static_input).cpu().numpy()

            print("Using default model wrapper for SHAP explainer")

        # Initialize the appropriate SHAP explainer based on the algorithm
        if algorithm == "kernel":
            # For black-box models
            self.explainer = shap.KernelExplainer(model_wrapper, X_temporal_bg, feature_names=self.feature_names)
        elif algorithm == "gradient":
            # For gradient-based models
            # Convert background data to temporal tensor for GradientExplainer
            temporal_bg_tensor = torch.tensor(X_temporal_bg, dtype=torch.float32).to(self.device)

            # For GradientExplainer, we need a proper torch.nn.Module to ensure correct framework detection
            class PyTorchModelWrapper(torch.nn.Module):
                def __init__(self, orig_model, custom_wrapper, static_tensor, device, use_custom):
                    super().__init__()
                    self.orig_model = orig_model
                    self.custom_wrapper = custom_wrapper
                    self.static_tensor = static_tensor
                    self.device = device
                    self.use_custom = use_custom

                def forward(self, x):
                    if self.use_custom and self.custom_wrapper is not None:
                        # Use the custom wrapper with return_pytorch_tensor=True
                        outputs = self.custom_wrapper(x, return_pytorch_tensor=True)
                        # Ensure outputs have shape (batch_size, num_outputs)
                        if len(outputs.shape) == 1:
                            # If output is 1D (e.g., single output value per sample)
                            outputs = outputs.unsqueeze(1)  # Add a dimension to make it (batch_size, 1)
                        return outputs
                    else:
                        # Default behavior - manage static tensor ourselves
                        batch_size = x.shape[0]
                        static_tensor_for_batch = None

                        if self.static_tensor is not None:
                            if batch_size <= self.static_tensor.shape[0]:
                                static_tensor_for_batch = self.static_tensor[:batch_size]
                            else:
                                # Handle unexpected larger batch size
                                repeat_factor = (batch_size + self.static_tensor.shape[0] - 1) // self.static_tensor.shape[0]
                                static_tensor_for_batch = self.static_tensor.repeat(repeat_factor, 1)[:batch_size]

                        # Call the original model
                        outputs = self.orig_model(x, static_tensor_for_batch)
                        # Ensure outputs have shape (batch_size, num_outputs)
                        if len(outputs.shape) == 1:
                            # If output is 1D (e.g., single output value per sample)
                            outputs = outputs.unsqueeze(1)  # Add a dimension to make it (batch_size, 1)
                        return outputs

            # Prepare static tensor
            static_bg_tensor = None
            if X_static_bg is not None:
                static_bg_tensor = torch.tensor(X_static_bg, dtype=torch.float32).to(self.device)

            # Create the module wrapper
            if self.custom_model_wrapper is not None:
                print("Using custom model wrapper with PyTorchModelWrapper for GradientExplainer")
                wrapped_model = PyTorchModelWrapper(
                    self.model,
                    self.custom_model_wrapper,
                    static_bg_tensor,
                    self.device,
                    use_custom=True
                )
            else:
                print("Using default PyTorchModelWrapper for GradientExplainer")
                wrapped_model = PyTorchModelWrapper(
                    self.model,
                    None,
                    static_bg_tensor,
                    self.device,
                    use_custom=False
                )

            # Initialize GradientExplainer with the PyTorch module wrapper
            print(f"Initializing GradientExplainer with PyTorchModelWrapper. Temporal background shape: {temporal_bg_tensor.shape}")
            self.explainer = shap.GradientExplainer(wrapped_model, temporal_bg_tensor)
        else:
            raise ValueError(f"Unsupported SHAP algorithm: {algorithm}. Use 'kernel' or 'gradient'.")

    def explain_batch(self, batch_data: Tuple[np.ndarray, Optional[np.ndarray]]) -> np.ndarray:
        """Explain a batch of samples using SHAP.

        Args:
            batch_data: Tuple of (X_temporal, X_static) numpy arrays

        Returns:
            SHAP values as numpy array
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer first.")

        X_temporal, X_static = batch_data

        if self.algorithm == "gradient":
            # GradientExplainer expects PyTorch tensors
            x_tensor = torch.tensor(X_temporal, dtype=torch.float32).to(self.device)

            # Print debug info about tensor shapes
            print(f"Input tensor shape for explainer: {x_tensor.shape}")
            print(f"Using algorithm: {self.algorithm}")
            # Directly call the proper method

            print("Calling GradientExplainer.shap_values...")
            # For gradient explainer, we maintain 3D structure for temporal analysis
            # We explicitly do NOT flatten the input for gradient explainer
            if len(x_tensor.shape) == 3:
                print(f"Using 3D input with shape {x_tensor.shape} for gradient explainer")
                shap_values = self.explainer.shap_values(x_tensor)
            else:
                # If input is already 2D, use it as is
                print(f"Using input with shape {x_tensor.shape} for gradient explainer")
                shap_values = self.explainer.shap_values(x_tensor)

            # Convert to numpy if needed
            if isinstance(shap_values, torch.Tensor):
                print(f"Converting PyTorch tensor with shape {shap_values.shape} to numpy")
                shap_values = shap_values.cpu().numpy()
            elif isinstance(shap_values, list):
                # Some explainers return a list of arrays (one per output)
                # We take the first one (assuming single output)
                print(f"Taking first element from list of {len(shap_values)} SHAP values")
                shap_values = shap_values[0]
                if isinstance(shap_values, torch.Tensor):
                    print(f"Converting PyTorch tensor with shape {shap_values.shape} to numpy")
                    shap_values = shap_values.cpu().numpy()

            # For gradient explainer, make sure we preserve the output shape to match input
            # If input is 3D and output shape doesn't match, try to reshape
            if len(x_tensor.shape) == 3 and len(shap_values.shape) > 1:
                orig_shape = x_tensor.shape
                if shap_values.shape[1:] != orig_shape[1:]:
                    print(f"Reshaping SHAP values: {shap_values.shape} -> ({orig_shape[0]}, {orig_shape[1]}, {orig_shape[2]})")
                    try:
                        # Try to reshape to match original 3D input
                        shap_values = shap_values.reshape(orig_shape)
                    except Exception as reshape_err:
                        print(f"Warning: Could not reshape SHAP values: {str(reshape_err)}")
                        # Don't fail completely, just warn and continue

            print(f"Final SHAP values shape: {shap_values.shape}")

        else:
            # KernelExplainer works with numpy arrays
            # For KernelExplainer, we DO need to flatten 3D input
            if len(X_temporal.shape) == 3:
                print(f"Flattening 3D input with shape {X_temporal.shape} for kernel explainer")
                batch_size = X_temporal.shape[0]
                X_temporal_flat = X_temporal.reshape(batch_size, -1)
                shap_values = self.explainer.shap_values(X_temporal_flat)

                # Attempt to reshape the output back to 3D if needed
                try:
                    # If shap_values is a list (multi-output model)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[0]

                    # Try to reshape back to original 3D shape for consistency
                    if len(shap_values.shape) == 2:  # Should be (batch_size, flattened_features)
                        print(f"Reshaping kernel SHAP values back to 3D: {shap_values.shape} -> {X_temporal.shape}")
                        shap_values = shap_values.reshape(X_temporal.shape)
                except Exception as reshape_err:
                    print(f"Warning: Could not reshape kernel SHAP values back to 3D: {str(reshape_err)}")
            else:
                # Already 2D input
                shap_values = self.explainer.shap_values(X_temporal)

            # Some explainers return a list of arrays (one per output)
            # We take the first one (assuming single output)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

        return shap_values

    def plot_feature_importance(self, shap_values: np.ndarray, feature_names: List[str],
                                max_display: int = 20, show: bool = True, title: str = "Feature Importance") -> plt.Figure:
        """Plot feature importance based on mean absolute SHAP values.

        Args:
            shap_values: SHAP values from explain_batch
            feature_names: Names of features corresponding to SHAP values
            max_display: Maximum number of features to display
            show: Whether to show the plot
            title: Title for the plot

        Returns:
            Matplotlib figure object
        """
        # Calculate mean absolute SHAP value for each feature
        feature_importance = np.abs(shap_values).mean(axis=0)

        # Create dataframe for plotting
        importance_df = pd.DataFrame({
            'Feature': feature_names[:len(feature_importance)],
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)

        # Take top features
        if max_display < len(importance_df):
            importance_df = importance_df.head(max_display)

        # Create plot
        fig, ax = plt.subplots(figsize=(10, max(6, max_display * 0.3)))
        ax.barh(importance_df['Feature'], importance_df['Importance'])
        ax.set_xlabel('Mean |SHAP Value|')
        ax.set_title(title)
        plt.tight_layout()

        if show:
            plt.show()

        return fig


class SensitivityAnalyzer(BaseExplainer):
    """Explainer using sensitivity analysis for model interpretability."""

    def __init__(self, model: torch.nn.Module, feature_names: List[str], static_feature_names: Optional[List[str]] = None):
        """Initialize the sensitivity analyzer.

        Args:
            model: Trained PyTorch model
            feature_names: List of names for temporal features
            static_feature_names: List of names for static features (if any)
        """
        super().__init__(model, feature_names, static_feature_names)

    def explain_batch(self, batch_data: Tuple[np.ndarray, Optional[np.ndarray]]) -> np.ndarray:
        """Explain a batch of samples using sensitivity analysis.

        This is a simple implementation that returns the gradient of outputs with respect to inputs.

        Args:
            batch_data: Tuple of (X_temporal, X_static) numpy arrays

        Returns:
            Sensitivity values as numpy array
        """
        X_temporal, X_static = batch_data

        # Convert to tensor and ensure it requires gradients
        X_temporal_tensor = torch.tensor(X_temporal, dtype=torch.float32, requires_grad=True).to(self.device)

        X_static_tensor = None
        if X_static is not None:
            X_static_tensor = torch.tensor(X_static, dtype=torch.float32).to(self.device)

        # Forward pass
        outputs = self.model(X_temporal_tensor, X_static_tensor)

        # Create gradient target (all ones for simplicity)
        gradient_target = torch.ones_like(outputs)

        # Backward pass to get gradients
        outputs.backward(gradient_target)

        # Get gradients
        gradients = X_temporal_tensor.grad.cpu().numpy()

        return gradients

    def analyze_feature_sensitivity(self, X_temporal: np.ndarray, X_static: Optional[np.ndarray],
                                    perturbation: float = 0.1, n_samples: int = 10) -> pd.DataFrame:
        """Analyze feature sensitivity by perturbing inputs.

        Args:
            X_temporal: Temporal features array (batch_size, lookback, n_features)
            X_static: Static features array (batch_size, n_static_features) or None
            perturbation: Fraction by which to perturb features (0.1 = 10%)
            n_samples: Number of samples to use for analysis

        Returns:
            DataFrame with sensitivity scores for each feature
        """
        # Select a subset of samples for efficiency
        indices = np.random.choice(X_temporal.shape[0], min(n_samples, X_temporal.shape[0]), replace=False)
        X_temp_subset = X_temporal[indices].copy()
        X_static_subset = None
        if X_static is not None:
            X_static_subset = X_static[indices].copy()

        # Get baseline predictions
        X_temp_tensor = torch.tensor(X_temp_subset, dtype=torch.float32).to(self.device)
        X_static_tensor = None
        if X_static_subset is not None:
            X_static_tensor = torch.tensor(X_static_subset, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            baseline_preds = self.model(X_temp_tensor, X_static_tensor).cpu().numpy()

        # Initialize results dictionary
        feature_sensitivity = {}

        # Analyze temporal features
        for feat_idx, feat_name in enumerate(self.feature_names):
            # Create perturbed version
            X_temp_perturbed = X_temp_subset.copy()

            # Perturb the feature by increasing its value
            feature_mean = np.mean(X_temp_subset[:, :, feat_idx])
            feature_std = np.std(X_temp_subset[:, :, feat_idx])
            if feature_std == 0:
                # Can't perturb constant features meaningfully
                feature_sensitivity[feat_name] = 0
                continue

            # Apply perturbation (multiplicative for non-zero values, additive for zeros)
            mask = X_temp_perturbed[:, :, feat_idx] != 0
            X_temp_perturbed[:, :, feat_idx][mask] *= (1 + perturbation)
            # For zero values, add a small perturbation based on the mean and std
            X_temp_perturbed[:, :, feat_idx][~mask] += perturbation * max(abs(feature_mean), 0.01 * feature_std)

            # Get predictions with perturbed feature
            X_temp_tensor = torch.tensor(X_temp_perturbed, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                perturbed_preds = self.model(X_temp_tensor, X_static_tensor).cpu().numpy()

            # Calculate sensitivity as mean absolute difference
            sensitivity = np.mean(np.abs(perturbed_preds - baseline_preds))
            feature_sensitivity[feat_name] = sensitivity

        # Analyze static features if available
        if self.static_feature_names is not None and X_static_subset is not None:
            for feat_idx, feat_name in enumerate(self.static_feature_names):
                # Create perturbed version
                X_static_perturbed = X_static_subset.copy()

                # Perturb the feature
                feature_mean = np.mean(X_static_subset[:, feat_idx])
                feature_std = np.std(X_static_subset[:, feat_idx])
                if feature_std == 0:
                    # Can't perturb constant features meaningfully
                    feature_sensitivity[feat_name] = 0
                    continue

                # Apply perturbation
                mask = X_static_perturbed[:, feat_idx] != 0
                X_static_perturbed[:, feat_idx][mask] *= (1 + perturbation)
                # For zero values, add a small perturbation based on the mean and std
                X_static_perturbed[:, feat_idx][~mask] += perturbation * max(abs(feature_mean), 0.01 * feature_std)

                # Get predictions with perturbed feature
                X_static_tensor = torch.tensor(X_static_perturbed, dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    perturbed_preds = self.model(X_temp_tensor, X_static_tensor).cpu().numpy()

                # Calculate sensitivity
                sensitivity = np.mean(np.abs(perturbed_preds - baseline_preds))
                feature_sensitivity[feat_name] = sensitivity

        # Create and return dataframe
        sensitivity_df = pd.DataFrame({
            'Feature': list(feature_sensitivity.keys()),
            'Sensitivity': list(feature_sensitivity.values())
        }).sort_values('Sensitivity', ascending=False)

        return sensitivity_df

    def plot_feature_sensitivity(self, sensitivity_df: pd.DataFrame, max_display: int = 20,
                                 show: bool = True, title: str = "Feature Sensitivity") -> plt.Figure:
        """Plot feature sensitivity.

        Args:
            sensitivity_df: DataFrame from analyze_feature_sensitivity
            max_display: Maximum number of features to display
            show: Whether to show the plot
            title: Title for the plot

        Returns:
            Matplotlib figure object
        """
        # Take top features
        if max_display < len(sensitivity_df):
            sensitivity_df = sensitivity_df.head(max_display)

        # Create plot
        fig, ax = plt.subplots(figsize=(10, max(6, max_display * 0.3)))
        ax.barh(sensitivity_df['Feature'], sensitivity_df['Sensitivity'])
        ax.set_xlabel('Sensitivity Score')
        ax.set_title(title)
        plt.tight_layout()

        if show:
            plt.show()

        return fig
