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
            algorithm: SHAP algorithm to use ('kernel', 'deep', or 'gradient')
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
        elif algorithm == "deep":
            # For deep learning models
            # Convert model to PyTorch model for DeepExplainer
            def model_pytorch(x):
                x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
                return self.model(x_tensor, None)

            self.explainer = shap.DeepExplainer(model_pytorch,
                torch.tensor(X_temporal_bg, dtype=torch.float32).to(self.device), )
        elif algorithm == "gradient":
            # For gradient-based models
            # Convert model to PyTorch model for GradientExplainer
            def model_pytorch(x):
                x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
                return self.model(x_tensor, None)

            self.explainer = shap.GradientExplainer(model_pytorch, torch.tensor(X_temporal_bg, dtype=torch.float32).to(self.device))
        else:
            raise ValueError(f"Unsupported SHAP algorithm: {algorithm}")

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

        if self.algorithm in ["deep", "gradient"]:
            # DeepExplainer and GradientExplainer expect PyTorch tensors
            x_tensor = torch.tensor(X_temporal, dtype=torch.float32).to(self.device)
            shap_values = self.explainer.shap_values(x_tensor)

            # Convert to numpy if needed
            if isinstance(shap_values, torch.Tensor):
                shap_values = shap_values.cpu().numpy()
            elif isinstance(shap_values, list):
                # Some explainers return a list of arrays (one per output)
                # We take the first one (assuming single output)
                shap_values = shap_values[0]
        else:
            # KernelExplainer works with numpy arrays
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


class MLPExplainer(ShapExplainer):
    """Explainer specialized for MLP models."""

    def __init__(self, model: torch.nn.Module, feature_names: List[str], static_feature_names: Optional[List[str]] = None):
        """Initialize the MLP explainer.

        Args:
            model: Trained MLP model
            feature_names: List of names for temporal features
            static_feature_names: List of names for static features (if any)
        """
        super().__init__(model, feature_names, static_feature_names)


class LSTMExplainer(ShapExplainer):
    """Explainer specialized for LSTM models."""

    def __init__(self, model: torch.nn.Module, feature_names: List[str], static_feature_names: Optional[List[str]] = None):
        """Initialize the LSTM explainer.

        Args:
            model: Trained LSTM model
            feature_names: List of names for temporal features
            static_feature_names: List of names for static features (if any)
        """
        super().__init__(model, feature_names, static_feature_names)


class CNNLSTMExplainer(ShapExplainer):
    """Explainer specialized for CNN-LSTM models."""

    def __init__(self, model: torch.nn.Module, feature_names: List[str], static_feature_names: Optional[List[str]] = None):
        """Initialize the CNN-LSTM explainer.

        Args:
            model: Trained CNN-LSTM model
            feature_names: List of names for temporal features
            static_feature_names: List of names for static features (if any)
        """
        super().__init__(model, feature_names, static_feature_names)


class TransformerExplainer(ShapExplainer):
    """Explainer specialized for Transformer models."""

    def __init__(self, model: torch.nn.Module, feature_names: List[str], static_feature_names: Optional[List[str]] = None):
        """Initialize the Transformer explainer.

        Args:
            model: Trained Transformer model
            feature_names: List of names for temporal features
            static_feature_names: List of names for static features (if any)
        """
        super().__init__(model, feature_names, static_feature_names)


class InformerExplainer(ShapExplainer):
    """Explainer specialized for Informer models."""

    def __init__(self, model: torch.nn.Module, feature_names: List[str], static_feature_names: Optional[List[str]] = None):
        """Initialize the Informer explainer.

        Args:
            model: Trained Informer model
            feature_names: List of names for temporal features
            static_feature_names: List of names for static features (if any)
        """
        super().__init__(model, feature_names, static_feature_names)


def create_explainer(model_type: str, model: torch.nn.Module, feature_names: List[str],
                     static_feature_names: Optional[List[str]] = None) -> BaseExplainer:
    """Factory function to create the appropriate explainer based on model type.

    Args:
        model_type: Type of model ('mlp', 'lstm', 'cnn_lstm', 'transformer', 'informer')
                   or model class name ('MLPModel', 'LSTMModel', etc.)
        model: Trained model
        feature_names: List of feature names
        static_feature_names: List of static feature names (if any)

    Returns:
        Appropriate explainer instance for the model type
    """
    # Convert model class name to short model type name if needed
    model_type_map = {
        'MLPModel': 'mlp',
        'LSTMModel': 'lstm',
        'CNNLSTMModel': 'cnn_lstm',
        'TransformerModel': 'transformer',
        'InformerModel': 'informer'
    }

    # Check if model_type is a class name and convert it
    if model_type in model_type_map:
        print(f"Converting model class name '{model_type}' to model type '{model_type_map[model_type]}'")
        model_type = model_type_map[model_type]

    # Ensure model_type is lowercase for case-insensitive matching
    model_type = model_type.lower()

    if model_type == 'mlp':
        return MLPExplainer(model, feature_names, static_feature_names)
    elif model_type == 'lstm':
        return LSTMExplainer(model, feature_names, static_feature_names)
    elif model_type in ['cnn_lstm', 'cnnlstm']:
        return CNNLSTMExplainer(model, feature_names, static_feature_names)
    elif model_type == 'transformer':
        return TransformerExplainer(model, feature_names, static_feature_names)
    elif model_type == 'informer':
        return InformerExplainer(model, feature_names, static_feature_names)
    else:
        # Default to sensitivity analyzer for unknown model types
        print(f"Warning: Unknown model type '{model_type}'. Using sensitivity analyzer.")
        return SensitivityAnalyzer(model, feature_names, static_feature_names)
