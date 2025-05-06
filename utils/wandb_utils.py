"""
Weights & Biases (wandb) utilities for experiment tracking.
"""

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import wandb
from functools import wraps

# Default settings
USE_WANDB = False
KEEP_RUN_OPEN = True
WANDB_USERNAME = "tin-hoang"
WANDB_PROJECT = "EEEM073-Solar-Radiation"


def is_wandb_enabled():
    """
    Check if wandb is enabled (with an active run)

    Returns:
        bool: True if wandb is enabled and has an active run
    """
    is_enabled = USE_WANDB and wandb.run is not None
    return is_enabled


def set_wandb_flag(value):
    """
    Set global USE_WANDB flag

    Args:
        value: Boolean value to set

    Returns:
        bool: The new value
    """
    global USE_WANDB
    USE_WANDB = bool(value)
    return USE_WANDB


def set_keep_run_open(value):
    """
    Set global KEEP_RUN_OPEN flag

    Args:
        value: Boolean value to set

    Returns:
        bool: The new value
    """
    global KEEP_RUN_OPEN
    KEEP_RUN_OPEN = bool(value)
    return KEEP_RUN_OPEN


def setup_wandb(username=None, project=None, force_enable=False):
    """
    Set up Weights & Biases tracking

    Args:
        username: Wandb username (default: None)
        project: Wandb project name (default: None)
        force_enable: Force enable wandb even if already configured

    Returns:
        bool: Whether wandb is enabled
    """
    global USE_WANDB, WANDB_USERNAME, WANDB_PROJECT

    # Use provided values or defaults
    wandb_username = username or WANDB_USERNAME
    wandb_project = project or WANDB_PROJECT

    # Only enable wandb if both username and project are provided
    if wandb_username and wandb_project:
        # Set the global flag
        set_wandb_flag(True)

        print(f"Weights & Biases tracking enabled with username '{wandb_username}' and project '{wandb_project}'")
        print(f"USE_WANDB flag is now: {USE_WANDB}")

        # Initialize wandb if needed or forced
        if force_enable or wandb.run is None:
            try:
                wandb.init(
                    project=wandb_project,
                    entity=wandb_username,
                    name=f"Manual-Init-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
                    config={
                        "initialized_by": "setup_wandb",
                        "force_enable": force_enable
                    }
                )
                print(f"Successfully initialized wandb run: {wandb.run.name}")
            except Exception as e:
                print(f"Error initializing wandb: {e}")
    else:
        # If parameters are missing, disable wandb
        set_wandb_flag(False)
        print("Weights & Biases tracking disabled. Provide both username and project to enable.")

    return USE_WANDB


def initialize_wandb(model_name="Model", **config_params):
    """
    Initialize a new wandb run if needed

    Args:
        model_name: Name of the model for the run name
        **config_params: Additional configuration parameters

    Returns:
        bool: Whether initialization was successful
    """
    global USE_WANDB, WANDB_USERNAME, WANDB_PROJECT

    if not USE_WANDB:
        print("WARNING: initialize_wandb called but USE_WANDB is False, will not initialize")
        return False

    # Don't initialize if we already have a run
    if wandb.run is not None:
        print(f"wandb already initialized with run name: {wandb.run.name}")
        return True

    # Create a unique run name
    run_name = f"{model_name}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    try:
        # Initialize wandb
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_USERNAME,
            name=run_name,
            config=config_params
        )
        print(f"Successfully initialized wandb run: {wandb.run.name}")
        return True
    except Exception as e:
        print(f"Error initializing wandb: {e}")
        return False


def track_experiment(func):
    """
    Decorator to track experiments with wandb

    This decorator wraps training functions to automatically log metrics to wandb
    when wandb tracking is enabled.

    The decorated function accepts an additional keyword parameter:
    - keep_run_open (bool): If True, don't close the wandb run after the function completes
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract model name if available
        model_name = kwargs.get('model_name', 'Model')

        # Print debug info
        print(f"track_experiment: USE_WANDB={USE_WANDB}, wandb.run={wandb.run}, keep_run_open={KEEP_RUN_OPEN}")

        # Start wandb run if tracking is enabled
        if USE_WANDB:
            train_config = kwargs.get('config', None)
            temporal_features_shape = train_config.get('TEMPORAL_FEATURES_SHAPE', None)
            static_features_shape = train_config.get('STATIC_FEATURES_SHAPE', None)
            # Try to find the model - check common parameter names and args
            model = kwargs.get('model', None)
            # If model wasn't found in kwargs, check if it might be the first positional argument
            if model is None and len(args) > 0:
                model = args[0]

            try:
                # Import get_model_summary here to avoid circular imports
                from utils.model_utils import get_model_summary
                model_summary = get_model_summary(model,
                                                temporal_features_shape,
                                                static_features_shape)
                model_summary = repr(model_summary)
            except:
                model_summary = repr(model)
            # Get model dictionary
            try:
                model_dict = dict(model.__dict__)
            except:
                model_dict = {}
            # Create config parameters from kwargs
            config = {
                'model_name': model_name,
                'epochs': kwargs.get('epochs', 50),
                'patience': kwargs.get('patience', 10),
                'learning_rate': kwargs.get('lr', 0.001),
                'config': train_config,
                'model_architecture': model_summary,
                'model_dict': model_dict
            }

            # Create the run if it doesn't exist
            if wandb.run is None:
                print(f"Creating new wandb run for {model_name}")

                # Create a unique run name
                run_name = f"{model_name}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

                # Initialize wandb
                wandb.init(
                    project=WANDB_PROJECT,
                    entity=WANDB_USERNAME,
                    name=run_name,
                    config=config
                )

                # Flag to indicate this wrapper created the run
                wrapper_created_run = True
            else:
                print(f"Using existing wandb run: {wandb.run.name}")
                wrapper_created_run = False
            # Run the original function
            history = func(*args, **kwargs)

            # Close wandb run if we created it and keep_run_open is False
            if wrapper_created_run and not KEEP_RUN_OPEN:
                print(f"Finishing wandb run from track_experiment (keep_run_open={KEEP_RUN_OPEN})")
                wandb.finish()
            elif KEEP_RUN_OPEN:
                print(f"Keeping wandb run open as requested (keep_run_open={KEEP_RUN_OPEN})")

            return history
        else:
            # Just run the function without wandb tracking
            return func(*args, **kwargs)

    return wrapper
