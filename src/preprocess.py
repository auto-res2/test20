"""
Script for preprocessing data for Hybrid On-Device tnGPS experiments.
This script handles initialization and data preparation.
"""

import os
import sys
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.experiment_config.config import *

def setup_environment():
    """
    Set up the environment for running experiments.
    Creates necessary directories if they don't exist.
    Returns the device to use for tensor operations.
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    return device

def create_dummy_input(input_size=INPUT_SIZE, batch_size=1, device=None):
    """
    Create a dummy input tensor for the models.
    """
    dummy_input = torch.randn(batch_size, input_size)
    if device is not None:
        dummy_input = dummy_input.to(device)
    return dummy_input

if __name__ == "__main__":
    print("Preprocessing module for Hybrid On-Device tnGPS experiments")
    device = setup_environment()
    print(f"Created dummy input: {create_dummy_input(device=device).shape}")
