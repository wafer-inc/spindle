"""
Utilities for working with SAE weights.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Union


def extract_encoder_weights(model) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract encoder weights and biases from an SAE model.
    
    Args:
        model: SAE model
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (W, b) encoder weights and biases
    """
    model.eval()
    W = model.enc.weight.detach().cpu().numpy()
    b = model.enc.bias.detach().cpu().numpy()
    return W, b


def save_encoder_weights(model, path: Union[str, Path]):
    """
    Save encoder weights and biases to an .npz file.
    
    Args:
        model: SAE model
        path (Union[str, Path]): Output path
    """
    W, b = extract_encoder_weights(model)
    np.savez(path, W=W, b=b)


def load_encoder_weights(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load encoder weights from an .npz file.
    
    Args:
        path (Union[str, Path]): Path to the .npz file
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (W, b) encoder weights and biases
    """
    data = np.load(path)
    return data["W"], data["b"]


def initialize_from_encoder_weights(
    model, 
    weights_path: Union[str, Path], 
    freeze: bool = False
):
    """
    Initialize a model from saved encoder weights.
    
    Args:
        model: SAE model
        weights_path (Union[str, Path]): Path to the weights file
        freeze (bool, optional): Whether to freeze the encoder parameters. Defaults to False.
    """
    W, b = load_encoder_weights(weights_path)
    
    # Load weights into model
    model.enc.weight.data = torch.tensor(W, dtype=torch.float32, device=model.enc.weight.device)
    model.enc.bias.data = torch.tensor(b, dtype=torch.float32, device=model.enc.bias.device)
    
    # Optionally freeze parameters
    if freeze:
        model.enc.weight.requires_grad = False
        model.enc.bias.requires_grad = False