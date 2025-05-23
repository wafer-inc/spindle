"""
Utilities for analyzing and visualizing SAE features.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from google import genai
from google.genai import types
from typing import Optional, Dict, List, Tuple, Union, Any


def get_top_features(z: torch.Tensor, k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get the top-k activated features for each sample.

    Args:
        z (torch.Tensor): Activation tensor from SAE, shape (batch_size, hidden_dim)
        k (int, optional): Number of top features to return. Defaults to 10.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (indices, values) of top-k features
    """
    return torch.topk(z, k)


def compute_feature_statistics(model, data_loader) -> Dict:
    """
    Compute statistics about feature activations across a dataset.

    Args:
        model: SAE model
        data_loader: DataLoader containing input data

    Returns:
        Dict: Dictionary of feature statistics
    """
    device = next(model.parameters()).device
    num_features = model.enc.weight.shape[0]

    # Initialize counters
    activation_count = torch.zeros(num_features, device=device)
    activation_strength = torch.zeros(num_features, device=device)
    sample_count = 0

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]  # Unpack if it's a tuple/list from DataLoader

            batch = batch.to(device)
            z, _ = model(batch)

            # Count activations
            active = (z > 0)
            activation_count += active.sum(dim=0)

            # Sum activation strengths
            activation_strength += z.sum(dim=0)

            sample_count += batch.size(0)

    # Calculate statistics
    activation_freq = activation_count / sample_count
    mean_activation = activation_strength / \
        activation_count.clamp(min=1)  # Avoid division by zero

    # Calculate feature importance (simple heuristic: frequency × mean strength)
    feature_importance = activation_freq * mean_activation

    return {
        'activation_frequency': activation_freq.cpu().numpy(),
        'mean_activation': mean_activation.cpu().numpy(),
        'feature_importance': feature_importance.cpu().numpy(),
        'dead_feature_count': (activation_count == 0).sum().item(),
        'dead_feature_ratio': (activation_count == 0).sum().item() / num_features,
    }


def extract_encoder_weights(model) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract weights and biases from the encoder part of an SAE.

    Args:
        model: SAE model

    Returns:
        Tuple[np.ndarray, np.ndarray]: (weights, biases) of the encoder
    """
    model.eval()

    # Extract encoder weights
    W = model.enc.weight.detach().cpu().numpy()  # shape (hidden_dim, input_dim)
    b = model.enc.bias.detach().cpu().numpy()    # shape (hidden_dim,)

    return W, b


def visualize_feature_activation(feature_activations, feature_idx, figsize=(12, 6)):
    """
    Visualize the activation distribution of a specific feature.

    Args:
        feature_activations (np.ndarray): Array of feature activations
        feature_idx (int): Index of the feature to visualize
        figsize (tuple, optional): Figure size. Defaults to (12, 6).
    """
    plt.figure(figsize=figsize)

    # Get activations for the specific feature
    activations = feature_activations[:, feature_idx]

    # Histogram of activations
    plt.hist(activations, bins=50, alpha=0.7)
    plt.title(f"Activation Distribution for Feature {feature_idx}")
    plt.xlabel("Activation Value")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)

    return plt.gcf()


def plot_top_features_distribution(feature_stats, top_k=20, figsize=(14, 8)):
    """
    Plot distribution of the top features by importance.

    Args:
        feature_stats (dict): Feature statistics from compute_feature_statistics
        top_k (int, optional): Number of top features to show. Defaults to 20.
        figsize (tuple, optional): Figure size. Defaults to (14, 8).
    """
    importance = feature_stats['feature_importance']
    top_indices = np.argsort(importance)[-top_k:][::-1]

    plt.figure(figsize=figsize)

    # Plot importance
    plt.bar(range(top_k), importance[top_indices])
    plt.xticks(range(top_k), [str(idx) for idx in top_indices], rotation=45)
    plt.title(f"Top {top_k} Features by Importance")
    plt.xlabel("Feature Index")
    plt.ylabel("Importance Score")
    plt.grid(alpha=0.3)

    return plt.gcf()


def analyze_reconstruction_quality(model, data_loader):
    """
    Analyze reconstruction quality of the SAE on a dataset.

    Args:
        model: SAE model
        data_loader: DataLoader containing input data

    Returns:
        Dict: Dictionary of reconstruction quality metrics
    """
    device = next(model.parameters()).device
    total_mse = 0.0
    sample_count = 0

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]

            batch = batch.to(device)
            _, recon = model(batch)

            # Compute MSE
            mse = torch.nn.functional.mse_loss(recon, batch, reduction='sum')
            total_mse += mse.item()
            sample_count += batch.size(0)

    avg_mse = total_mse / sample_count

    return {
        'total_mse': total_mse,
        'avg_mse': avg_mse,
        'rmse': np.sqrt(avg_mse),
    }


def get_top_activated_sources(z_np, texts, feature_idx, top_k=10):
    feature_activations = z_np[:, feature_idx]
    top_indices = np.argsort(-feature_activations)[:top_k]
    top_texts = [texts[i] for i in top_indices]
    top_values = feature_activations[top_indices]
    return list(zip(top_texts, top_values))


def label_features_with_gemini(
    text_samples: List[str],
    feature_index: int,
    api_key: str,
    model_name: str = "gemini-2.5-flash-preview-05-20"
) -> str:
    """
    Label SAE features with human-understandable descriptions using Google's Gemini 2.5 Flash model.

    Args:
        text_samples (List[str]): List of text samples that highly activate for a given feature
        feature_indices (int): The feature being analyzed
        api_key (str): Google AI API key
        model_name (str): Gemini model name to use

    Returns:
        str: Human-readable label for a feature
    """

    # Configure the Gemini API
    client = genai.Client(api_key=api_key)

    feature_label = ""
    prompt = f"""
    You are an expert at analyzing and labeling latent features in neural network models.
    
    I have a sparse autoencoder with various features. Feature {feature_index} is highly 
    activated by the following text sample:
    
    "{text_samples}"
    
    Based on this text sample, please provide a concise, human-understandable label for 
    this feature (1-5 words max). The label should capture the semantic concept that this 
    feature might be detecting in the text.
    
    Your response should contain ONLY the label, nothing else.
    """

    response_schema = {
        "type": "OBJECT",
        "properties": {
            "label": {"type": "STRING"}
        }
    }

    try:
        response = client.models.generate_content(
            model=model_name, contents=prompt, config={"response_schema": response_schema})
        feature_label = response.text
    except Exception as e:
        print(f"Error labeling feature {feature_index}: {e}")
        feature_label = f"Feature {feature_index}"

    return feature_label
