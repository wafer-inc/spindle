"""
Example script for analyzing SAE features.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from spindle.models.autoencoder import SAE
from spindle.data.embedding import EmbeddingManager
from spindle.utils.analysis import (
    compute_feature_statistics,
    analyze_reconstruction_quality,
    visualize_feature_activation,
    plot_top_features_distribution
)
from torch.utils.data import DataLoader


# Configuration
INPUT_DIM = 384  # Dimension of input vectors
HIDDEN_DIM = 500  # Size of the sparse feature space
MODEL_PATH = "sae_model.pt"
VECTORS_PATH = "vectors.npy"

# Load data
embedding_manager = EmbeddingManager()
embeddings = embedding_manager.load_embeddings(VECTORS_PATH)
dataset = embedding_manager.get_torch_dataset()
data_loader = DataLoader(dataset, batch_size=128)

# Load model
model = SAE.load(MODEL_PATH, INPUT_DIM, HIDDEN_DIM)

# Analyze reconstruction quality
print("Analyzing reconstruction quality...")
recon_stats = analyze_reconstruction_quality(model, data_loader)
print(f"Average MSE: {recon_stats['avg_mse']:.6f}")
print(f"RMSE: {recon_stats['rmse']:.6f}")

# Compute feature statistics
print("\nComputing feature statistics...")
feature_stats = compute_feature_statistics(model, data_loader)
print(
    f"Dead features: {feature_stats['dead_feature_count']} ({feature_stats['dead_feature_ratio']:.2%})")

# Get feature activations for all samples
print("Computing feature activations...")
all_activations = []
model.eval()
with torch.no_grad():
    for batch in data_loader:
        z, _ = model(batch[0])
        all_activations.append(z)
feature_activations = torch.cat(all_activations, dim=0).cpu().numpy()

# Visualize specific features
print("\nVisualizing feature activations...")
feature_to_visualize = 0  # Change this to visualize different features
fig = visualize_feature_activation(feature_activations, feature_to_visualize)
plt.savefig(f"feature_{feature_to_visualize}_activation.png")
print(f"Saved visualization to feature_{feature_to_visualize}_activation.png")

# Visualize top features by importance
print("\nPlotting top features by importance...")
fig = plot_top_features_distribution(feature_stats)
plt.savefig("top_features.png")
print("Saved visualization to top_features.png")

print("\nAnalysis complete!")
