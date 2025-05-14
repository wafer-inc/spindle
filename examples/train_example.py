"""
Example script for training a Sparse Autoencoder.
"""

import torch
import numpy as np
from spindle.models.autoencoder import SAE
from spindle.models.trainer import train_sae
from spindle.data.embedding import EmbeddingManager
from spindle.utils.weights import save_encoder_weights
from spindle.utils.analysis import compute_feature_statistics, analyze_reconstruction_quality
from torch.utils.data import DataLoader


# Parameters
INPUT_DIM = 384  # Dimension of input vectors (embedding size)
HIDDEN_DIM = 500  # Size of the sparse feature space
BATCH_SIZE = 128
EPOCHS = 10
SPARSITY_WEIGHT = 1e-3  # L1 regularization weight

# Load data
embedding_manager = EmbeddingManager()
embeddings = embedding_manager.load_embeddings('vectors.npy')
ids = embedding_manager.load_ids('ids.npy')

# Create model
model = SAE(INPUT_DIM, HIDDEN_DIM)

# Train the model
data = torch.tensor(embeddings, dtype=torch.float32)
train_stats = train_sae(
    model=model,
    data=data,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    sparsity_weight=SPARSITY_WEIGHT,
    save_path='sae_model.pt'
)

print(f"Training complete! Final loss: {train_stats['final_loss']:.6f}")

# Evaluate the model
data_loader = DataLoader(
    embedding_manager.get_torch_dataset(), batch_size=BATCH_SIZE)
reconstruction_stats = analyze_reconstruction_quality(model, data_loader)
feature_stats = compute_feature_statistics(model, data_loader)

print("\nReconstruction Quality:")
print(f"Average MSE: {reconstruction_stats['avg_mse']:.6f}")
print(f"RMSE: {reconstruction_stats['rmse']:.6f}")

print("\nFeature Statistics:")
print(
    f"Dead features: {feature_stats['dead_feature_count']} ({feature_stats['dead_feature_ratio']:.2%})")

# Save encoder weights for use in other applications
save_encoder_weights(model, 'encoder_weights.npz')
print("\nEncoder weights saved to encoder_weights.npz")
