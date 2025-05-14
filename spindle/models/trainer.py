"""
Training utilities for Sparse Autoencoders.
"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


def train_sae(
    model,
    data,
    epochs=10,
    batch_size=128,
    lr=1e-3,
    sparsity_weight=1e-3,
    device=None,
    verbose=True,
    save_path=None
):
    """
    Train a Sparse Autoencoder model.

    Args:
        model (spindle.models.autoencoder.SAE): The model to train
        data (torch.Tensor): Input data tensor of shape (num_samples, input_dim)
        epochs (int, optional): Number of training epochs. Defaults to 10.
        batch_size (int, optional): Training batch size. Defaults to 128.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        sparsity_weight (float, optional): L1 regularization weight (λ) for sparsity. Defaults to 1e-3.
        device (str, optional): Device to train on ('cuda', 'cpu', etc). Defaults to None (auto-detect).
        verbose (bool, optional): Whether to print training progress. Defaults to True.
        save_path (str, optional): If provided, save the model to this path. Defaults to None.

    Returns:
        dict: Training stats including loss history
    """
    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Move model and data to device
    model = model.to(device)
    data = data.to(device)

    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loader = DataLoader(TensorDataset(
        data), batch_size=batch_size, shuffle=True)

    # Training loop
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0

        for (batch,) in loader:
            # Forward pass
            z, recon = model(batch)

            # Reconstruction loss + L1 sparsity regularization
            loss = criterion(recon, batch) + sparsity_weight * \
                torch.mean(torch.abs(z))

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track loss
            epoch_loss += loss.item()
            batch_count += 1

        # Log progress
        epoch_avg_loss = epoch_loss / batch_count
        losses.append(epoch_avg_loss)

        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_avg_loss:.6f}")

    # Save model if requested
    if save_path:
        model.save(save_path)

    # Return training stats
    return {
        'loss_history': losses,
        'final_loss': losses[-1]
    }


def calculate_sparsity(model, data, batch_size=256, device=None):
    """
    Calculate the sparsity metrics of a trained SAE model on a dataset.

    Args:
        model (spindle.models.autoencoder.SAE): The trained SAE model
        data (torch.Tensor): Input data tensor of shape (num_samples, input_dim)
        batch_size (int, optional): Batch size for processing. Defaults to 256.
        device (str, optional): Device to use ('cuda', 'cpu', etc). Defaults to None (auto-detect).

    Returns:
        dict: Sparsity metrics
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    model.eval()

    loader = DataLoader(TensorDataset(
        data), batch_size=batch_size, shuffle=False)

    total_neurons = model.enc.weight.shape[0]
    active_per_sample = torch.zeros(total_neurons, device=device)
    total_samples = 0

    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            z = model.encode(batch)

            # Count activated neurons (binary)
            active = (z > 0).float()
            active_per_sample += active.sum(dim=0)
            total_samples += batch.size(0)

    # Average activation frequency per neuron
    activation_frequency = active_per_sample / total_samples

    # Metrics
    mean_sparsity = 1.0 - (activation_frequency.mean().item())
    dead_neurons = (activation_frequency == 0).sum().item()
    dead_ratio = dead_neurons / total_neurons

    return {
        'mean_sparsity': mean_sparsity,
        'dead_neurons': dead_neurons,
        'dead_ratio': dead_ratio,
        'neuron_activation_rates': activation_frequency.cpu().numpy()
    }
