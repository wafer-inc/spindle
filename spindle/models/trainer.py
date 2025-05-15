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
    save_path=None,
    warmup_epochs=0
):
    """
    Train a Sparse Autoencoder (SAE or TopKSAE) model.

    Args:
        model (nn.Module): The SAE model to train (supports SAE or TopKSAE)
        data (torch.Tensor): Input data tensor of shape (num_samples, input_dim)
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        lr (float): Learning rate.
        sparsity_weight (float): L1 regularization weight (ignored for TopKSAE).
        device (str): Device to train on.
        verbose (bool): Print progress.
        save_path (str): Optional save path.
        warmup_epochs (int): Epochs to gradually scale λ (ignored for TopKSAE).

    Returns:
        dict: Training stats
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    data = data.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.0, 0.999))
    criterion = nn.MSELoss()
    loader = DataLoader(TensorDataset(
        data), batch_size=batch_size, shuffle=True)

    is_topk = hasattr(model, 'k')

    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0

        if not is_topk:
            if warmup_epochs > 0:
                curr_lambda = sparsity_weight * min(1.0, epoch / warmup_epochs)
            else:
                curr_lambda = sparsity_weight

        for (batch,) in loader:
            z, recon = model(batch)
            recon_loss = criterion(recon, batch)

            if is_topk:
                loss = recon_loss  # No L1 penalty
            else:
                l1_loss = torch.mean(torch.abs(z))
                loss = recon_loss + curr_lambda * l1_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / batch_count
        losses.append(avg_loss)

        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

    if save_path:
        model.save(save_path)

    return {
        'loss_history': losses,
        'final_loss': losses[-1]
    }


def init_orthogonal_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def calculate_sparsity(model, data, batch_size=256, device=None):
    """
    Compute neuron activation sparsity stats.

    Returns:
        dict: {
            mean_sparsity, dead_neurons, dead_ratio, neuron_activation_rates
        }
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
            active = (z > 0).float()
            active_per_sample += active.sum(dim=0)
            total_samples += batch.size(0)

    activation_frequency = active_per_sample / total_samples
    mean_sparsity = 1.0 - activation_frequency.mean().item()
    dead_neurons = (activation_frequency == 0).sum().item()
    dead_ratio = dead_neurons / total_neurons

    return {
        'mean_sparsity': mean_sparsity,
        'dead_neurons': dead_neurons,
        'dead_ratio': dead_ratio,
        'neuron_activation_rates': activation_frequency.cpu().numpy()
    }
