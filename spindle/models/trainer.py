import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


def train_sae(
    model,
    data,
    epochs=10,
    batch_size=128,
    lr=1e-3,
    sparsity_weight=1e-3,
    decorrelation_weight=1e-2,
    noise_std=0.05,
    neg_loss_weight=0.01,
    device=None,
    verbose=True,
    save_path=None,
    warmup_epochs=0
):
    """
    Train a Sparse Autoencoder (SAE or TopKSAE) with denoising + decorrelation.

    Args:
        model: SAE or TopKSAE instance
        data: torch.Tensor of shape (N, D)
        epochs: int
        batch_size: int
        lr: float
        sparsity_weight: float
        decorrelation_weight: float
        noise_std: float
        device: torch.device or None
        verbose: bool
        save_path: str or None
        warmup_epochs: int

    Returns:
        dict with training stats
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Normalize the embeddings
    data = F.normalize(data, p=2, dim=1)
    data = data.to(device)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.0, 0.999))
    criterion = nn.MSELoss()
    loader = DataLoader(TensorDataset(
        data), batch_size=batch_size, shuffle=True)

    is_topk = hasattr(model, 'k')
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0

        curr_lambda = sparsity_weight
        if not is_topk and warmup_epochs > 0:
            curr_lambda *= min(1.0, epoch / warmup_epochs)

        for (batch,) in loader:
            batch = batch.to(device)

            # Denoising input
            noisy_batch = batch + noise_std * torch.randn_like(batch)

            z, recon = model(noisy_batch)
            recon_loss = criterion(recon, batch)

            loss = recon_loss

            # L1 sparsity (ignored for TopKSAE)
            if not is_topk:
                l1_loss = torch.mean(torch.abs(z))
                loss += curr_lambda * l1_loss

            # Feature decorrelation penalty
            loss += decorrelation_weight * decorrelation_loss(z)

            # Negative sampling
            neg_batch = batch[torch.randperm(batch.size(0))]
            z_neg, _ = model(neg_batch)
            neg_loss = F.mse_loss(z, z_neg.detach())
            loss += neg_loss_weight * neg_loss

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


def decorrelation_loss(z):
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / z.shape[0]
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag ** 2).sum() / z.shape[1]


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


def run_sweep(
    model_class,
    data,
    sweep_config,
    fixed_params,
):
    from itertools import product

    results = []

    # Get all combinations of hyperparams
    keys = list(sweep_config.keys())
    values = list(sweep_config.values())

    for combo in product(*values):
        sweep_params = dict(zip(keys, combo))

        model = model_class(
            input_dim=data.shape[1],
            hidden_dim=fixed_params["hidden_dim"],
            k=fixed_params["k"]
        )
        model.apply(init_orthogonal_weights)

        result = train_sae(
            model=model,
            data=data,
            epochs=fixed_params["epochs"],
            batch_size=fixed_params["batch_size"],
            lr=sweep_params.get("lr", fixed_params["lr"]),
            sparsity_weight=sweep_params.get("sparsity_weight", 0),
            decorrelation_weight=sweep_params.get("decorrelation_weight", 0),
            noise_std=sweep_params.get("noise_std", 0.05),
            neg_loss_weight=sweep_params.get("neg_loss_weight", 0.01),
            warmup_epochs=sweep_params.get("warmup_epochs", 0),
            device=fixed_params.get("device", "cuda"),
            verbose=False
        )

        results.append({
            "params": sweep_params,
            "final_loss": result["final_loss"]
        })

        print(f"✅ Sweep: {sweep_params} → Loss: {result['final_loss']:.6f}")

    return results
