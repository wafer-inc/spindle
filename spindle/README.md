# Spindle

A library for training and deploying Sparse Autoencoders (SAEs).

## Basic Concepts

### What is a Sparse Autoencoder?

A Sparse Autoencoder (SAE) is a type of neural network that learns to reconstruct its input while enforcing sparsity in the hidden layer activations. This sparsity constraint helps the model learn more meaningful and interpretable features.

In Spindle, the SAE implementation consists of:

- A linear encoder that maps input data to a higher-dimensional latent space
- A ReLU activation that induces sparsity
- A linear decoder that maps the sparse latent representation back to the input space

### Why Use Sparse Autoencoders?

SAEs are particularly useful for:

- Feature discovery in high-dimensional data
- Interpretable representations
- Disentangling latent factors
- Improving transfer learning by isolating specific features

## Example Usage

```python
# Training an SAE
import torch
from spindle.models import SAE
from spindle.models.trainer import train_sae

# Load embeddings
embeddings = torch.tensor(np.load("vectors.npy"), dtype=torch.float32)

# Set dimensions
input_dim = embeddings.shape[1]
hidden_dim = 500  # Overcomplete dimension

# Create model
model = SAE(input_dim, hidden_dim)

# Train model
train_stats = train_sae(
    model=model,
    data=embeddings,
    epochs=10,
    batch_size=128,
    lr=1e-3,
    sparsity_weight=1e-3,
    save_path="sae_model.pt"
)
```

For detailed usage instructions, see the [full documentation](https://github.com/wafer-inc/spindle/blob/master/docs/usage.md).