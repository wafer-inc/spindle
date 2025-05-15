# Spindle

A library for training and deploying Sparse Autoencoders (SAEs) in PyTorch.

**Current version:** 0.1.2

---

## Installation

Install the core package:

```bash
pip install spindle-ml
```

For transformer-based embedding support:

```bash
pip install "spindle-ml[transformers]"
```

For the FastAPI server components:

```bash
pip install "spindle-ml[server]"
```

---

## Basic Concepts

### What is a Sparse Autoencoder?

A **Sparse Autoencoder (SAE)** is a neural network that learns to reconstruct its inputs while enforcing sparsity in its hidden representation. This constraint encourages the model to discover **interpretable**, **disentangled** features.

In Spindle, an SAE consists of:

- **Encoder**: linear layer mapping inputs → high-dimensional latent
- **Sparsity**: ReLU activation (and KL-divergence penalty during training)
- **Decoder**: linear layer mapping latent → reconstructed inputs

### Why Use a Sparse Autoencoder?

- **Feature discovery** in high-dimensional or text-embedding data
- **Interpretable** latent factors
- **Improved transfer** via disentangled representations
- **Downstream retrieval** when you want explicit “concept” activations

---

## Quickstart

### 1. Train your SAE

```bash
# make sure you have your training embeddings saved as `vectors.npy`
python train_sae.py
```

**`train_sae.py`** (example):

```python
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from spindle.models.autoencoder import SAE
from spindle.models.trainer      import train_sae
from spindle.utils.weights       import save_encoder_weights

# Load your precomputed embeddings
embs   = np.load("vectors.npy")
tensor = torch.tensor(embs, dtype=torch.float32)

# Define dimensions
input_dim  = tensor.shape[1]
hidden_dim = 6000

# Build and train
model = SAE(input_dim=input_dim, hidden_dim=hidden_dim)
stats = train_sae(
    model=model,
    data=tensor,
    epochs=30,
    batch_size=128,
    lr=1e-3,
    sparsity_weight=5e-3,
    save_path="sae_model.pt"
)

print("Final loss:", stats["final_loss"])

# Export just the encoder weights for serving
save_encoder_weights(model, "encoder_weights.npz")
```

---

### 2. Serve and Visualize

After training, spin up your FastAPI server (example in `server.py`):

```bash
python server.py
```

- **`/`** serves your custom `static/index.html`
- **`/api/explain`** returns per-token SAE feature activations

You’ll need:

```python
from spindle.data.embedding      import EmbeddingManager
from spindle.utils.weights       import save_encoder_weights
from spindle.models.autoencoder  import SAE
from spindle.models.trainer      import train_sae
from spindle.utils.server        import SaeServer
```

---

## Detailed Documentation

See the full docs:

- [Usage guide](https://github.com/wafer-inc/spindle/blob/master/docs/usage.md)
- [API reference](https://github.com/wafer-inc/spindle/blob/master/docs/api_reference.md)

---

## Contributing

1. Fork the repo
2. Install dev requirements: `pip install -e .[transformers,server]`
3. Add/modify code, tests under `tests/`
4. Bump version in `setup.py` & `spindle/__init__.py`
5. Submit a PR

---

## License

MIT © 2025 Sam Hall
