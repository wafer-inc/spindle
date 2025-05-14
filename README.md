# Spindle: Sparse Autoencoder Library

Structured Projection INdex for Dense Latent Embeddings

Spindle is a Python library for training and deploying Sparse Autoencoders (SAEs). It provides a simple, flexible API for working with SAEs in PyTorch.

## Features

- Train sparse autoencoders with configurable architectures
- Analyze feature activations and reconstruction quality
- Serve SAE models via a FastAPI server
- Utilities for working with embeddings and weights
- Visualization tools for interpreting SAE features

## Installation

Install from PyPI:

```bash
pip install spindle
```

For additional features:

```bash
# For transformer model support
pip install "spindle[transformers]"

# For server components
pip install "spindle[server]"
```

Or install from source:

```bash
git clone https://github.com/wafer-inc/spindle
cd spindle
pip install -e .
```

## Quick Start

### Training an SAE

```python
import torch
import numpy as np
from spindle.models.autoencoder import SAE
from spindle.models.trainer import train_sae

# Load your data
embedding_data = np.load('vectors.npy')
data = torch.tensor(embedding_data, dtype=torch.float32)

# Create and train the model
input_dim = data.shape[1]  # Embedding dimension
hidden_dim = 500  # Sparse feature dimension

model = SAE(input_dim, hidden_dim)
train_stats = train_sae(
    model=model,
    data=data,
    epochs=10,
    batch_size=128,
    sparsity_weight=1e-3,
    save_path='sae_model.pt'
)

print(f"Training complete! Final loss: {train_stats['final_loss']}")
```

### Analyzing Features

```python
from spindle.utils.analysis import compute_feature_statistics
from torch.utils.data import DataLoader, TensorDataset

# Load model and data
model = SAE.load('sae_model.pt', input_dim, hidden_dim)
dataset = TensorDataset(data)
loader = DataLoader(dataset, batch_size=128)

# Compute statistics
stats = compute_feature_statistics(model, loader)
print(f"Dead features: {stats['dead_feature_count']} ({stats['dead_feature_ratio']:.2%})")
```

### Running a Server

```python
from transformers import AutoTokenizer, AutoModel
from spindle.utils.server import SaeServer

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embedding_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Create and run server
server = SaeServer(
    encoder_weights="encoder_weights.npz",
    tokenizer=tokenizer,
    embedding_model=embedding_model
)
server.run()
```

## Documentation

For detailed documentation and examples, see the [examples](./examples) directory.
