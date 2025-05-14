# Spindle Usage Guide

This guide provides detailed instructions for using the Spindle library for training and working with Sparse Autoencoders (SAEs).

## Installation

Install the library using pip:

```bash
pip install spindle
```

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

## Training an SAE

### Preparing Your Data

First, you'll need to prepare your input data. Typically, this involves creating embeddings from raw input:

```python
from spindle.data.embedding import EmbeddingManager
from sentence_transformers import SentenceTransformer

# Initialize embedding model
st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Create embedding manager
manager = EmbeddingManager(embedding_model=st_model)

# Generate embeddings from text
texts = ["This is a sample text", "Another example sentence", ...]
embeddings = manager.encode_texts(texts)

# Save embeddings for later use
manager.embeddings = embeddings
manager.save_embeddings("vectors.npy")
```

### Training a Model

Once you have your embeddings, you can train an SAE:

```python
import torch
from spindle.models.autoencoder import SAE
from spindle.models.trainer import train_sae

# Load embeddings
embeddings = torch.tensor(np.load("vectors.npy"), dtype=torch.float32)

# Set dimensions
input_dim = embeddings.shape[1]  # Dimension of your embeddings
hidden_dim = 500  # Overcomplete dimension (typically larger than input_dim)

# Create model
model = SAE(input_dim, hidden_dim)

# Train model
train_stats = train_sae(
    model=model,
    data=embeddings,
    epochs=10,
    batch_size=128,
    lr=1e-3,
    sparsity_weight=1e-3,  # Controls how sparse the representation should be
    save_path="sae_model.pt"
)

print(f"Final training loss: {train_stats['final_loss']}")
```

## Analyzing SAE Features

After training, you can analyze the learned features:

```python
from spindle.utils.analysis import compute_feature_statistics
from torch.utils.data import DataLoader, TensorDataset

# Create a dataloader
dataset = TensorDataset(embeddings)
loader = DataLoader(dataset, batch_size=128)

# Compute statistics
stats = compute_feature_statistics(model, loader)

# Print statistics
print(f"Dead features: {stats['dead_feature_count']} ({stats['dead_feature_ratio']:.2%})")
print(f"Average activation frequency: {stats['activation_frequency'].mean():.4f}")

# Analyze reconstruction quality
from spindle.utils.analysis import analyze_reconstruction_quality
recon_stats = analyze_reconstruction_quality(model, loader)
print(f"Average reconstruction error (MSE): {recon_stats['avg_mse']:.6f}")
```

## Visualizing Features

Spindle includes tools for visualizing learned features:

```python
import matplotlib.pyplot as plt
from spindle.utils.analysis import visualize_feature_activation, plot_top_features_distribution

# Get feature activations
activations = []
model.eval()
with torch.no_grad():
    for batch in loader:
        z, _ = model(batch[0])
        activations.append(z)
all_activations = torch.cat(activations, dim=0).cpu().numpy()

# Visualize a specific feature
feature_idx = 42  # Replace with feature index of interest
fig = visualize_feature_activation(all_activations, feature_idx)
plt.show()

# Plot top features by importance
fig = plot_top_features_distribution(stats)
plt.show()
```

## Extracting Encoder Weights

For deployment or further analysis, you can extract the encoder weights:

```python
from spindle.utils.weights import save_encoder_weights

# Save encoder weights
save_encoder_weights(model, "encoder_weights.npz")
```

## Deploying an SAE Server

Spindle includes utilities for serving SAE models via FastAPI:

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
    embedding_model=embedding_model,
    host="0.0.0.0",
    port=8000
)

# Start the server
server.run()
```

Then you can query it:

```bash
curl -X POST "http://localhost:8000/explain" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a sample text to analyze."}'
```

## Advanced Usage

### Custom Training Loops

If you need more control over the training process, you can create your own training loop:

```python
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

# Create model and setup training
model = SAE(input_dim, hidden_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
loader = DataLoader(TensorDataset(embeddings), batch_size=128, shuffle=True)

# Custom training loop
for epoch in range(10):
    epoch_loss = 0.0
    batch_count = 0

    for (batch,) in loader:
        # Forward pass
        z, recon = model(batch)

        # Compute loss with L1 regularization for sparsity
        loss = criterion(recon, batch) + 1e-3 * torch.mean(torch.abs(z))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        batch_count += 1

    print(f"Epoch {epoch+1}, Loss: {epoch_loss/batch_count:.6f}")
```

### Working with Database Sources

If your embeddings are stored in a database, you can use the EmbeddingManager to extract them:

```python
from spindle.data.embedding import EmbeddingManager

manager = EmbeddingManager()
manager.extract_from_database(
    db_path="your_database.db",
    query="SELECT id, vector FROM sources WHERE vector IS NOT NULL",
    id_field="id",
    vector_field="vector"
)

# Now you can access the data
embeddings = manager.embeddings
ids = manager.ids
```

## Best Practices

1. **Dimensionality**: For best results, make the hidden dimension larger than the input dimension (overcomplete).

2. **Sparsity Regularization**: Adjust the sparsity weight (`sparsity_weight`) to control how sparse the representations should be. Start with a small value like 1e-3 and adjust based on results.

3. **Data Preprocessing**: Normalize input embeddings for best results (many embedding models already output normalized vectors).

4. **Model Evaluation**: Regularly check:

   - Reconstruction error (should decrease during training)
   - Sparsity levels (% of inactive neurons)
   - Dead neuron count (neurons that never activate)

5. **Weight Saving**: Save both the full model and just the encoder weights for deployment.
