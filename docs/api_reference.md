# Spindle API Reference

This document provides a detailed reference for the Spindle API.

## Models

### SAE

```python
from spindle.models.autoencoder import SAE
```

The core Sparse Autoencoder implementation.

#### Constructor

```python
SAE(input_dim, hidden_dim)
```

- `input_dim` (int): Dimensionality of the input vectors
- `hidden_dim` (int): Dimensionality of the sparse feature space

#### Methods

##### forward

```python
forward(x)
```

Forward pass through the autoencoder.

- `x` (torch.Tensor): Input tensor of shape (batch_size, input_dim)
- Returns: tuple (z, x_hat)
  - `z` (torch.Tensor): Sparse encoding (batch_size, hidden_dim)
  - `x_hat` (torch.Tensor): Reconstruction (batch_size, input_dim)

##### encode

```python
encode(x)
```

Encode inputs to sparse features.

- `x` (torch.Tensor): Input tensor of shape (batch_size, input_dim)
- Returns: torch.Tensor - Sparse encodings of shape (batch_size, hidden_dim)

##### decode

```python
decode(z)
```

Decode sparse features back to input space.

- `z` (torch.Tensor): Sparse tensor of shape (batch_size, hidden_dim)
- Returns: torch.Tensor - Reconstructed inputs of shape (batch_size, input_dim)

##### save

```python
save(path)
```

Save the model state to a file.

- `path` (str): Path to save the model

##### load (class method)

```python
SAE.load(path, input_dim, hidden_dim)
```

Load a model from a saved state.

- `path` (str): Path to the saved model state
- `input_dim` (int): Input dimensionality
- `hidden_dim` (int): Hidden dimensionality
- Returns: SAE - Loaded model instance

## Trainer

### train_sae

```python
from spindle.models.trainer import train_sae
```

```python
train_sae(model, data, epochs=10, batch_size=128, lr=1e-3, sparsity_weight=1e-3, device=None, verbose=True, save_path=None)
```

Train a Sparse Autoencoder model.

- `model` (spindle.models.autoencoder.SAE): The model to train
- `data` (torch.Tensor): Input data tensor of shape (num_samples, input_dim)
- `epochs` (int, optional): Number of training epochs. Defaults to 10.
- `batch_size` (int, optional): Training batch size. Defaults to 128.
- `lr` (float, optional): Learning rate. Defaults to 1e-3.
- `sparsity_weight` (float, optional): L1 regularization weight (λ) for sparsity. Defaults to 1e-3.
- `device` (str, optional): Device to train on ('cuda', 'cpu', etc). Defaults to None (auto-detect).
- `verbose` (bool, optional): Whether to print training progress. Defaults to True.
- `save_path` (str, optional): If provided, save the model to this path. Defaults to None.
- Returns: dict - Training stats including loss history

### calculate_sparsity

```python
from spindle.models.trainer import calculate_sparsity
```

```python
calculate_sparsity(model, data, batch_size=256, device=None)
```

Calculate the sparsity metrics of a trained SAE model on a dataset.

- `model` (spindle.models.autoencoder.SAE): The trained SAE model
- `data` (torch.Tensor): Input data tensor of shape (num_samples, input_dim)
- `batch_size` (int, optional): Batch size for processing. Defaults to 256.
- `device` (str, optional): Device to use ('cuda', 'cpu', etc). Defaults to None (auto-detect).
- Returns: dict - Sparsity metrics

## Data Utilities

### EmbeddingManager

```python
from spindle.data.embedding import EmbeddingManager
```

Manages the creation, storage, and retrieval of embeddings for SAE training.

#### Constructor

```python
EmbeddingManager(embedding_model=None)
```

- `embedding_model` (optional): Optional embedding model (e.g., SentenceTransformer)

#### Methods

##### set_embedding_model

```python
set_embedding_model(model)
```

Set the embedding model to use.

- `model`: The embedding model to use

##### encode_texts

```python
encode_texts(texts, normalize=True)
```

Encode a list of texts using the embedding model.

- `texts` (List[str]): List of text strings to encode
- `normalize` (bool, optional): Whether to normalize embeddings. Defaults to True.
- Returns: np.ndarray - Array of embeddings

##### load_embeddings

```python
load_embeddings(path)
```

Load embeddings from a file.

- `path` (Union[str, Path]): Path to the .npy file containing embeddings
- Returns: np.ndarray - Array of embeddings

##### load_ids

```python
load_ids(path)
```

Load IDs from a file.

- `path` (Union[str, Path]): Path to the .npy file containing IDs
- Returns: np.ndarray - Array of IDs

##### save_embeddings

```python
save_embeddings(path)
```

Save embeddings to a file.

- `path` (Union[str, Path]): Path to save the embeddings

##### save_ids

```python
save_ids(path)
```

Save IDs to a file.

- `path` (Union[str, Path]): Path to save the IDs

##### get_torch_dataset

```python
get_torch_dataset()
```

Convert the loaded embeddings to a PyTorch TensorDataset.

- Returns: torch.utils.data.TensorDataset - PyTorch dataset of embeddings

##### extract_from_database

```python
extract_from_database(db_path, query, id_field="id", vector_field="vector")
```

Extract embeddings from a database.

- `db_path` (str): Path to the SQLite database
- `query` (str): SQL query to execute
- `id_field` (str, optional): Name of the ID field. Defaults to "id".
- `vector_field` (str, optional): Name of the vector field. Defaults to "vector".
- Returns: tuple - (embeddings, ids)

## Analysis Utilities

### Compute Statistics

```python
from spindle.utils.analysis import compute_feature_statistics
```

```python
compute_feature_statistics(model, data_loader)
```

Compute statistics about feature activations across a dataset.

- `model`: SAE model
- `data_loader`: DataLoader containing input data
- Returns: Dict - Dictionary of feature statistics

### Analyze Reconstruction

```python
from spindle.utils.analysis import analyze_reconstruction_quality
```

```python
analyze_reconstruction_quality(model, data_loader)
```

Analyze reconstruction quality of the SAE on a dataset.

- `model`: SAE model
- `data_loader`: DataLoader containing input data
- Returns: Dict - Dictionary of reconstruction quality metrics

### Visualize Features

```python
from spindle.utils.analysis import visualize_feature_activation
```

```python
visualize_feature_activation(feature_activations, feature_idx, figsize=(12, 6))
```

Visualize the activation distribution of a specific feature.

- `feature_activations` (np.ndarray): Array of feature activations
- `feature_idx` (int): Index of the feature to visualize
- `figsize` (tuple, optional): Figure size. Defaults to (12, 6).
- Returns: Figure object

```python
from spindle.utils.analysis import plot_top_features_distribution
```

```python
plot_top_features_distribution(feature_stats, top_k=20, figsize=(14, 8))
```

Plot distribution of the top features by importance.

- `feature_stats` (dict): Feature statistics from compute_feature_statistics
- `top_k` (int, optional): Number of top features to show. Defaults to 20.
- `figsize` (tuple, optional): Figure size. Defaults to (14, 8).
- Returns: Figure object

## Weight Utilities

### Extract Weights

```python
from spindle.utils.weights import extract_encoder_weights
```

```python
extract_encoder_weights(model)
```

Extract encoder weights and biases from an SAE model.

- `model`: SAE model
- Returns: Tuple[np.ndarray, np.ndarray] - (W, b) encoder weights and biases

### Save Weights

```python
from spindle.utils.weights import save_encoder_weights
```

```python
save_encoder_weights(model, path)
```

Save encoder weights and biases to an .npz file.

- `model`: SAE model
- `path` (Union[str, Path]): Output path

### Load Weights

```python
from spindle.utils.weights import load_encoder_weights
```

```python
load_encoder_weights(path)
```

Load encoder weights from an .npz file.

- `path` (Union[str, Path]): Path to the .npz file
- Returns: Tuple[np.ndarray, np.ndarray] - (W, b) encoder weights and biases

### Initialize from Weights

```python
from spindle.utils.weights import initialize_from_encoder_weights
```

```python
initialize_from_encoder_weights(model, weights_path, freeze=False)
```

Initialize a model from saved encoder weights.

- `model`: SAE model
- `weights_path` (Union[str, Path]): Path to the weights file
- `freeze` (bool, optional): Whether to freeze the encoder parameters. Defaults to False.

## Server Utilities

### SaeServer

```python
from spindle.utils.server import SaeServer
```

FastAPI server for serving SAE models.

#### Constructor

```python
SaeServer(encoder_weights=None, tokenizer=None, embedding_model=None, host="0.0.0.0", port=8000)
```

- `encoder_weights` (Union[str, np.ndarray, torch.Tensor], optional): SAE encoder weights. Can be a path to .npz file or actual weights.
- `tokenizer`: Tokenizer for text processing.
- `embedding_model`: Model for generating embeddings from text.
- `host` (str, optional): Host to run the server on. Defaults to "0.0.0.0".
- `port` (int, optional): Port to run the server on. Defaults to 8000.

#### Methods

##### set_encoder_weights

```python
set_encoder_weights(weights)
```

Set the encoder weights for the server.

- `weights` (Union[str, np.ndarray, torch.Tensor]): Encoder weights

##### set_tokenizer

```python
set_tokenizer(tokenizer)
```

Set the tokenizer for the server.

- `tokenizer`: Tokenizer instance

##### set_embedding_model

```python
set_embedding_model(model)
```

Set the embedding model for the server.

- `model`: Embedding model instance

##### run

```python
run()
```

Run the FastAPI server.
