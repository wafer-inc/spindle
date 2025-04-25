# Spindle Library Structure

This document provides an overview of how the Spindle library is structured.

## Directory Structure

```
spindle/
├── __init__.py             # Package initialization
├── data/                   # Data handling utilities
│   ├── __init__.py
│   └── embedding.py        # Embedding management
├── models/                 # Model implementations
│   ├── __init__.py
│   ├── autoencoder.py      # SAE model definition
│   └── trainer.py          # Training utilities
└── utils/                  # Utility functions
    ├── __init__.py
    ├── analysis.py         # Feature analysis tools
    ├── server.py           # Serving utilities
    └── weights.py          # Weight management
```

## Module Overview

### spindle.models

This module contains the core model implementations:

- `autoencoder.py`: Defines the Sparse Autoencoder (SAE) model
- `trainer.py`: Provides utilities for training SAE models

### spindle.data

This module handles data preparation and management:

- `embedding.py`: Manages embeddings for SAE training and inference

### spindle.utils

This module contains various utility functions:

- `analysis.py`: Tools for analyzing SAE features and behavior
- `server.py`: FastAPI server implementation for deploying SAE models
- `weights.py`: Utilities for working with model weights

## Key Components

### SAE Model

The SAE model is implemented in `models/autoencoder.py`. It's a simple autoencoder with a ReLU activation to induce sparsity, consisting of:

- A linear encoder (input_dim → hidden_dim)
- A ReLU activation function
- A linear decoder (hidden_dim → input_dim)

### Training Utilities

The training utilities in `models/trainer.py` provide:

- A `train_sae` function for training SAE models
- A `calculate_sparsity` function for evaluating sparsity metrics

### Embedding Management

The `EmbeddingManager` class in `data/embedding.py` handles:

- Loading and saving embeddings
- Converting embeddings to PyTorch datasets
- Extracting embeddings from databases
- Encoding text with embedding models

### Analysis Tools

The `utils/analysis.py` module provides tools for:

- Computing feature statistics
- Analyzing reconstruction quality
- Visualizing feature activations
- Plotting feature distributions

### Weight Utilities

The `utils/weights.py` module provides utilities for:

- Extracting encoder weights
- Saving and loading weights
- Initializing models from saved weights

### Server Implementation

The `SaeServer` class in `utils/server.py` implements a FastAPI server for:

- Serving SAE models via HTTP
- Providing APIs for analyzing text with SAE features