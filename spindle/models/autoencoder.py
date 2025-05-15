"""
Core implementation of the Sparse Autoencoder model.
"""

import torch
from torch import nn
import torch.nn.functional as F


class SAE(nn.Module):
    """
    Sparse Autoencoder (SAE) implementation.

    A simple autoencoder with ReLU activation on the latent space to induce sparsity,
    consisting of a linear encoder and a linear decoder.

    Args:
        input_dim (int): Dimensionality of the input vectors
        hidden_dim (int): Dimensionality of the sparse feature space

    Attributes:
        enc (nn.Linear): The encoder network (input_dim → hidden_dim)
        dec (nn.Linear): The decoder network (hidden_dim → input_dim)
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.enc = nn.Linear(input_dim, hidden_dim)
        self.dec = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            tuple: (z, x_hat) where z is the sparse encoding (batch_size, hidden_dim)
                  and x_hat is the reconstruction (batch_size, input_dim)
        """
        z = F.relu(self.enc(x))  # ReLU to induce sparsity
        x_hat = self.dec(z)
        return z, x_hat

    def encode(self, x):
        """
        Encode inputs to sparse features.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            torch.Tensor: Sparse encodings of shape (batch_size, hidden_dim)
        """
        return F.relu(self.enc(x))

    def decode(self, z):
        """
        Decode sparse features back to input space.

        Args:
            z (torch.Tensor): Sparse tensor of shape (batch_size, hidden_dim)

        Returns:
            torch.Tensor: Reconstructed inputs of shape (batch_size, input_dim)
        """
        return self.dec(z)

    def save(self, path):
        """
        Save the model state to a file.

        Args:
            path (str): Path to save the model
        """
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, input_dim, hidden_dim):
        """
        Load a model from a saved state.

        Args:
            path (str): Path to the saved model state
            input_dim (int): Input dimensionality
            hidden_dim (int): Hidden dimensionality

        Returns:
            SAE: Loaded model instance
        """
        model = cls(input_dim, hidden_dim)
        model.load_state_dict(torch.load(path))
        model.eval()
        return model


class TopKSAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, k):
        super().__init__()
        self.enc = nn.Linear(input_dim, hidden_dim)
        self.dec = nn.Linear(hidden_dim, input_dim)
        self.k = k

    def forward(self, x):
        z = self.enc(x)
        topk_vals, topk_idx = torch.topk(z, self.k, dim=1)
        mask = torch.zeros_like(z)
        mask.scatter_(1, topk_idx, 1)
        z_sparse = z * mask
        x_hat = self.dec(z_sparse)
        return z_sparse, x_hat

    def encode(self, x):
        z = self.enc(x)
        topk_vals, topk_idx = torch.topk(z, self.k, dim=1)
        mask = torch.zeros_like(z)
        mask.scatter_(1, topk_idx, 1)
        return z * mask

    def decode(self, z):
        return self.dec(z)

    def save(self, path):
        torch.save({
            'state_dict': self.state_dict(),
            'input_dim': self.enc.in_features,
            'hidden_dim': self.enc.out_features,
            'k': self.k,
        }, path)

    @classmethod
    def load(cls, path):
        obj = torch.load(path)
        model = cls(obj['input_dim'], obj['hidden_dim'], obj['k'])
        model.load_state_dict(obj['state_dict'])
        model.eval()
        return model
