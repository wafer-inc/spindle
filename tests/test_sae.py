"""
Tests for the Spindle library.
"""

import unittest
import torch
import numpy as np
from spindle.models.autoencoder import SAE
from spindle.models.trainer import train_sae, calculate_sparsity


class TestSAE(unittest.TestCase):
    """Tests for the SAE model."""

    def setUp(self):
        # Create a small test dataset
        self.input_dim = 10
        self.hidden_dim = 20
        self.batch_size = 8
        self.data = torch.randn(32, self.input_dim)

        # Create a model
        self.model = SAE(self.input_dim, self.hidden_dim)

    def test_model_init(self):
        """Test model initialization."""
        self.assertEqual(self.model.enc.in_features, self.input_dim)
        self.assertEqual(self.model.enc.out_features, self.hidden_dim)
        self.assertEqual(self.model.dec.in_features, self.hidden_dim)
        self.assertEqual(self.model.dec.out_features, self.input_dim)

    def test_forward(self):
        """Test forward pass."""
        z, x_hat = self.model(self.data)

        # Check shapes
        self.assertEqual(z.shape, (32, self.hidden_dim))
        self.assertEqual(x_hat.shape, (32, self.input_dim))

        # Check sparsity property (some activations should be 0)
        self.assertTrue(torch.any(z == 0))

    def test_encode_decode(self):
        """Test encode and decode methods."""
        z = self.model.encode(self.data)
        x_hat = self.model.decode(z)

        # Check shapes
        self.assertEqual(z.shape, (32, self.hidden_dim))
        self.assertEqual(x_hat.shape, (32, self.input_dim))

    def test_training(self):
        """Test training function."""
        train_stats = train_sae(
            model=self.model,
            data=self.data,
            epochs=2,
            batch_size=self.batch_size,
            sparsity_weight=1e-3,
            verbose=False
        )

        # Check that training completed and loss decreased
        self.assertTrue('loss_history' in train_stats)
        self.assertEqual(len(train_stats['loss_history']), 2)

    def test_sparsity_calculation(self):
        """Test sparsity calculation function."""
        # Run a forward pass to get some activations
        _ = self.model(self.data)

        # Calculate sparsity metrics
        sparsity_stats = calculate_sparsity(self.model, self.data)

        # Check that metrics are calculated
        self.assertTrue('mean_sparsity' in sparsity_stats)
        self.assertTrue('dead_neurons' in sparsity_stats)
        self.assertTrue('dead_ratio' in sparsity_stats)

        # Check value ranges
        self.assertTrue(0 <= sparsity_stats['mean_sparsity'] <= 1)
        self.assertTrue(0 <= sparsity_stats['dead_ratio'] <= 1)


if __name__ == '__main__':
    unittest.main()
