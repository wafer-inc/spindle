"""
Utilities for generating and managing embeddings for SAE training and inference.
"""

import torch
import numpy as np
import json
from typing import Optional, List, Union, Dict, Any
from pathlib import Path


class EmbeddingManager:
    """
    Manages the creation, storage, and retrieval of embeddings for SAE training.
    
    This class can work with various embedding models and data sources, providing
    a unified interface for preparing training data for SAEs.
    """
    
    def __init__(self, embedding_model=None):
        """
        Initialize the EmbeddingManager.
        
        Args:
            embedding_model: Optional embedding model (e.g., SentenceTransformer)
        """
        self.embedding_model = embedding_model
        self.embeddings = None
        self.ids = None
    
    def set_embedding_model(self, model):
        """
        Set the embedding model to use.
        
        Args:
            model: The embedding model to use
        """
        self.embedding_model = model
    
    def encode_texts(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """
        Encode a list of texts using the embedding model.
        
        Args:
            texts (List[str]): List of text strings to encode
            normalize (bool, optional): Whether to normalize embeddings. Defaults to True.
            
        Returns:
            np.ndarray: Array of embeddings
            
        Raises:
            ValueError: If no embedding model is set
        """
        if self.embedding_model is None:
            raise ValueError("No embedding model set. Please set a model first.")
            
        # Assuming the model has an encode method (like sentence-transformers)
        embeddings = []
        for text in texts:
            embedding = self.embedding_model.encode(text, normalize_embeddings=normalize)
            embeddings.append(embedding)
            
        return np.array(embeddings, dtype=np.float32)
    
    def load_embeddings(self, path: Union[str, Path]) -> np.ndarray:
        """
        Load embeddings from a file.
        
        Args:
            path (Union[str, Path]): Path to the .npy file containing embeddings
            
        Returns:
            np.ndarray: Array of embeddings
        """
        self.embeddings = np.load(str(path))
        return self.embeddings
    
    def load_ids(self, path: Union[str, Path]) -> np.ndarray:
        """
        Load IDs from a file.
        
        Args:
            path (Union[str, Path]): Path to the .npy file containing IDs
            
        Returns:
            np.ndarray: Array of IDs
        """
        self.ids = np.load(str(path))
        return self.ids
    
    def save_embeddings(self, path: Union[str, Path]):
        """
        Save embeddings to a file.
        
        Args:
            path (Union[str, Path]): Path to save the embeddings
            
        Raises:
            ValueError: If no embeddings are loaded
        """
        if self.embeddings is None:
            raise ValueError("No embeddings to save. Please load or create embeddings first.")
            
        np.save(str(path), self.embeddings)
    
    def save_ids(self, path: Union[str, Path]):
        """
        Save IDs to a file.
        
        Args:
            path (Union[str, Path]): Path to save the IDs
            
        Raises:
            ValueError: If no IDs are loaded
        """
        if self.ids is None:
            raise ValueError("No IDs to save. Please load IDs first.")
            
        np.save(str(path), self.ids)
    
    def get_torch_dataset(self) -> torch.utils.data.TensorDataset:
        """
        Convert the loaded embeddings to a PyTorch TensorDataset.
        
        Returns:
            torch.utils.data.TensorDataset: PyTorch dataset of embeddings
            
        Raises:
            ValueError: If no embeddings are loaded
        """
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Please load embeddings first.")
            
        tensor_data = torch.tensor(self.embeddings, dtype=torch.float32)
        return torch.utils.data.TensorDataset(tensor_data)
    
    def extract_from_database(self, db_path, query, id_field="id", vector_field="vector"):
        """
        Extract embeddings from a database.
        
        Args:
            db_path (str): Path to the SQLite database
            query (str): SQL query to execute
            id_field (str, optional): Name of the ID field. Defaults to "id".
            vector_field (str, optional): Name of the vector field. Defaults to "vector".
            
        Returns:
            tuple: (embeddings, ids)
        """
        import sqlite3
        
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Execute query
        cursor.execute(query)
        rows = cursor.fetchall()
        
        # Extract IDs and vectors
        ids = []
        vectors = []
        for row in rows:
            row_dict = {desc[0]: row[i] for i, desc in enumerate(cursor.description)}
            ids.append(row_dict[id_field])
            
            # Handle vector data (which could be a JSON string)
            vector_data = row_dict[vector_field]
            if isinstance(vector_data, str):
                vector = json.loads(vector_data)
            else:
                vector = vector_data
                
            vectors.append(vector)
        
        # Convert to arrays
        self.embeddings = np.array(vectors, dtype=np.float32)
        self.ids = np.array(ids)
        
        conn.close()
        return self.embeddings, self.ids