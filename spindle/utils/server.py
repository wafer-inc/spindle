"""
Server utilities for deploying SAE models.
"""

from fastapi.staticfiles import StaticFiles
import torch
import numpy as np
import fastapi
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse


class TextInput(BaseModel):
    """Input model for text explanation API."""
    text: str


class SaeServer:
    """
    FastAPI server for serving SAE models.

    This class provides utilities for setting up a FastAPI server
    that can analyze text inputs with an SAE.
    """

    def __init__(
        self,
        encoder_weights=None,
        tokenizer=None,
        embedding_model=None,
        host="0.0.0.0",
        port=8000
    ):
        """
        Initialize the SAE server.

        Args:
            encoder_weights (Union[str, np.ndarray, torch.Tensor], optional): 
                SAE encoder weights. Can be a path to .npz file or actual weights.
            tokenizer: Tokenizer for text processing.
            embedding_model: Model for generating embeddings from text.
            host (str, optional): Host to run the server on. Defaults to "0.0.0.0".
            port (int, optional): Port to run the server on. Defaults to 8000.
        """
        self.host = host
        self.port = port
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model

        # Load encoder weights
        if isinstance(encoder_weights, str):
            encoder_data = np.load(encoder_weights)
            self.W = torch.tensor(encoder_data["W"], dtype=torch.float32)
        elif isinstance(encoder_weights, np.ndarray):
            self.W = torch.tensor(encoder_weights, dtype=torch.float32)
        elif isinstance(encoder_weights, torch.Tensor):
            self.W = encoder_weights
        elif encoder_weights is None:
            self.W = None

        # Initialize FastAPI app
        self.app = fastapi.FastAPI()
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"]
        )

        # Register routes
        if self.W is not None and self.tokenizer is not None and self.embedding_model is not None:
            self._setup_routes()

        self.app.mount(
            "/static", StaticFiles(directory="static"), name="static")

    def _setup_routes(self):
        """Set up API routes."""
        @self.app.get("/")
        def root():
            return FileResponse("static/index.html")

        @self.app.post("/explain")
        def explain(data: TextInput):
            # Tokenize input
            inputs = self.tokenizer(
                data.text, return_tensors="pt", add_special_tokens=True)
            tokens = self.tokenizer.convert_ids_to_tokens(
                inputs["input_ids"][0])

            # Get token embeddings
            with torch.no_grad():
                token_embs = self.embedding_model(
                    **inputs).last_hidden_state.squeeze(0)

            # Dot with encoder weights
            scores = torch.matmul(token_embs, self.W.T)

            # Transpose to: feature_id → [scores per token]
            feature_scores = scores.T.tolist()

            # Feature strengths: max score across all tokens for each feature
            feature_strengths = {str(i): max(map(abs, s))
                                 for i, s in enumerate(feature_scores)}

            # Reformat feature_scores as {feature_id: [score, score, ...]}
            feature_scores_dict = {
                str(i): s for i, s in enumerate(feature_scores)}

            return {
                "tokens": tokens,
                "feature_scores": feature_scores_dict,
                "feature_strengths": feature_strengths
            }

    def set_encoder_weights(self, weights):
        """
        Set the encoder weights for the server.

        Args:
            weights (Union[str, np.ndarray, torch.Tensor]): Encoder weights
        """
        if isinstance(weights, str):
            encoder_data = np.load(weights)
            self.W = torch.tensor(encoder_data["W"], dtype=torch.float32)
        elif isinstance(weights, np.ndarray):
            self.W = torch.tensor(weights, dtype=torch.float32)
        elif isinstance(weights, torch.Tensor):
            self.W = weights

        # Reinitialize routes if all components are available
        if self.W is not None and self.tokenizer is not None and self.embedding_model is not None:
            self._setup_routes()

    def set_tokenizer(self, tokenizer):
        """
        Set the tokenizer for the server.

        Args:
            tokenizer: Tokenizer instance
        """
        self.tokenizer = tokenizer

        # Reinitialize routes if all components are available
        if self.W is not None and self.tokenizer is not None and self.embedding_model is not None:
            self._setup_routes()

    def set_embedding_model(self, model):
        """
        Set the embedding model for the server.

        Args:
            model: Embedding model instance
        """
        self.embedding_model = model

        # Reinitialize routes if all components are available
        if self.W is not None and self.tokenizer is not None and self.embedding_model is not None:
            self._setup_routes()

    def run(self):
        """Run the FastAPI server."""
        import uvicorn
        uvicorn.run(self.app, host=self.host, port=self.port)
