"""
Example script for running an SAE server.
"""

from transformers import AutoTokenizer, AutoModel
from spindle.utils.server import SaeServer


# Configuration
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_WEIGHTS_PATH = "encoder_weights.npz"
HOST = "0.0.0.0"
PORT = 8000

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

# Create and run server
server = SaeServer(
    encoder_weights=ENCODER_WEIGHTS_PATH,
    tokenizer=tokenizer,
    embedding_model=model,
    host=HOST,
    port=PORT
)

print(f"Starting SAE server on http://{HOST}:{PORT}")
print("Endpoints:")
print("  POST /explain - Analyze text and explain features")

server.run()