from spindle.utils.server import SaeServer
from transformers import AutoTokenizer, AutoModel

# === Load your SAE encoder weights (after training)
ENCODER_WEIGHTS_PATH = "encoder_weights.npz"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# === Load tokenizer and embedding model
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)

# === Initialize server
server = SaeServer(
    encoder_weights=ENCODER_WEIGHTS_PATH,
    tokenizer=tokenizer,
    embedding_model=embedding_model,
    host="0.0.0.0",
    port=8000,
)

# === Run it
server.run()
