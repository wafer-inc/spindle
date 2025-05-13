import sqlite3
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from spindle.data.embedding import EmbeddingManager
from spindle.models.autoencoder import SAE
from spindle.models.trainer import train_sae
from spindle.utils.analysis import (
    compute_feature_statistics,
    analyze_reconstruction_quality,
    visualize_feature_activation,
    plot_top_features_distribution
)
from spindle.utils.weights import save_encoder_weights

# ---------------------------
# CONFIGURATION
# ---------------------------
DB_PATH = "wafer.db"
HIDDEN_DIM = 6000
EPOCHS = 30
SPARSITY_WEIGHT = 5e-3
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDINGS_OUTPUT_PATH = "vectors.npy"

# ---------------------------
# STEP 1: Load Text from SQLite
# ---------------------------


def load_texts_from_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT source_text FROM sources WHERE source_text IS NOT NULL")
    texts = [row[0] for row in cursor.fetchall()]
    conn.close()
    return texts


texts = load_texts_from_db(DB_PATH)
print(f"✅ Loaded {len(texts)} texts from wafer.db")

# ---------------------------
# STEP 2: Generate Embeddings
# ---------------------------
st_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
manager = EmbeddingManager(embedding_model=st_model)
embeddings = manager.encode_texts(texts)
manager.embeddings = embeddings
manager.save_embeddings(EMBEDDINGS_OUTPUT_PATH)
print(f"✅ Generated and saved embeddings: {EMBEDDINGS_OUTPUT_PATH}")

# ---------------------------
# STEP 3: Train the SAE
# ---------------------------
embedding_tensor = torch.tensor(embeddings, dtype=torch.float32)
input_dim = embedding_tensor.shape[1]
model = SAE(input_dim=input_dim, hidden_dim=HIDDEN_DIM)

train_stats = train_sae(
    model=model,
    data=embedding_tensor,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    sparsity_weight=SPARSITY_WEIGHT,
    save_path="sae_model.pt"
)

print(f"✅ SAE trained. Final loss: {train_stats['final_loss']}")

# ---------------------------
# STEP 4: Analyze the Model
# ---------------------------
loader = DataLoader(TensorDataset(embedding_tensor), batch_size=BATCH_SIZE)
stats = compute_feature_statistics(model, loader)

print(
    f"🔍 Dead features: {stats['dead_feature_count']} ({stats['dead_feature_ratio']:.2%})")
print(
    f"🔍 Avg activation frequency: {stats['activation_frequency'].mean():.4f}")

recon_stats = analyze_reconstruction_quality(model, loader)
print(f"🔍 Avg reconstruction error (MSE): {recon_stats['avg_mse']:.6f}")

# ---------------------------
# STEP 5: Visualize Activations
# ---------------------------
model.eval()
with torch.no_grad():
    z_all, _ = model(embedding_tensor)
z_np = z_all.cpu().numpy()

feature_idx = 42  # Change this to explore other features
fig1 = visualize_feature_activation(z_np, feature_idx)
fig2 = plot_top_features_distribution(stats)
plt.show()

# ---------------------------
# STEP 6: Save Encoder Weights
# ---------------------------
save_encoder_weights(model, "encoder_weights.npz")
print("💾 Encoder weights saved.")
