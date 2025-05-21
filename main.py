import sqlite3
import torch
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

from spindle.data.embedding import EmbeddingManager
from spindle.models.autoencoder import TopKSAE
from spindle.models.trainer import (
    train_sae, init_orthogonal_weights, run_sweep)
from spindle.utils.analysis import (
    compute_feature_statistics,
    analyze_reconstruction_quality,
    visualize_feature_activation,
    plot_top_features_distribution,
    label_features_with_gemini,
    get_top_activated_sources
)
from spindle.utils.weights import save_encoder_weights

# ---------------------------
# CONFIGURATION
# ---------------------------
DB_PATH = "wafer.db"
HIDDEN_DIM = 3600
EPOCHS = 30
WARMUP_EPOCHS = 5
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDINGS_OUTPUT_PATH = "vectors.npy"
K = 800
TOP_X = 10

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

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


def deduplicate_embeddings(embeddings: np.ndarray, texts: list[str], threshold=0.98):
    sim_matrix = cosine_similarity(embeddings)
    keep = []
    seen = set()
    for i in range(len(texts)):
        if i in seen:
            continue
        keep.append(i)
        for j in range(i + 1, len(texts)):
            if sim_matrix[i, j] > threshold:
                seen.add(j)
    dedup_texts = [texts[i] for i in keep]
    dedup_embeddings = embeddings[keep]
    return dedup_texts, dedup_embeddings


texts = load_texts_from_db(DB_PATH)
print(f"✅ Loaded {len(texts)} texts from wafer.db")

# ---------------------------
# STEP 2: Generate Embeddings
# ---------------------------
st_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
manager = EmbeddingManager(embedding_model=st_model)
embeddings = manager.encode_texts(texts)

texts, embeddings = deduplicate_embeddings(embeddings, texts)
print(f"🧹 Deduplicated to {len(texts)} unique texts")

manager.embeddings = embeddings
manager.save_embeddings(EMBEDDINGS_OUTPUT_PATH)
print(f"✅ Generated and saved embeddings: {EMBEDDINGS_OUTPUT_PATH}")

# ---------------------------
# STEP 3: Hyperparameter Sweep
# ---------------------------
embedding_tensor = torch.tensor(embeddings, dtype=torch.float32)
input_dim = embedding_tensor.shape[1]

sweep_config = {
    "sparsity_weight": [1e-3, 5e-3],
    "decorrelation_weight": [0.0, 1e-2],
    "noise_std": [0.01, 0.05],
    "neg_loss_weight": [0.005, 0.01],
    "lr": [1e-4, 5e-4],
    "warmup_epochs": [0, 5],
}

fixed_params = {
    "hidden_dim": HIDDEN_DIM,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "lr": LEARNING_RATE,
    "k": K,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

sweep_results = run_sweep(
    model_class=TopKSAE,
    data=embedding_tensor,
    sweep_config=sweep_config,
    fixed_params=fixed_params
)

best = sorted(sweep_results, key=lambda x: x["final_loss"])[0]
best_params = best["params"]
print(f"🏆 Best config: {best_params} → Final Loss: {best['final_loss']:.6f}")

best_model = TopKSAE(input_dim=input_dim, hidden_dim=HIDDEN_DIM, k=K)
best_model.apply(init_orthogonal_weights)
train_sae(
    model=best_model,
    data=embedding_tensor,
    epochs=EPOCHS,
    warmup_epochs=WARMUP_EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    sparsity_weight=best_params["sparsity_weight"],
    decorrelation_weight=best_params["decorrelation_weight"],
    save_path="sae_model.pt",
    verbose=True
)
print("✅ Best model saved to sae_model.pt")

# Reload model for downstream use
model = TopKSAE(input_dim=input_dim, hidden_dim=HIDDEN_DIM, k=K)
model.load("sae_model.pt")
model.eval()

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

max_strength = np.max(z_np, axis=0)
feature_idx = np.argsort(-max_strength)[2]

fig1 = visualize_feature_activation(z_np, feature_idx)
fig2 = plot_top_features_distribution(stats)
plt.show()

# ---------------------------
# STEP 6: Save Encoder Weights
# ---------------------------
save_encoder_weights(model, "encoder_weights.npz")
print("📂 Encoder weights saved.")

# ---------------------------
# STEP 7: Create human-readable labels
# ---------------------------
top_feature_indices = np.argsort(-max_strength)[:TOP_X]
print(f"🧠 Labeling top {TOP_X} features by max activation...\n")

labeled_features = {}
for feature_idx in top_feature_indices:
    print(f"🧠 Labeling feature {feature_idx}...")
    texts_for_feature = get_top_activated_sources(
        z_np, texts, feature_idx, top_k=10)
    label = label_features_with_gemini(
        texts_for_feature, feature_idx, api_key=GEMINI_API_KEY)
    labeled_features[str(feature_idx)] = {
        "label": label,
        "examples": [(text, float(score)) for text, score in texts_for_feature],
        "max_activation": float(max_strength[feature_idx])
    }

with open("feature_labels.json", "w") as f:
    json.dump(labeled_features, f, indent=2)
print("🔖 Feature labels written to feature_labels.json")
