import sqlite3
import json
from sentence_transformers import SentenceTransformer

DB_PATH = "wafer.db"
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Connect to DB
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Select rows with existing source_texts
cursor.execute("SELECT id, source_text FROM sources WHERE source_text IS NOT NULL")
rows = cursor.fetchall()

# Embed and update
for i, (doc_id, text) in enumerate(rows):
    embedding = model.encode(text.strip(), normalize_embeddings=True).tolist()
    embedding_json = json.dumps(embedding)
    cursor.execute("UPDATE sources SET vector = ? WHERE id = ?", (embedding_json, doc_id))

    if i % 100 == 0:
        print(f"Processed {i} rows...")
        conn.commit()  # batch commit every 100

conn.commit()
conn.close()
print("✅ Embeddings regenerated and stored.")
