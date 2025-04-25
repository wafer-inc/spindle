import sqlite3
import numpy as np
import json

conn = sqlite3.connect('wafer.db')
cursor = conn.cursor()

cursor.execute("SELECT id, vector FROM sources WHERE vector IS NOT NULL")
rows = cursor.fetchall()

ids = []
vectors = []
for row in rows:
    ids.append(row[0])
    vectors.append(json.loads(row[1]))  # assuming the vector is stored as a JSON array

vectors_np = np.array(vectors, dtype=np.float32)
np.save('vectors.npy', vectors_np)
np.save('ids.npy', np.array(ids))
