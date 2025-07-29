import numpy as np
import faiss

# === Config Paths ===
EMBEDDINGS_PATH = "embeddings/dino_aug_embeddings.npy"
INDEX_PATH = "embeddings/all_embeddings_index.faiss"

# === Load DINOv2 Embeddings ===
embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
assert embeddings.ndim == 2, "❌ Embeddings should be 2D (N, D)"

# === Normalize for Cosine Similarity ===
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

# === Build Cosine Similarity Index ===
index = faiss.IndexFlatIP(embeddings.shape[1])  # IP = Inner Product → Cosine if normalized
index.add(embeddings)

# === Save Index ===
faiss.write_index(index, INDEX_PATH)
print("✅ Cosine-based FAISS index built and saved at:", INDEX_PATH)

