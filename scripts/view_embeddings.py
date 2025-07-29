import numpy as np

# Load the embeddings
embeddings = np.load("./embeddings/dino_aug_embeddings.npy")

# View shape
print(f"Embedding shape: {embeddings.shape}")  # Should be (3629, 512)

# View sample values
print("\n🔹 First embedding vector (truncated):")
print(embeddings[0][:10])  # Print first 10 elements of the first vector