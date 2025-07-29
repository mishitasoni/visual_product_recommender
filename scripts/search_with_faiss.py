import os
import numpy as np
import pandas as pd
import torch
import faiss
from PIL import Image
import matplotlib.pyplot as plt
import timm
from torchvision import transforms

# === CONFIG ===
EMBEDDING_PATH = "embeddings/dino_aug_embeddings.npy"
METADATA_PATH = "data/metadata.csv"
QUERY_IMAGE = "data/mvtec_ad/zipper/train/good/001.png"
TOP_K = 5

# === LOAD EMBEDDINGS ===
print("📦 Loading DINOv2 embeddings...")
embeddings = np.load(EMBEDDING_PATH).astype("float32")
print(f"✅ Loaded {embeddings.shape[0]} embeddings with dim {embeddings.shape[1]}")

# === Normalize for cosine similarity ===
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

# === FAISS Index (Inner Product) ===
print("🔍 Building FAISS index using cosine similarity...")
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
print(f"✅ FAISS index ready with {index.ntotal} vectors.")

# === LOAD METADATA ===
metadata = pd.read_csv(METADATA_PATH)
if "path" not in metadata.columns:
    raise ValueError("Metadata CSV must contain a 'path' column with image paths.")

# === LOAD DINOv2 MODEL ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print("⚙️ Loading DINOv2 model...")
model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True)
model.head = torch.nn.Identity()
model.eval().to(device)

# === TRANSFORM for DINOv2 (518x518) ===
preprocess = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# === FUNCTION TO EMBED QUERY IMAGE ===
def get_query_vector(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model(tensor)
    vec = vec.cpu().numpy()
    vec /= np.linalg.norm(vec)
    return vec.astype("float32")

# === EMBED QUERY & SEARCH ===
query_vector = get_query_vector(QUERY_IMAGE)
D, I = index.search(query_vector, TOP_K)

# === VISUALIZE RESULTS ===
print("\n🎯 Top similar results:\n")
for rank, idx in enumerate(I[0]):
    img_path = metadata.iloc[idx]["path"]
    label = f"{rank+1}: {os.path.basename(img_path)}\nCosine Sim: {D[0][rank]:.3f}"

    print(label)
    try:
        img = Image.open(img_path)
        plt.imshow(img)
        plt.title(label)
        plt.axis("off")
        plt.show()
    except Exception as e:
        print(f"❌ Failed to load image {img_path}: {e}")

