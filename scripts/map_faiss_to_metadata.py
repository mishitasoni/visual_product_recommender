import os
import json
import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import faiss

# === Config ===
INDEX_PATH = "embeddings/all_embeddings_index.faiss"
IMAGE_PATHS_JSON = "embeddings/image_paths.json"
METADATA_CSV = "data/metadata.csv"
QUERY_IMAGE = "data/mvtec_ad/screw/train/good/001.png"
TOP_K = 5

# === Load Assets ===
index = faiss.read_index(INDEX_PATH)
image_paths = json.load(open(IMAGE_PATHS_JSON))
metadata_df = pd.read_csv(METADATA_CSV)
project_root = os.path.abspath(".")

print("✅ Unique categories in metadata:", metadata_df["category"].unique())

# === Load DINOv2 Model ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True)
model.head = torch.nn.Identity()
model.eval().to(device)

# === Preprocessing for DINOv2 (518×518) ===
preprocess = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# === Query Embedding ===
def embed_image(img_path):
    img = Image.open(img_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model(tensor).cpu().numpy()
    vec /= np.linalg.norm(vec)  # cosine normalization
    assert vec.shape[1] == index.d, f"Embedding shape mismatch: {vec.shape} vs {index.d}"
    return vec.astype("float32")

query_vec = embed_image(QUERY_IMAGE)
D, I = index.search(query_vec, TOP_K)

# === Normalize Paths for Matching ===
def normalize_path(path):
    return os.path.normpath(os.path.relpath(path, start=project_root))

results = [{"title": "Query Image", "path": QUERY_IMAGE}]

# === Map and Display Results ===
for rank, idx in enumerate(I[0]):
    print(f"\n🔎 Result {rank+1}")
    if idx >= len(image_paths):
        print("❌ Invalid index:", idx)
        continue

    img_path = os.path.normpath(image_paths[idx])
    rel_path = normalize_path(img_path)

    match = metadata_df[metadata_df["path"].apply(os.path.normpath) == rel_path]

    if match.empty:
        title = f"{rank+1}. ⚠️ Metadata Not Found"
        print("🚫 No metadata match for:", rel_path)
    else:
        row = match.iloc[0]
        sim = D[0][rank]
        title = f"{rank+1}. {row['category']} ({os.path.basename(img_path)})\nSim: {sim:.3f}"
        print("✅ Matched metadata:", row.to_dict())

    results.append({"title": title, "path": img_path})

# === Display Query and Top Matches ===
plt.figure(figsize=(4 * len(results), 4))
for i, item in enumerate(results):
    try:
        img = Image.open(item["path"]).convert("RGB")
        plt.subplot(1, len(results), i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(item["title"], fontsize=10)
    except Exception as e:
        print(f"❌ Could not open {item['path']}: {e}")

plt.tight_layout()
plt.show()
