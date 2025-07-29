import os
import json
import numpy as np
import pandas as pd
import torch
import timm
import faiss
from PIL import Image
from torchvision import transforms

# --- Config ---
QUERY_IMAGE = "data/mvtec_ad/zipper/train/good/001.png"
TOP_K = 10  # top-K from FAISS
TOP_FINAL = 5  # top-N after merging
project_root = os.path.abspath(".")
EMBEDDINGS_PATH = "embeddings/dino_aug_embeddings.npy"
INDEX_PATH = "embeddings/all_embeddings_index.faiss"
IMAGE_PATHS_JSON = "embeddings/image_paths.json"
METADATA_CSV = "data/metadata.csv"

# --- Load assets ---
embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
index = faiss.read_index(INDEX_PATH)
image_paths = json.load(open(IMAGE_PATHS_JSON))
metadata = pd.read_csv(METADATA_CSV)

# --- Load DINOv2 model ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True)
model.head = torch.nn.Identity()
model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def embed_image(img_path):
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model(tensor)
    vec = vec.cpu().numpy()
    vec /= np.linalg.norm(vec)
    return vec.astype("float32")

# --- Embed and search ---
query_vector = embed_image(QUERY_IMAGE)
D, I = index.search(query_vector, TOP_K)

# --- Extract metadata for Top-K ---
top_metadata = metadata.iloc[I[0]].copy()
top_metadata["visual_score"] = D[0]

# --- Metadata scoring logic ---
def compute_meta_score(row, query_cat):
    score = 0
    if row["category"].strip().lower() == query_cat.strip().lower():
        score += 0.5
    if row["material"].lower() in ["plastic","copper","gelatin","fabric","steel","alloy steel",
                                   "silicon","stainless steel","wood","metal"]:
        score += 0.2
    if "durable" in str(row["description"]).lower():
        score += 0.1
    if row["type"].lower() in ["container", "wiring", "pill","floor covering","fastener","tool",
                               "hygiene tool","electronic","spacer","material"]:
        score += 0.2
    return score

# Infer query category from filename path
query_cat = os.path.normpath(QUERY_IMAGE).split(os.sep)[-4]  # e.g., 'zipper'
top_metadata["meta_score"] = top_metadata.apply(lambda row: compute_meta_score(row, query_cat), axis=1)

# --- Final score and reranking ---
top_metadata["final_score"] = 0.6 * top_metadata["visual_score"] + 0.4 * top_metadata["meta_score"]
top_final = top_metadata.sort_values("final_score", ascending=False).head(TOP_FINAL)

# --- Display final results ---
print("\n🔝 Final Top-5 Results (Visual + Metadata merged):\n")
for i, row in top_final.iterrows():
    print(f"{i+1}. {row['image_name']} | Cat: {row['category']} | Material: {row['material']}")
    print(f"   Visual Score: {row['visual_score']:.3f} | Meta Score: {row['meta_score']:.3f} | Final: {row['final_score']:.3f}")
    print(f"   Desc: {row['description'][:80]}...\n")
