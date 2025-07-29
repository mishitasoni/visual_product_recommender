import os
import torch
import timm
import numpy as np
import json
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# === Config ===
BASE_DIR = "data/mvtec_ad"
SAVE_EMBEDDINGS_PATH = "embeddings/dino_aug_embeddings.npy"
SAVE_IMAGE_PATHS_JSON = "embeddings/image_paths.json"
AUGMENTATIONS_PER_IMAGE = 5
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Load DINOv2 Base Model ===
print("📦 Loading DINOv2 base model...")
model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True)
model.head = torch.nn.Identity()
model.eval().to(device)

# === Augmentation pipeline ===
print("⚙️ Setting up augmentation pipeline...")
augmentation_pipeline = transforms.Compose([
    transforms.Resize((518, 518)),  # ✅ Required by DINOv2 model
    transforms.RandomResizedCrop(518, scale=(0.8, 1.0), ratio=(0.95, 1.05)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.01),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# === Feature extraction ===
def extract_features(img_path, augmentations=AUGMENTATIONS_PER_IMAGE):
    try:
        img = Image.open(img_path).convert("RGB")
        vectors = []
        for _ in range(augmentations):
            aug_img = augmentation_pipeline(img).unsqueeze(0).to(device)
            with torch.no_grad():
                vec = model(aug_img)
            vec = vec.squeeze().cpu().numpy()
            vec /= np.linalg.norm(vec)
            vectors.append(vec)
        return np.mean(vectors, axis=0)
    except Exception as e:
        print(f"❌ Failed to embed {img_path}: {e}")
        return None

# === Embedding generation loop ===
print("🔍 Generating embeddings...")
image_paths = []
embeddings = []

for category in tqdm(os.listdir(BASE_DIR), desc="📁 Categories"):
    good_dir = os.path.join(BASE_DIR, category, "train", "good")
    if not os.path.isdir(good_dir):
        continue

    for img_name in os.listdir(good_dir):
        img_path = os.path.join(good_dir, img_name)
        vec = extract_features(img_path)
        if vec is not None:
            image_paths.append(img_path)
            embeddings.append(vec)

# === Save outputs ===
if len(embeddings) > 0:
    print("💾 Saving outputs...")
    np.save(SAVE_EMBEDDINGS_PATH, np.stack(embeddings).astype("float32"))
    with open(SAVE_IMAGE_PATHS_JSON, "w") as f:
        json.dump(image_paths, f)
    print(f"✅ Saved {len(embeddings)} embeddings to {SAVE_EMBEDDINGS_PATH}")
    print(f"✅ Saved image paths to {SAVE_IMAGE_PATHS_JSON}")
else:
    print("⚠️ No embeddings generated. Check your dataset paths and augmentation pipeline.")
