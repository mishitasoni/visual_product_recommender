import os
import csv
import random
from tqdm import tqdm

# Config
BASE_DIR = "data/mvtec_ad"
OUTPUT_CSV = "data/metadata.csv"

# Categories you mentioned
CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "metalnut", "screw",
    "spanner", "toothbrush", "transistor", "washer", "wood", "zipper"
]

# Metadata Pools
materials_by_category = {
    "bottle": "plastic",
    "cable": "copper",
    "capsule": "gelatin",
    "carpet": "fabric",
    "metalnut": "steel",
    "screw": "steel",
    "spanner": "alloy steel",
    "toothbrush": "plastic",
    "transistor": "silicon",
    "washer": "stainless steel",
    "wood": "wood",
    "zipper": "metal"
}

types_by_category = {
    "bottle": "container",
    "cable": "wiring",
    "capsule": "pill",
    "carpet": "floor covering",
    "metalnut": "fastener",
    "screw": "fastener",
    "spanner": "tool",
    "toothbrush": "hygiene tool",
    "transistor": "electronic",
    "washer": "spacer",
    "wood": "material",
    "zipper": "fastener"
}

vendors = ["TechCo", "InduGear", "PartsPro", "MechaMart", "SupplyChainX"]

def generate_size(category):
    # You can make this more intelligent if needed
    return random.choice(["small", "medium", "large"])

def generate_description(cat):
    return f"High-quality {cat} used in industrial applications."

# Create metadata
metadata = []

print("🔍 Scanning dataset and generating metadata...")
for category in tqdm(CATEGORIES):
    good_dir = os.path.join(BASE_DIR, category, "train", "good")
    if not os.path.exists(good_dir):
        continue

    for file in os.listdir(good_dir):
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            rel_path = os.path.normpath(os.path.join(good_dir, file))
            metadata.append({
                "image_name": file,
                "path": rel_path,
                "category": category,
                "material": materials_by_category.get(category, "unknown"),
                "vendor": random.choice(vendors),
                "type": types_by_category.get(category, "component"),
                "size": generate_size(category),
                "description": generate_description(category)
            })

# Save to CSV
print(f"💾 Saving metadata to {OUTPUT_CSV} ...")
with open(OUTPUT_CSV, "w", newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=[
        "image_name", "path", "category", "material",
        "vendor", "type", "size", "description"
    ])
    writer.writeheader()
    writer.writerows(metadata)

print(f"✅ Metadata generated for {len(metadata)} images.")
