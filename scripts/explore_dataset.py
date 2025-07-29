# scripts/explore_dataset.py

import os
import matplotlib.pyplot as plt
from PIL import Image

BASE_PATH = "data/mvtec_ad"
CATEGORIES = os.listdir(BASE_PATH)

def show_sample_images(category, defect_type='good', split='train', num_images=5):
    path = os.path.join(BASE_PATH, category, split, defect_type)
    images = sorted(os.listdir(path))[:num_images]

    print(f"\nCategory: {category} | Defect Type: {defect_type}")
    for img_file in images:
        img_path = os.path.join(path, img_file)
        img = Image.open(img_path)
        plt.imshow(img)
        plt.title(f"{category} - {defect_type}")
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    for cat in CATEGORIES:
        cat_path = os.path.join(BASE_PATH, cat)
        if not os.path.isdir(cat_path): continue
        show_sample_images(cat)
