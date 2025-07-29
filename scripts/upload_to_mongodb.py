import numpy as np
import pandas as pd
from pymongo import MongoClient
import base64
import json

# === CONFIG ===
EMBEDDING_FILE = "embeddings/dino_aug_embeddings.npy"
METADATA_FILE = "data/metadata.csv"
MONGO_URI = "mongodb://localhost:27017" 
DB_NAME = "visual_recommender"
COLLECTION_NAME = "products3"

# === LOAD DATA ===
print("Loading embeddings...")
embeddings = np.load(EMBEDDING_FILE)
print(f"Embeddings loaded: {embeddings.shape}")

print("Loading metadata...")
metadata = pd.read_csv(METADATA_FILE)
print(f"Metadata rows: {metadata.shape[0]}")

# === CONNECT TO MONGODB ===
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# === UPLOAD RECORDS ===
print("Uploading to MongoDB...")

for i in range(len(metadata)):
    doc = metadata.iloc[i].to_dict()
    doc["embedding"] = embeddings[i].tolist()  # Convert np.array to list
    collection.insert_one(doc)

print(f"Uploaded {len(metadata)} documents to MongoDB collection: {COLLECTION_NAME}")
