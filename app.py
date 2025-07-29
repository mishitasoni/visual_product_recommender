import streamlit as st
import os, json
import numpy as np
import pandas as pd
import torch, timm, faiss
from PIL import Image
from torchvision import transforms
from openai import OpenAI  # Compatible with Groq API

# --- API Config for Groq ---
client = OpenAI(
    api_key=st.secrets["GROQ_API_KEY"],  # Add this to your .streamlit/secrets.toml
    base_url="https://api.groq.com/openai/v1"
)

# --- Config ---
project_root = "."

IMAGE_PATHS = os.path.join(project_root, "embeddings/image_paths.json")
EMBEDDINGS_PATH = os.path.join(project_root, "embeddings/dino_aug_embeddings.npy")
INDEX_PATH = os.path.join(project_root, "embeddings/all_embeddings_index.faiss")
METADATA_CSV = os.path.join(project_root, "data/metadata.csv")
TOP_N_FINAL = 5

# --- Load Resources ---
@st.cache_resource
def load_resources():
    with open(IMAGE_PATHS, "r") as f:
        image_paths = json.load(f)

    metadata = pd.read_csv(METADATA_CSV)
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
    index = faiss.read_index(INDEX_PATH)
    return image_paths, metadata, embeddings, index

image_paths, metadata, embeddings, index = load_resources()

# --- Load DINOv2 Model ---
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True)
    model.head = torch.nn.Identity()
    model.eval().to(device)
    transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return model, transform, device

model, transform, device = load_model()

# --- Embed Uploaded Image ---
def embed_image(img):
    img = img.convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model(img_tensor)
    vec = vec.cpu().numpy()
    vec /= np.linalg.norm(vec)
    return vec.astype("float32")

# --- Rerank with Groq LLaMA3-70B ---
import re
import streamlit as st

def rerank_with_gpt(prompt, items_df, top_n=5):
    scored_items = []

    for idx, row in items_df.iterrows():
        description = row.get("description", "No description available")
        vendor = row.get("vendor", "")
        category = row.get("category", "")
        material = row.get("material", "")
        type_ = row.get("type", "")

        try:
            # LLM call
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for reranking product recommendations based on user needs. You must give a relevance score between 0 and 1, considering all provided metadata."},
                    {"role": "user", "content": f"User is looking for: {prompt}"},
                    {"role": "assistant", "content": f"""
Evaluate the relevance of the following product to the user query.

User Query: {prompt}

Product Metadata:
Category: {category}
Type: {type_}
Material: {material}
Vendor: {vendor}
Description: {description}

Return only a number from 0 (not relevant) to 1 (highly relevant). For example: 0.8
"""}
                ],
                temperature=0.3
            )

            score_str = response.choices[0].message.content.strip()

            # Debug: show LLM response
            print(f"Row {idx} LLM raw output: {score_str}")

            # Extract numeric score using regex
            match = re.search(r"\b(?:score\s*[:=]?\s*)?([01](?:\.\d+)?)\b", score_str.lower())
            if not match:
                st.warning(f"⚠️ Could not parse score from: `{score_str}`")
                score = 0.0
            else:
                score = float(match.group(1))

        except Exception as e:
            st.warning(f"❌ LLM scoring failed: {e}\n\nLLM raw output: {score_str}")
            score = 0.0

        scored_items.append(score)

    # Add LLM scores and final combined score
    items_df["llm_score"] = scored_items
    items_df["final_score"] = 0.6 * items_df["visual_score"] + 0.4 * items_df["llm_score"]

    # Sort and display
    reranked_df = items_df.sort_values("final_score", ascending=False).head(top_n)

    # ✅ Show full table in Streamlit
    st.subheader("🔍 LLM Reranked Products")
    st.dataframe(reranked_df.reset_index(drop=True), use_container_width=True)

    return reranked_df





# --- Streamlit UI ---
st.set_page_config(page_title="Visual Product Recommender", layout="wide")
st.title("🔍 Visual Product Recommender")
st.write("Upload a product image to find visually similar manufacturing items.")

# Sidebar Filters
st.sidebar.header("Filter Options")
materials = st.sidebar.multiselect("Select Material", metadata["material"].dropna().unique())
vendors = st.sidebar.multiselect("Select Vendor", metadata["vendor"].dropna().unique())
categories = st.sidebar.multiselect("Select Type", metadata["category"].dropna().unique())

# LLM Reranking
st.sidebar.markdown("---")
use_llm = st.sidebar.checkbox("Enable LLM-based Reranking")
llm_prompt = ""
if use_llm:
    llm_prompt = st.sidebar.text_area("Describe use-case (for reranking)", height=100)

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded:
    try:
        img = Image.open(uploaded)
        st.image(img, caption="Query Image", use_container_width=True)

        query_vec = embed_image(img)
        D, I = index.search(query_vec, 20)  # top-20 for reranking flexibility

        matched = []
        for dist, idx in zip(D[0], I[0]):
            img_path = os.path.normpath(os.path.join(project_root, image_paths[idx]))
            row = metadata[metadata["path"].apply(lambda x: os.path.normpath(os.path.join(project_root, x))) == img_path]
            if not row.empty:
                row = row.copy()
                row["visual_score"] = 1 - dist  # higher is better
                matched.append(row)

        if not matched:
            st.warning("⚠️ No similar items found.")
        else:
            result_df = pd.concat(matched, ignore_index=True)

            # Filters
            if materials:
                result_df = result_df[result_df["material"].isin(materials)]
            if vendors:
                result_df = result_df[result_df["vendor"].isin(vendors)]
            if categories:
                result_df = result_df[result_df["category"].isin(categories)]

            # Final ranking
            if result_df.empty:
                st.warning("⚠️ No matches after filtering.")
            else:
                if use_llm and llm_prompt.strip():
                    result_df = rerank_with_gpt(llm_prompt, result_df)
                else:
                    result_df = result_df.sort_values("visual_score", ascending=False).head(TOP_N_FINAL)

                st.subheader("🖼️ Top 5 Similar Matches")
                cols = st.columns(len(result_df))

                for i, (_, info) in enumerate(result_df.iterrows()):
                    with cols[i]:
                        img_path = os.path.normpath(os.path.join(project_root, info["path"]))
                        if os.path.exists(img_path):
                            st.image(img_path, caption=f"{info['category']} ({info['image_name']})", use_container_width=True)
                        else:
                            st.warning("Image not found.")
                        st.markdown(f"**Material:** {info['material']}")
                        st.markdown(f"**Vendor:** {info['vendor']}")
                        st.markdown(f"**Description:** {info['description']}")

    except Exception as e:
        st.error(f"Error processing image: {e}")
else:
    st.info("📎 Please upload an image to start.")
