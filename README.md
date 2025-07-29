---
title: Visual Product Recommender
emoji: 🔍
colorFrom: gray
colorTo: gray
sdk: streamlit
sdk_version: "1.46.0"
app_file: app.py
pinned: false
license: mit
---


🔍 Visual Product Recommender
A Streamlit-based web application that allows users to upload an image of a manufacturing component (e.g., transistor) and get visually similar recommendations using deep visual embeddings. It also offers optional LLM-based reranking powered by Groq's LLaMA3 to tailor results based on specific use-cases.

🚀 Features

->🔎 Visual search using DINOv2 (ViT) embeddings.

->⚡ Fast nearest-neighbor retrieval with FAISS.

->🧠 Optional semantic reranking using LLM based on user intent.

->🧵 Filter results by material, vendor, and product category.

->💻 Interactive UI using Streamlit.

->📊 View detailed product metadata and similarity scores.

📁 Project Structure
bash
Copy
Edit
VisualProductRecommendor/
│
├── app.py                     # Main Streamlit app
├── data/
│   └── metadata.csv           # Product metadata
│
├── embeddings/
│   ├── all_embeddings_index.faiss
│   ├── dino_aug_embeddings.npy
│   └── image_paths.json
│
├── .streamlit/
│   └── secrets.toml           # Add your GROQ_API_KEY here
│
├── requirements.txt
└── README.md


🛠️ Tools & Technologies

| Component      | Description                                    |
| -------------- | ---------------------------------------------- |
| `Streamlit`    | UI framework for web app                       |
| `DINOv2 (ViT)` | Image embeddings via Vision Transformer        |
| `FAISS`        | Fast Approximate Nearest Neighbor Search       |
| `Groq`         | LLM inference (LLaMA3-8B) for reranking        |
| `OpenAI SDK`   | Used for interacting with Groq-compatible APIs |
| `Pandas`       | Data manipulation and filtering                |
| `Torchvision`  | Preprocessing for vision models                |

📦 Installation
🔧 1. Clone the repository

git clone https://github.com/your-username/VisualProductRecommendor.git
cd VisualProductRecommendor

📄 2. Create secrets.toml
Create a .streamlit/secrets.toml file and add your Groq API key:

GROQ_API_KEY = "your_groq_api_key_here"

📦 3. Install dependencies

pip install -r requirements.txt
🖥️ Usage

streamlit run app.py
Upload an image of a component (e.g., transistor).

Optionally filter by material, vendor, or category.

Enable LLM reranking and describe the use-case (e.g., "looking for high-quality transistors from MechaMart used in industrial applications").

View top-5 recommended results with metadata and similarity scores.

✨ Example Use-Case
User Prompt:

"Looking for high-quality transistors from vendor MechaMart used in industrial applications"

Output:
Top 5 visually and semantically matched transistors with metadata such as:

Material: Silicon

Vendor: MechaMart

Category: Transistor

Description: NPN high-gain industrial transistor




