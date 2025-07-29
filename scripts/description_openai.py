import os
import openai
import pandas as pd
from tqdm import tqdm

# ✅ Set your OpenAI API key (use environment variable or paste directly — NOT recommended for production)
openai.api_key = os.getenv("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")  # safer
# OR uncomment and paste manually (less safe)
# openai.api_key = "your-api-key-here"

# ✅ Load your metadata file (must exist in the same folder or provide correct path)
metadata_path = "data/metadata.csv"
if not os.path.exists(metadata_path):
    raise FileNotFoundError(f"📂 File not found: {metadata_path}")

df = pd.read_csv(metadata_path)

# ✅ Define a reusable function using the new SDK interface
client = openai.OpenAI(api_key="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")  # <- paste your key here


def generate_description(category, material, type_, size):
    prompt = (
        f"Write a short, clear, product-style description for a manufacturing item "
        f"with the following details:\n"
        f"- Category: {category}\n"
        f"- Material: {material}\n"
        f"- Type: {type_}\n"
        f"- Size: {size}\n\n"
        f"Use professional language, keep it under 30 words."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4",  # or "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "You are a helpful assistant for manufacturing companies."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Error generating description: {e}")
        return "N/A"

# ✅ Add 'description' column if not present
if "description" not in df.columns:
    df["description"] = ""

# ✅ Iterate and fill missing descriptions
print("🧠 Generating missing product descriptions using GPT...")
for i, row in tqdm(df.iterrows(), total=len(df)):
    if pd.isna(row["description"]) or row["description"].strip() == "":
        desc = generate_description(
            category=row["category"],
            material=row["material"],
            type_=row["type"],
            size=row["size"]
        )
        df.at[i, "description"] = desc

# ✅ Save back to CSV
df.to_csv("metadata_with_descriptions.csv", index=False)
print("✅ Descriptions saved to metadata_with_descriptions.csv")

