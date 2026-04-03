"""
Ingest candidates and jobs data, clean it, generate embeddings.
"""

import pickle
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


def clean_text(text):
    """Remove excessive newlines and normalize whitespace."""
    text = str(text) if text else ""
    text = re.sub(r'\n{2,}', '\n', text).strip()
    return text


def load_and_clean(csv_file):
    """Load CSV and clean first column."""
    df = pd.read_csv(csv_file)
    texts = df.iloc[:, 0].fillna("").astype(str).tolist()
    texts = [clean_text(text) for text in texts]
    return texts


def generate_embeddings(texts):
    """Convert texts to embeddings using all-MiniLM-L6-v2."""
    print(f"Generating embeddings for {len(texts)} items...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True)
    return np.array(embeddings)


def save_data(texts, embeddings, prefix):
    """Save texts and embeddings to disk."""
    pkl_file = f"{prefix}.pkl"
    npy_file = f"{prefix}_embeddings.npy"
    
    with open(pkl_file, "wb") as f:
        pickle.dump(texts, f)
    np.save(npy_file, embeddings)
    
    print(f"✓ Saved {len(texts)} items to {pkl_file} and {npy_file}")


def ingest_candidates():
    """Load, clean, embed candidates."""
    print("\n=== Candidates ===")
    print("Loading candidates.csv...")
    texts = load_and_clean("candidates.csv")
    print(f"Loaded {len(texts)} candidates")
    
    embeddings = generate_embeddings(texts)
    save_data(texts, embeddings, "candidates")


def ingest_jobs():
    """Load, clean, embed jobs."""
    print("\n=== Jobs ===")
    print("Loading jobs.csv...")
    texts = load_and_clean("jobs.csv")
    print(f"Loaded {len(texts)} jobs")
    
    embeddings = generate_embeddings(texts)
    save_data(texts, embeddings, "jobs")


if __name__ == "__main__":
    ingest_candidates()
    ingest_jobs()
    print("\n✓ All data ingested and embedded!")
