"""
Data ingestion pipeline for candidate embeddings.
Reads candidate data, generates embeddings, and saves for search/analysis.
"""

import pickle
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


def load_and_clean_candidates(csv_path: str) -> list[str]:
    """
    Load candidates from CSV and clean text.
    
    Args:
        csv_path: Path to candidates CSV file
        
    Returns:
        List of cleaned candidate text strings
    """
    df = pd.read_csv(csv_path)
    
    # Extract first column, replace NaN with empty strings
    candidates = df.iloc[:, 0].fillna("").astype(str).tolist()
    
    # Strip excessive newlines and normalize whitespace
    candidates = [
        re.sub(r'\n{2,}', '\n', text).strip()
        for text in candidates
    ]
    
    return candidates


def generate_embeddings(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Generate embeddings for text list using SentenceTransformer.
    
    Args:
        texts: List of text strings to embed
        model_name: SentenceTransformer model identifier
        
    Returns:
        NumPy array of embeddings (n_samples, embedding_dim)
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    return np.array(embeddings)


def save_artifacts(candidates: list[str], embeddings: np.ndarray) -> None:
    """
    Save candidates and embeddings to disk.
    
    Args:
        candidates: List of candidate text strings
        embeddings: NumPy array of embeddings
    """
    with open("candidates.pkl", "wb") as f:
        pickle.dump(candidates, f)
    
    np.save("embeddings.npy", embeddings)
    print(f"Saved {len(candidates)} candidates and embeddings")


def main() -> None:
    """Main ingestion pipeline."""
    print("Loading candidates...")
    candidates = load_and_clean_candidates("candidates.csv")
    print(f"Loaded {len(candidates)} candidates")
    
    print("Generating embeddings...")
    embeddings = generate_embeddings(candidates)
    print(f"Generated embeddings with shape {embeddings.shape}")
    
    print("Saving artifacts...")
    save_artifacts(candidates, embeddings)
    print("Done!")


if __name__ == "__main__":
    main()
