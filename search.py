"""
Vector search engine for candidate matching.
Uses sentence embeddings and cosine similarity for semantic search.
"""

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class CandidateSearch:
    """Semantic search engine for candidates using embeddings."""
    
    def __init__(self, candidates_pkl: str = "candidates.pkl", 
                 embeddings_npy: str = "embeddings.npy",
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize search engine with loaded data and model.
        
        Args:
            candidates_pkl: Path to pickled candidates list
            embeddings_npy: Path to embeddings NumPy array
            model_name: SentenceTransformer model identifier
        """
        # Load candidate texts
        with open(candidates_pkl, "rb") as f:
            self.candidates = pickle.load(f)
        
        # Load embeddings
        self.embeddings = np.load(embeddings_npy)
        
        # Load model
        self.model = SentenceTransformer(model_name)
        
        print(f"Loaded {len(self.candidates)} candidates")
        print(f"Embeddings shape: {self.embeddings.shape}")
    
    def vector_search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Search for top k candidates matching the query using cosine similarity.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of dicts with 'candidate' and 'score' keys, sorted by score
        """
        # Encode query to embedding
        query_embedding = self.model.encode(query)
        query_embedding = np.array(query_embedding).reshape(1, -1)
        
        # Compute cosine similarity against all candidates
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build results with candidate text and similarity score
        results = [
            {
                "candidate": self.candidates[idx],
                "score": float(similarities[idx])
            }
            for idx in top_indices
        ]
        
        return results
