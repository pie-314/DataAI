"""
Vector search engine for candidate matching.
Uses sentence embeddings and cosine similarity for semantic search.
"""

import os
import time
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel


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
        
        # Initialize Groq client
        self.llm_client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )
        
        # Rich console for beautiful output
        self.console = Console()
        
        self.console.print(f"Loaded {len(self.candidates)} candidates")
        self.console.print(f"Embeddings shape: {self.embeddings.shape}")
    
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
    
    def generate_explanation(self, query: str, candidate_text: str) -> str:
        """
        Generate LLM explanation for why candidate matches query.
        
        Args:
            query: User's search query
            candidate_text: Candidate profile text
            
        Returns:
            LLM-generated explanation
        """
        response = self.llm_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert technical recruiter. Be concise."
                },
                {
                    "role": "user",
                    "content": (
                        f"Query: {query}. Candidate profile: {candidate_text}. "
                        "Provide a 1-sentence explanation of why they are a match, "
                        "and point out 1 thing they might be missing."
                    )
                }
            ]
        )
        return response.choices[0].message.content
    
    def run_interactive_search(self) -> None:
        """Run interactive CLI search loop."""
        self.console.print(
            Panel(
                "[bold cyan]Candidate Search Engine[/bold cyan]\n"
                "Type your search query (or 'exit' to quit)",
                style="bold blue"
            )
        )
        
        while True:
            query = self.console.input("[bold green]Query:[/bold green] ").strip()
            
            if query.lower() == "exit":
                self.console.print("[bold yellow]Goodbye![/bold yellow]")
                break
            
            if not query:
                continue
            
            self.console.print("\n[bold]Searching...[/bold]")
            results = self.vector_search(query, top_k=20)
            
            self.console.print(f"\n[bold cyan]Found {len(results)} candidates[/bold cyan]\n")
            
            for idx, result in enumerate(results, 1):
                candidate = result["candidate"]
                score = result["score"]
                
                # Truncate candidate text for display
                snippet = candidate[:300] + "..." if len(candidate) > 300 else candidate
                
                self.console.print(
                    Panel(
                        f"[bold yellow]Similarity: {score:.3f}[/bold yellow]\n\n"
                        f"{snippet}",
                        title=f"[bold]Candidate {idx}[/bold]",
                        style="blue"
                    )
                )
                
                # Generate and display explanation
                self.console.print("[dim]Generating explanation...[/dim]")
                explanation = self.generate_explanation(query, candidate)
                
                # Rate limit protection
                time.sleep(2)
                
                self.console.print(
                    Panel(
                        f"[bold green]{explanation}[/bold green]",
                        title="[bold]Match Analysis[/bold]",
                        style="green"
                    )
                )
                self.console.print()


def main() -> None:
    """Initialize search engine and run interactive CLI."""
    search = CandidateSearch()
    search.run_interactive_search()


if __name__ == "__main__":
    main()

