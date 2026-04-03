"""
Semantic search for candidates and jobs using embeddings and LLM explanations.
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


# Initialize once
console = Console()
model = None
llm_client = None


def initialize_llm():
    """Initialize Groq LLM client."""
    global llm_client
    llm_client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )


def initialize_model():
    """Initialize sentence transformer model."""
    global model
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")


def load_data(pkl_file, npy_file):
    """Load pickled data and embeddings."""
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
    embeddings = np.load(npy_file)
    return data, embeddings


def search(query, data, embeddings, top_k=20):
    """Find top k items matching query using cosine similarity."""
    initialize_model()
    
    # Encode query
    query_embedding = model.encode(query)
    query_embedding = np.array(query_embedding).reshape(1, -1)
    
    # Compute similarities
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Get top k
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            "text": data[idx],
            "score": float(similarities[idx])
        })
    
    return results


def explain_match(query, item_text, system_prompt):
    """Generate LLM explanation for why item matches query."""
    initialize_llm()
    
    try:
        response = llm_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Query: {query}. Item: {item_text}. "
                        "Provide a 1-sentence explanation of why they are a match, "
                        "and point out 1 thing they might be missing."
                    )
                }
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"High semantic match via vector space. [LLM generation failed: {str(e)}]"


def search_candidates(query):
    """Search for candidates matching a job query."""
    initialize_model()
    initialize_llm()
    
    console.print("\n[bold cyan]Loading candidates...[/bold cyan]")
    data, embeddings = load_data("candidates.pkl", "candidates_embeddings.npy")
    console.print(f"[dim]Loaded {len(data)} candidates[/dim]")
    
    console.print("[bold]Searching...[/bold]")
    results = search(query, data, embeddings, top_k=20)
    
    console.print(f"\n[bold cyan]Found {len(results)} matches[/bold cyan]\n")
    
    system_prompt = "You are an expert technical recruiter. Be concise."
    
    for idx, result in enumerate(results, 1):
        item = result["text"]
        score = result["score"]
        snippet = item[:300] + "..." if len(item) > 300 else item
        
        console.print(
            Panel(
                f"[bold yellow]Similarity: {score:.3f}[/bold yellow]\n\n{snippet}",
                title=f"[bold]Match {idx}[/bold]",
                style="blue"
            )
        )
        
        console.print("[dim]Generating explanation...[/dim]")
        explanation = explain_match(query, item, system_prompt)
        time.sleep(2)  # Rate limit
        
        console.print(
            Panel(
                f"[bold green]{explanation}[/bold green]",
                title="[bold]Match Analysis[/bold]",
                style="green"
            )
        )
        console.print()


def search_jobs(query):
    """Search for jobs matching a candidate profile."""
    initialize_model()
    initialize_llm()
    
    console.print("\n[bold cyan]Loading jobs...[/bold cyan]")
    data, embeddings = load_data("jobs.pkl", "jobs_embeddings.npy")
    console.print(f"[dim]Loaded {len(data)} jobs[/dim]")
    
    console.print("[bold]Searching...[/bold]")
    results = search(query, data, embeddings, top_k=20)
    
    console.print(f"\n[bold cyan]Found {len(results)} matches[/bold cyan]\n")
    
    system_prompt = "You are an expert job recruiter and career advisor. Be concise."
    
    for idx, result in enumerate(results, 1):
        item = result["text"]
        score = result["score"]
        snippet = item[:300] + "..." if len(item) > 300 else item
        
        console.print(
            Panel(
                f"[bold yellow]Similarity: {score:.3f}[/bold yellow]\n\n{snippet}",
                title=f"[bold]Job {idx}[/bold]",
                style="blue"
            )
        )
        
        console.print("[dim]Generating explanation...[/dim]")
        explanation = explain_match(query, item, system_prompt)
        time.sleep(2)  # Rate limit
        
        console.print(
            Panel(
                f"[bold green]{explanation}[/bold green]",
                title="[bold]Match Analysis[/bold]",
                style="green"
            )
        )
        console.print()


def main():
    """Main interactive search loop."""
    console.print(
        Panel(
            "[bold cyan]Candidate-Job Matching System[/bold cyan]\n"
            "Type 'exit' to quit",
            style="bold blue"
        )
    )
    
    while True:
        console.print("\n[bold]What would you like to search?[/bold]")
        console.print("[1] Candidates (for a job)")
        console.print("[2] Jobs (for a candidate)")
        console.print("[3] Exit")
        
        choice = console.input("\n[bold green]Pick (1-3):[/bold green] ").strip()
        
        if choice == "1":
            query = console.input("[bold green]Describe the job:[/bold green] ").strip()
            if query:
                try:
                    search_candidates(query)
                except FileNotFoundError as e:
                    console.print(f"[bold red]Error:[/bold red] {e}")
                    console.print("[dim]Run: python ingest.py[/dim]")
                except Exception as e:
                    console.print(f"[bold red]Error:[/bold red] {e}")
        
        elif choice == "2":
            query = console.input("[bold green]Describe the candidate:[/bold green] ").strip()
            if query:
                try:
                    search_jobs(query)
                except FileNotFoundError as e:
                    console.print(f"[bold red]Error:[/bold red] {e}")
                    console.print("[dim]Run: python ingest.py[/dim]")
                except Exception as e:
                    console.print(f"[bold red]Error:[/bold red] {e}")
        
        elif choice == "3":
            console.print("[bold yellow]Goodbye![/bold yellow]")
            break
        
        else:
            console.print("[bold red]Invalid choice.[/bold red]")


if __name__ == "__main__":
    main()
