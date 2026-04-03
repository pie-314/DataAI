"""
Quick test of search functionality - no interactive input.
"""

import os
os.environ["GROQ_API_KEY"] = "test_key"  # Dummy key for testing search without LLM

from search import search, load_data

print("\n=== Testing Candidate Search ===\n")
cands, cand_emb = load_data("candidates.pkl", "candidates_embeddings.npy")
query = "senior backend engineer, Python, 4+ years"
results = search(query, cands, cand_emb, top_k=3)

for i, r in enumerate(results, 1):
    print(f"{i}. Similarity: {r['score']:.3f}")
    print(f"   {r['text'][:200]}...\n")

print("=== Testing Job Search ===\n")
jobs, job_emb = load_data("jobs.pkl", "jobs_embeddings.npy")
query = "I'm a React developer looking for remote work"
results = search(query, jobs, job_emb, top_k=3)

for i, r in enumerate(results, 1):
    print(f"{i}. Similarity: {r['score']:.3f}")
    print(f"   {r['text'][:200]}...\n")

print("✓ Search functionality working!")
