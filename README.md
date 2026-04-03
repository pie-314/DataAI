# Candidate-Job Matching System

A semantic search engine that matches candidates to jobs and jobs to candidates using sentence embeddings and AI-powered explanations.

## Features

- **Bidirectional Search**: Search for candidates given job requirements OR search for jobs given candidate profiles
- **Semantic Matching**: Uses sentence transformers to understand intent beyond keywords
- **AI Explanations**: LLM-powered reasoning for why each match is relevant
- **Fast Retrieval**: In-memory vector search with O(n) cosine similarity
- **Beautiful CLI**: Rich terminal output with formatted panels and progress indicators

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Groq API Key

```bash
export GROQ_API_KEY=your_api_key_here
```

Get your free API key from [console.groq.com](https://console.groq.com)

### 3. Ingest Data (One-time)

This generates embeddings for candidates and jobs. Takes ~15 minutes.

```bash
python ingest.py
```

This creates:
- `candidates.pkl` - pickled candidate texts
- `candidates_embeddings.npy` - precomputed embeddings
- `jobs.pkl` - pickled job texts
- `jobs_embeddings.npy` - precomputed embeddings

### 4. Run the System

```bash
python main.py
```

You'll see an interactive menu:
```
What would you like to do?
[1] Search candidates for a job
[2] Search jobs for a candidate
[3] Exit
```

## Example Queries

### Candidate Search (Find candidates for a job)
- "senior backend engineer, 4+ years, Python and Go, Bangalore"
- "ML engineer who's done production deployment, not just notebooks"
- "React Native developer who has worked at consumer tech company"
- "devops person who actually understands networking, not just yaml pushers"

### Job Search (Find jobs for a candidate)
- "I'm a full-stack developer, React and Node.js, remote preferred"
- "Data engineer with 5 years experience, worked with Kafka at scale"
- "Founding engineer background, generalist, 2-5 years startup experience"
- "Engineering manager who still codes, prefer early-stage startups"

## How It Works

### Architecture

```
Raw Data (candidates.csv, jobs.csv)
    ↓
[Ingest Phase]
  - Clean & normalize text
  - Generate embeddings (all-MiniLM-L6-v2)
  - Save to disk (pkl + npy files)
    ↓
[Search Phase]
  - Encode query to embedding
  - Cosine similarity search
  - Return top 20 matches
    ↓
[Reasoning Phase]
  - For each result, call Groq LLM
  - Generate 1-sentence explanation
  - Display with similarity scores
```

### Why This Approach?

1. **Semantic Understanding**: Embeddings capture meaning beyond keywords (e.g., "software engineer" ≈ "developer")
2. **Efficiency**: ~3-5 MB of embeddings fit in memory, enabling fast <100ms searches
3. **Interpretability**: LLM explanations show why results match, not just scores
4. **Speed**: Groq provides fastest inference (280-560 tokens/sec)

## Files

- `ingest.py` - Data pipeline: load CSV → clean → embed → save
- `search.py` - Core search engine (SemanticSearch base class + CandidateSearch/JobSearch)
- `main.py` - Interactive CLI menu
- `ANALYSIS.md` - Detailed technical analysis and design decisions
- `candidates.csv` - Raw candidate profiles (~2000 rows)
- `jobs.csv` - Raw job listings (~70k rows)

## Performance

| Operation | Latency |
|-----------|---------|
| Single query embedding | <10ms |
| Vector search (2000 candidates) | <50ms |
| LLM explanation per result | ~2-3s |
| Full top-20 retrieval + explanations | ~40-50s |

## Limitations & Future Work

### Current Limitations
- Pure semantic matching doesn't enforce hard constraints (salary < 12L, location = Bangalore)
- No re-ranking by recruiter rules (experience gates, company filters)
- Single-threaded LLM calls (could batch for speed)

### Future Improvements
1. **Hybrid filtering**: Extract constraints via LLM, pre-filter before vector search
2. **Re-ranking**: Rule-based scoring combined with semantic similarity
3. **Batch LLM calls**: Submit multiple candidates per API request
4. **Caching**: Store explanations locally to avoid redundant API calls
5. **Fine-tuned embeddings**: Train domain-specific model on labeled pairs

## Troubleshooting

**"candidates.pkl not found"**
```bash
python ingest.py  # Generate embeddings first
```

**"GROQ_API_KEY not set"**
```bash
export GROQ_API_KEY=gsk_...  # Get from console.groq.com
```

**"Rate limit exceeded (429)"**
- System has built-in 2-second delays between API calls
- If still hitting limits, request higher tier on Groq console

**Slow embedding generation**
- First run loads the model (~400MB) → caches locally
- Subsequent runs are faster
- Generation is single-threaded; could parallelize with multiprocessing

## Tech Stack

- **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`)
- **Similarity**: scikit-learn (`cosine_similarity`)
- **LLM**: Groq API (`llama-3.1-8b-instant`)
- **CLI**: Rich (beautiful terminal output)
- **ML**: NumPy, pandas

## License

This project is built as part of an internship assignment.

---

Built with using semantic search and LLM reasoning.
