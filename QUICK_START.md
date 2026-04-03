# Quick Start

## Setup

```bash
pip install -r requirements.txt
export GROQ_API_KEY=gsk_...  # Get from console.groq.com
```

## Run

```bash
python search.py
```

## What It Does

Choose:
1. **Search Candidates** - Find candidates matching a job query
2. **Search Jobs** - Find jobs matching a candidate profile

### Example Queries

**Candidates:**
- "senior backend engineer, 4+ years, Python and Go"
- "ML engineer who's deployed production models"
- "React Native developer from consumer tech"

**Jobs:**
- "I'm a senior backend engineer with Go and Python"
- "Full-stack generalist, 2-5 years startup experience"

## Code Structure

- `search.py` - Main search logic (simple functions, no classes)
- `ingest.py` - Load CSV → clean → embed → save
- `main.py` - Entry point
- `ANALYSIS.md` - Technical deep dive
- `test_search.py` - Quick test without interactive mode
