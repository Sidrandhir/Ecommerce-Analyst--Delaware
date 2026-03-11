# 🛒 E-Commerce AI Analyst

A **Hybrid RAG + Agentic** system that answers complex business questions over the
[Online Retail II (UCI)](https://archive.ics.uci.edu/dataset/502/online+retail+ii) dataset
using **Gemini 1.5 Flash**, **LangChain**, and **ChromaDB**.

---

## What It Does

Ask natural language business questions and get structured analyst-grade answers:

```
❯ Which country generates the most revenue?

## Answer
The United Kingdom dominates revenue at £6,747,123 — over 80% of total sales.

## Key Insights
- UK accounts for 23,494 orders from 3,950 unique customers
- Average order value of £287.12 is highest among all regions
- Netherlands and Germany are the next largest markets, at ~3% each

## Data Evidence
UK Total Revenue: £6,747,123 | Orders: 23,494 | AOV: £287.12
Germany Total Revenue: £221,698 | Netherlands: £284,661

## Confidence
HIGH — multiple country-level summary documents confirm this directly.

## Recommended Actions
- Prioritise UK retention campaigns given its outsized revenue share
- Investigate Netherlands' high AOV (£416) for premium product expansion
```

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────┐
│     Retrieval Agent          │  ← Routes query, selects filter_type
│  (Query Router + Fetcher)    │    Runs hybrid search (Vector + BM25)
└──────────────┬──────────────┘
               │ Retrieved Context (top-K docs)
               ▼
┌─────────────────────────────┐
│      Analyst Agent           │  ← Synthesises context, calculates,
│  (Reasoning + Formatting)    │    generates structured business insight
└──────────────┬──────────────┘
               │
               ▼
        Formatted Answer
        (Markdown Report)
```

**Two-agent design:**

| Agent | Role | Model |
|---|---|---|
| Retrieval Agent | Query routing, hybrid search, deduplication | Gemini 1.5 Flash (routing) |
| Analyst Agent | Business reasoning, calculations, recommendations | Gemini 1.5 Flash (analysis) |

**Hybrid search combines:**
- **Dense (semantic)** — ChromaDB with Google `embedding-001`, MMR diversity
- **BM25 (keyword)** — exact term matching for product codes, country names
- **Weighted fusion** — 60% semantic, 40% BM25

---

## Dataset

**Online Retail II** (UCI Machine Learning Repository)  
- ~1M transactions, 2009–2011, UK-based online retailer
- Fields: Invoice, StockCode, Description, Quantity, InvoiceDate, Price, CustomerID, Country

Download: https://archive.ics.uci.edu/dataset/502/online+retail+ii  
Place the file at `data/online_retail_II.xlsx`

---

## Quick Start

### 1. Clone & configure

```bash
git clone <repo-url>
cd ecommerce-analyst
cp .env.example .env
# Edit .env — add your GOOGLE_API_KEY
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the dataset

```bash
# Download from UCI and place here:
data/online_retail_II.xlsx
```

### 4. Run

```bash
# Interactive REPL
python main.py

# Single query
python main.py --query "What are the top 5 products by revenue?"

# Force re-ingestion (rebuilds vectorstore)
python main.py --ingest
```

---

## Docker

```bash
# Build
docker build -t ecommerce-analyst .

# Run (interactive)
docker run -it \
  -e GOOGLE_API_KEY=your_key_here \
  -v $(pwd)/data:/app/data:ro \
  ecommerce-analyst

# Or with docker-compose
GOOGLE_API_KEY=your_key docker-compose up
```

---

## Running Tests

```bash
# All unit tests (no API key needed)
pytest tests/ -v -k "not integration"

# Full suite including integration (needs API key + dataset)
pytest tests/ -v

# Integration tests only
pytest tests/ -v -m integration
```

---

## Example Queries

| Query Type | Example |
|---|---|
| **Top products** | "What are the 5 best-selling products by revenue?" |
| **Regional analysis** | "Which countries have the highest average order value?" |
| **Trends** | "How did monthly revenue trend in 2011?" |
| **Customer value** | "Who are the top 10 highest-value customers?" |
| **Risk flags** | "Which months show the sharpest revenue drops?" |
| **Comparison** | "Compare Q1 2011 vs Q4 2010 performance" |

---

## Project Structure

```
ecommerce-analyst/
├── main.py                     # CLI entry point
├── requirements.txt
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── docs/
│   └── ADR.md                  # Architecture Decision Record
├── src/
│   ├── config.py               # Centralised config from .env
│   ├── agents/
│   │   ├── orchestrator.py     # Coordinates both agents
│   │   ├── retrieval_agent.py  # Query routing + hybrid retrieval
│   │   └── analyst_agent.py    # Business reasoning + answer generation
│   ├── rag/
│   │   ├── ingest.py           # Data loading, cleaning, embedding, persist
│   │   └── retriever.py        # HybridRetriever (vector + BM25)
│   └── utils/
│       ├── cache.py            # Disk-based LLM response cache
│       ├── retry.py            # Retry + graceful degradation
│       ├── llm_client.py       # Gemini LLM wrapper
│       └── logger.py           # Logging setup
└── tests/
    ├── conftest.py
    └── test_suite.py           # Unit + RAG eval + integration tests
```

---

## Measuring Success in Production

| Metric | Target | How to Measure |
|---|---|---|
| **Answer Faithfulness** | > 0.85 | RAGAS `faithfulness` — hallucination rate |
| **Context Relevance** | > 0.80 | RAGAS `context_relevancy` — retrieved docs relevance |
| **Answer Relevance** | > 0.80 | RAGAS `answer_relevancy` |
| **Latency (P95)** | < 8 sec | End-to-end query time logging |
| **Cache Hit Rate** | > 40% | Cache hit/miss counters |
| **API Error Rate** | < 2% | Retry failure tracking |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | **required** | Gemini API key |
| `GEMINI_MODEL` | `gemini-1.5-flash` | LLM model |
| `CACHE_ENABLED` | `true` | Enable disk cache |
| `CACHE_TTL_SECONDS` | `86400` | Cache TTL (24hr) |
| `RETRIEVAL_TOP_K` | `6` | Docs to retrieve per query |
| `CHUNK_SIZE` | `500` | Max chunk characters |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` |
