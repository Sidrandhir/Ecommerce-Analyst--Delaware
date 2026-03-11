# Architecture Decision Record (ADR)
## E-Commerce AI Analyst — Delaware Take-Home Assignment

---

## 1. Business Problem

Online retailers generate enormous transactional datasets, but extracting business intelligence from them typically requires skilled analysts or complex BI dashboards. The goal of this system is to allow business users to ask natural-language questions — "Which products are underperforming?", "How does our German market compare to France?" — and receive structured, data-grounded answers instantly.

I chose the **Online Retail II (UCI)** dataset because it reflects a real UK-based retailer with ~1M transactions across 38 countries and two fiscal years. It contains the right mix of richness (products, customers, geographies, time) to demonstrate both RAG retrieval depth and multi-hop agentic reasoning.

---

## 2. Approach: Hybrid RAG + Agentic (Option C)

I chose Option C for the following reasons:

**Why not RAG-only?** Pure RAG excels at document look-up but struggles with reasoning tasks that require comparison (UK vs Germany), trend analysis (month-over-month), or multi-step inference (which customers are at churn risk based on their purchasing patterns). A single retrieval + single generation step would miss these.

**Why not Agentic-only?** A pure agentic approach without RAG would require the LLM to hallucinate or guess numeric facts — unacceptable for a business analytics tool where precision matters.

**Hybrid wins** because the Retrieval Agent fetches grounded data, and the Analyst Agent applies structured reasoning over it. The split also makes each component independently testable.

---

## 3. Key Technical Decisions

### 3.1 LLM: Gemini 1.5 Flash

**Chosen over:** GPT-4o, Claude Sonnet, Gemini 1.5 Pro

**Reasoning:**
- **Cost efficiency**: Gemini 1.5 Flash is ~10× cheaper than GPT-4o at comparable quality for analytical tasks. For a production system with high query volume, this is critical.
- **128K context window**: Allows large context without chunking overhead in edge cases.
- **LangChain native support**: `langchain-google-genai` has first-class support.
- **Trade-off accepted**: Flash is slightly weaker on complex multi-step chain-of-thought vs Pro. I mitigate this with explicit structured prompting (section headers, rules) to constrain the output format.

### 3.2 Vector Database: ChromaDB (local)

**Chosen over:** Pinecone, Weaviate, Qdrant, pgvector

**Reasoning:**
- **Runs locally** — zero infrastructure, zero cost, satisfies the "standard laptop" constraint.
- **Persistent** — ChromaDB's SQLite backend persists across restarts without a running server.
- **LangChain integration** — `langchain-chroma` is the simplest integration path.
- **Trade-off accepted**: ChromaDB does not scale to billions of vectors. For production, I would migrate to Qdrant (self-hosted) or Pinecone (managed) at ~1M+ document scale.

### 3.3 Chunking Strategy: Mixed-Granularity Summary Documents

This was the most deliberate design choice in the system.

**Problem with naive chunking:** The dataset is tabular (transactions), not prose. Splitting raw rows into chunks produces fragments like "536365, 85123A, 6, 2.55" which carry no semantic meaning for embedding.

**My approach — pre-aggregated summaries:**
I converted the raw transactions into four document types before chunking:

| Type | Granularity | Purpose |
|---|---|---|
| `product` | Per SKU | Answers "what sells best", "which items are popular" |
| `country` | Per country | Answers geographic and regional questions |
| `monthly` | Per month | Answers trend and time-series questions |
| `customer` | Per customer (top 200) | Answers customer value / segmentation questions |

Each summary document is ~200–300 characters — already smaller than the 500-char chunk size — so most documents are NOT split. `RecursiveCharacterTextSplitter` is applied only as a safety net for any outlier documents.

**Why this works:** The LLM receives compact, semantically rich summaries ("Country: Germany, AOV: £367") rather than raw transaction rows. This dramatically improves retrieval precision and reduces context waste.

**Trade-off accepted:** Pre-aggregation loses row-level detail. I cannot answer "show me order #536365 specifically". For production, I would add a tool-calling layer where the agent can query a SQL/Pandas engine for row-level lookups.

### 3.4 Framework: LangChain

**Chosen over:** LlamaIndex, Haystack, custom

**Reasoning:**
- LangChain's `EnsembleRetriever` + `BM25Retriever` combination is the simplest way to implement hybrid search.
- Wide ecosystem: every component (Chroma, Gemini, BM25) has a maintained LangChain integration.
- The assignment mentions Azure Agent Framework as preferred but LangChain as acceptable. Given that the dataset is not Azure-hosted and no Azure services are needed, LangChain is the right pragmatic choice.
- **Trade-off accepted**: LangChain has a steeper abstraction overhead vs custom code. For a simple pipeline, this adds boilerplate. I offset this by keeping the agent logic in plain Python classes that wrap, not inherit, LangChain components.

### 3.5 Embeddings: Google `embedding-001`

**Chosen over:** OpenAI `text-embedding-3-small`, local sentence-transformers

**Reasoning:**
- Consistent API vendor (Gemini for LLM + embeddings = one API key, one billing account).
- Quality is comparable to OpenAI's small embedding model for English text.
- **Trade-off accepted**: Vendor lock-in to Google. In production I would evaluate switching to a local embedding model (e.g., `BAAI/bge-small-en`) to eliminate embedding costs entirely, as the vectorstore is built once and queried many times.

---

## 4. Agentic Design

### Agent 1: Retrieval Agent

**Role:** Decide *what* to fetch and fetch it.

**Workflow:**
1. Receives raw user query.
2. Calls LLM with a routing prompt → produces `{filter_type, sub_queries, reasoning}`.
3. For each sub-query, runs HybridRetriever (vector MMR + BM25).
4. Deduplicates results by content prefix.
5. Returns formatted context string + raw docs.

**Why an LLM for routing?** The filter type (product/country/monthly/customer) cannot be reliably determined by keyword matching alone. "Which items are losing popularity?" is clearly a product query, but "What happened in November?" is clearly monthly — a simple regex would struggle with "Compare our best customers in Germany".

### Agent 2: Analyst Agent

**Role:** Reason over retrieved context and produce structured answers.

**Workflow:**
1. Receives query + context from Retrieval Agent.
2. Calls LLM with a structured analysis prompt.
3. Forces output into: Answer / Key Insights / Data Evidence / Confidence / Recommended Actions.
4. Returns formatted markdown report.

**Why structured output?** Business users need scannable reports, not essay-style responses. Confidence scores allow downstream systems to flag low-confidence answers for human review.

---

## 5. Trade-offs Summary

| Decision | What I optimised for | What I gave up |
|---|---|---|
| Gemini Flash | Cost + speed | Max reasoning depth |
| ChromaDB local | Zero infra, easy setup | Scale beyond ~500K docs |
| Pre-aggregated summaries | Retrieval precision | Row-level query capability |
| Disk cache (24hr TTL) | Cost reduction, speed | Fresh data for rapidly changing facts |
| Two-agent split | Testability, separation of concerns | Latency (two LLM calls per query) |

---

## 6. What I Would Improve With More Time

1. **SQL/Pandas tool calling**: Add a third agent that can execute programmatic queries for precise calculations (e.g. exact percentile, cohort analysis). Currently all numerics come from pre-aggregated summaries.

2. **RAGAS evaluation pipeline**: Run full automated RAGAS evaluation (faithfulness, context precision, answer relevance) over a golden evaluation set of 50 hand-labelled Q&A pairs.

3. **Streaming responses**: Add streaming support for the Analyst Agent so users see tokens appear progressively, reducing perceived latency.

4. **Multi-turn conversation**: Add a conversation memory buffer so users can ask follow-up questions ("What about Germany specifically?") without repeating context.

5. **Local embeddings**: Replace `embedding-001` with a local `bge-small-en` model to eliminate embedding API cost entirely for the ingest step.

6. **Production deployment**: Package as a FastAPI service with a `/query` endpoint, enabling integration with BI tools or Slack bots.

---

## 7. Production Considerations

- **Monitoring**: Log query latency, cache hit rates, retrieval doc counts, and LLM token usage per query. Alert on P95 latency > 15s or error rate > 2%.
- **Cost control**: The disk cache reduces repeat query costs to ~$0. Estimated cost for a fresh query is ~$0.001–0.003 (2× Flash calls × ~1K tokens each).
- **Data freshness**: Re-ingest (`python main.py --ingest --force`) when new transaction data arrives. The vectorstore build takes ~5–10 minutes for 1M rows.
- **Security**: `GOOGLE_API_KEY` is never logged or included in cached responses. The `.env` file is `.gitignore`d.
- **Scalability**: Swap ChromaDB for Qdrant and add horizontal API replicas when query volume exceeds ~100 concurrent users.
