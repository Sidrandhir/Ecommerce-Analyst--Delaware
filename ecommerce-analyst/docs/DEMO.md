# Demo Documentation
## E-Commerce AI Analyst — Delaware Take-Home Assignment

---

## System Overview

A two-agent Hybrid RAG system that answers natural language business questions over the
Online Retail II (UCI) transactional dataset (~1M orders, UK-based retailer, 2009–2011).

**Pipeline:** User Query → Retrieval Agent (routes + fetches) → Analyst Agent (reasons + reports)

---

## Example 1 — Geographic Revenue Analysis

**Query:**
```
Which country generates the most revenue, and what is their average order value?
```

**Routing Decision:** `filter_type: country`, `sub_queries: ["country revenue ranking", "average order value by country"]`

**Retrieved Context (excerpt):**
```
[1] (COUNTRY)
Country: United Kingdom
Total Revenue: £6,747,123.00
Number of Orders: 23,494
Unique Customers: 3,950
Average Order Value: £287.12

[2] (COUNTRY)
Country: Netherlands
Total Revenue: £284,661.00
Number of Orders: 683
Average Order Value: £416.78

[3] (COUNTRY)
Country: Germany
Total Revenue: £221,698.00
Average Order Value: £367.66
```

**Answer:**
```markdown
## Answer
The United Kingdom is by far the top revenue-generating country at £6,747,123 —
representing over 80% of total recorded sales. Their average order value is £287.12.

## Key Insights
- UK generates 30× more revenue than the next highest country (Netherlands at £284,661)
- Netherlands actually has a higher AOV (£416.78) than the UK — indicating a smaller
  but higher-value customer base
- Germany follows at £221,698 with an AOV of £367.66

## Confidence
HIGH — Dedicated country-level summary documents exist for all three countries.

## Recommended Actions
- Investigate Netherlands' premium purchasing behaviour for upsell opportunity
- UK retention is critical — losing this market would be existential for the business
```

---

## Example 2 — Monthly Trend Analysis

**Query:**
```
How did revenue trend across months in 2011? Were there any sharp drops?
```

**Routing Decision:** `filter_type: monthly`, `sub_queries: ["monthly revenue 2011", "revenue trend drop 2011"]`

**Answer:**
```markdown
## Answer
Revenue in 2011 followed a strong upward trend from January through November,
peaking at £714,012 in November. December showed a sharp decline to £433,668 —
a 39% drop from November's peak.

## Key Insights
- Q1 2011 averaged ~£487K/month — solid base
- Q3 2011 showed consistent growth: July (£533K), Aug (£573K), Sep (£609K)
- November 2011 was the single highest revenue month across both years
- December drop is likely seasonal (holiday cutoff) not a business risk signal

## Confidence
MEDIUM — Monthly summaries cover 2009-12 to 2011-12. Exact December figure
may reflect incomplete data capture rather than true revenue decline.

## Recommended Actions
- Plan inventory and marketing campaigns to capitalise on the Nov peak
- Verify December data completeness before drawing conclusions on year-end performance
```

---

## Example 3 — Top Customer Identification

**Query:**
```
Who are the highest-value customers and what country are they from?
```

**Answer:**
```markdown
## Answer
The highest-value customer is Customer 14646 from the Netherlands with £279,489 in
total spend across 97 orders. The top 3 customers are all international (non-UK).

## Key Insights
- Top customer (14646, Netherlands): £279,489 total spend — a wholesale buyer
- Second highest (18102, unknown): £259,657 across 89 orders
- International high-value customers show consistent long-term order patterns
  (3+ year tenure from first to last order)

## Confidence
MEDIUM-HIGH — Based on top-200 customer summaries. Full ranked list would require
a direct database query for completeness beyond the top tier.

## Recommended Actions
- Assign dedicated account management to customers spending £100K+/year
- Investigate if top Netherlands buyer represents a single company — churn risk is high
  if they are a reseller who switches supplier
```

---

## Running the Demo Yourself

```bash
# Setup
cp .env.example .env   # Add your GOOGLE_API_KEY
pip install -r requirements.txt

# Place dataset at data/online_retail_II.xlsx
# (Download: https://archive.ics.uci.edu/dataset/502/online+retail+ii)

# First run — ingests and builds vectorstore (~5 min)
python main.py

# Subsequent runs — loads existing vectorstore (<5 sec startup)
python main.py
```

**Expected startup output:**
```
11:42:03 | INFO     | src.rag.ingest | Loading dataset from data/online_retail_II.xlsx
11:42:08 | INFO     | src.rag.ingest | Dataset loaded: 824,364 rows, 5,878 customers, 38 countries
11:42:08 | INFO     | src.rag.ingest | Building product summaries...
11:42:12 | INFO     | src.rag.ingest | Total documents built: 762
11:42:30 | INFO     | src.rag.ingest | Vectorstore persisted successfully.
✅ System ready.
```

---

## Cost Estimate

| Operation | Tokens (est.) | Cost (Gemini 1.5 Flash) |
|---|---|---|
| Routing call | ~300 in, ~100 out | ~$0.0001 |
| Analysis call | ~1,500 in, ~600 out | ~$0.0005 |
| **Total per fresh query** | | **~$0.0006** |
| Cached query | 0 (disk cache hit) | $0.00 |

At 1,000 queries/day with 40% cache hit rate: ~$0.36/day.
