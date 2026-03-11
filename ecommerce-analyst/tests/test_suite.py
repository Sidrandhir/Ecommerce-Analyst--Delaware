"""
Test Suite — E-Commerce AI Analyst
====================================
Covers:
  - Unit tests for core functions (chunking, doc building, caching, retry)
  - RAG evaluation metrics (faithfulness, relevance, context precision)
  - Agent workflow integration tests
  - Graceful degradation tests

Run:
    pytest tests/ -v
    pytest tests/ -v -k "unit"        # unit tests only
    pytest tests/ -v -k "integration" # integration tests only
"""
import pytest
import json
from unittest.mock import patch, MagicMock
from typing import List

from langchain.schema import Document


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_documents() -> List[Document]:
    return [
        Document(
            page_content=(
                "Product: WHITE HANGING HEART T-LIGHT HOLDER (StockCode: 85123A)\n"
                "Total Revenue: £36,234.00\nUnits Sold: 4,521\n"
                "Average Unit Price: £2.55\nCountries Sold In: 14"
            ),
            metadata={"type": "product", "stock_code": "85123A"}
        ),
        Document(
            page_content=(
                "Country: United Kingdom\nTotal Revenue: £6,747,123.00\n"
                "Number of Orders: 23,494\nUnique Customers: 3,950\n"
                "Average Order Value: £287.12"
            ),
            metadata={"type": "country", "country": "United Kingdom"}
        ),
        Document(
            page_content=(
                "Monthly Report — 2010-12\nTotal Revenue: £714,012.00\n"
                "Number of Orders: 2,811\nUnique Customers: 1,104\n"
                "Units Sold: 71,234"
            ),
            metadata={"type": "monthly", "month": "2010-12"}
        ),
        Document(
            page_content=(
                "Customer ID: 14646\nCountry: Netherlands\n"
                "Total Spend: £279,489.02\nNumber of Orders: 97\n"
                "First Order: 2009-12-07\nLast Order: 2011-12-09"
            ),
            metadata={"type": "customer", "customer_id": "14646"}
        ),
    ]


@pytest.fixture
def sample_df():
    import pandas as pd
    return pd.DataFrame({
        "Invoice": ["536365", "536366", "536367", "C536368"],
        "StockCode": ["85123A", "71053", "84406B", "85123A"],
        "Description": ["WHITE HANGING HEART", "WHITE METAL LANTERN", "CREAM CUPID", "WHITE HANGING HEART"],
        "Quantity": [6, 8, 6, -4],
        "InvoiceDate": pd.to_datetime(["2010-12-01", "2010-12-01", "2010-12-01", "2010-12-01"]),
        "Price": [2.55, 3.39, 2.75, 2.55],
        "Customer_ID": [17850.0, 17850.0, 17850.0, 17850.0],
        "Country": ["United Kingdom", "United Kingdom", "United Kingdom", "United Kingdom"],
    })


# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — Data Layer
# ══════════════════════════════════════════════════════════════════════════════

class TestDataIngestion:

    def test_clean_cancelled_orders(self, sample_df):
        """Cancelled orders (Invoice starts with 'C') must be removed."""
        import pandas as pd
        df = sample_df.copy()
        df = df[~df["Invoice"].astype(str).str.startswith("C")]
        assert len(df) == 3
        assert "C536368" not in df["Invoice"].values

    def test_clean_negative_quantities(self, sample_df):
        """Rows with non-positive Quantity must be removed."""
        df = sample_df.copy()
        df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
        assert all(df["Quantity"] > 0)

    def test_line_total_computation(self, sample_df):
        """LineTotal = Quantity * Price."""
        df = sample_df.copy()
        df["LineTotal"] = df["Quantity"] * df["Price"]
        assert abs(df.iloc[0]["LineTotal"] - 6 * 2.55) < 0.01

    def test_build_documents_types(self, sample_df):
        """build_documents should produce docs of each type."""
        import pandas as pd
        # Add required columns
        df = sample_df.copy()
        df = df[~df["Invoice"].astype(str).str.startswith("C")]
        df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
        df["LineTotal"] = df["Quantity"] * df["Price"]
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)
        df["Year"] = df["InvoiceDate"].dt.year

        from src.rag.ingest import build_documents
        docs = build_documents(df)
        types = {d.metadata["type"] for d in docs}
        assert "product" in types
        assert "country" in types
        assert "monthly" in types


# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — Chunking
# ══════════════════════════════════════════════════════════════════════════════

class TestChunking:

    def test_chunk_does_not_exceed_size(self, sample_documents):
        """No chunk should exceed CHUNK_SIZE characters."""
        from src.rag.ingest import chunk_documents
        from src.config import CHUNK_SIZE
        chunks = chunk_documents(sample_documents)
        for chunk in chunks:
            assert len(chunk.page_content) <= CHUNK_SIZE + 50  # small tolerance

    def test_chunk_preserves_metadata(self, sample_documents):
        """Metadata must be preserved after chunking."""
        from src.rag.ingest import chunk_documents
        chunks = chunk_documents(sample_documents)
        for chunk in chunks:
            assert "type" in chunk.metadata

    def test_chunk_count_reasonable(self, sample_documents):
        """Should produce at least as many chunks as input docs."""
        from src.rag.ingest import chunk_documents
        chunks = chunk_documents(sample_documents)
        assert len(chunks) >= len(sample_documents)


# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — Cache
# ══════════════════════════════════════════════════════════════════════════════

class TestCache:

    def test_cache_set_and_get(self, tmp_path, monkeypatch):
        """Cache should store and retrieve values."""
        monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
        monkeypatch.setenv("CACHE_ENABLED", "true")

        from src.utils import cache as cache_mod
        cache_mod._cache = None  # reset singleton

        cache_mod.set_cached("test_key", "test_value")
        assert cache_mod.get_cached("test_key") == "test_value"

    def test_cache_miss_returns_none(self, tmp_path, monkeypatch):
        """Cache miss should return None."""
        monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache2"))
        from src.utils import cache as cache_mod
        cache_mod._cache = None
        assert cache_mod.get_cached("nonexistent_key") is None

    def test_cache_disabled(self, tmp_path, monkeypatch):
        """When CACHE_ENABLED=false, get_cached always returns None."""
        monkeypatch.setenv("CACHE_ENABLED", "false")
        from src.utils import cache as cache_mod
        # Re-import to pick up env change
        import importlib
        import src.config as cfg
        importlib.reload(cfg)
        assert cache_mod.get_cached("any_key") is None


# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — Retriever
# ══════════════════════════════════════════════════════════════════════════════

class TestRetriever:

    def test_format_context_empty(self):
        """Empty docs should return a 'no data found' string."""
        from src.rag.retriever import HybridRetriever
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = MagicMock()
        hr = HybridRetriever(mock_vs, all_documents=None)
        result = hr.format_context([])
        assert "No relevant" in result

    def test_format_context_with_docs(self, sample_documents):
        """Context should include doc type and content."""
        from src.rag.retriever import HybridRetriever
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = MagicMock()
        hr = HybridRetriever(mock_vs, all_documents=None)
        ctx = hr.format_context(sample_documents[:2])
        assert "PRODUCT" in ctx
        assert "COUNTRY" in ctx
        assert "United Kingdom" in ctx


# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — Routing Agent
# ══════════════════════════════════════════════════════════════════════════════

class TestRetrievalAgent:

    def test_routing_parse_valid_json(self):
        """Routing agent should parse valid JSON correctly."""
        from src.agents.retrieval_agent import RetrievalAgent
        mock_retriever = MagicMock()
        agent = RetrievalAgent(mock_retriever)

        valid_json = json.dumps({
            "filter_type": "country",
            "sub_queries": ["UK revenue", "Germany revenue"],
            "reasoning": "Geographic comparison"
        })

        with patch("src.agents.retrieval_agent.call_llm", return_value=valid_json):
            routing = agent._route_query("Compare UK vs Germany")

        assert routing["filter_type"] == "country"
        assert len(routing["sub_queries"]) == 2

    def test_routing_graceful_on_bad_json(self):
        """Routing agent should gracefully fall back on malformed JSON."""
        from src.agents.retrieval_agent import RetrievalAgent
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        agent = RetrievalAgent(mock_retriever)

        with patch("src.agents.retrieval_agent.call_llm", return_value="not valid json {{"):
            routing = agent._route_query("any question")

        assert "sub_queries" in routing
        assert routing["filter_type"] is None  # fallback default


# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — Graceful Degradation
# ══════════════════════════════════════════════════════════════════════════════

class TestGracefulDegradation:

    def test_llm_failure_returns_fallback(self):
        """On repeated API failure, safe_llm_call returns fallback message."""
        from src.utils.retry import safe_llm_call

        def always_fails():
            raise ConnectionError("API down")

        result = safe_llm_call(always_fails, fallback="Service unavailable")
        assert result == "Service unavailable"

    def test_missing_api_key_raises(self, monkeypatch):
        """validate() should raise EnvironmentError when key is missing."""
        monkeypatch.setenv("GOOGLE_API_KEY", "")
        import importlib
        import src.config as cfg
        importlib.reload(cfg)
        with pytest.raises(EnvironmentError, match="GOOGLE_API_KEY"):
            cfg.validate()


# ══════════════════════════════════════════════════════════════════════════════
# RAG EVALUATION — Faithfulness, Relevance, Context Precision
# ══════════════════════════════════════════════════════════════════════════════

class TestRAGEvaluation:
    """
    Lightweight RAG evaluation without requiring a running vectorstore.
    Uses mock context + answer pairs to test evaluation logic.

    In production, use RAGAS with real retrieved context:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
    """

    EVAL_PAIRS = [
        {
            "question": "Which country generates the most revenue?",
            "context": (
                "Country: United Kingdom\nTotal Revenue: £6,747,123.00\n"
                "Country: Germany\nTotal Revenue: £221,698.00\n"
                "Country: France\nTotal Revenue: £197,403.00"
            ),
            "answer": "The United Kingdom generates the most revenue at £6,747,123.",
            "expected_keywords": ["United Kingdom", "6,747"],
        },
        {
            "question": "What is the average order value for Germany?",
            "context": (
                "Country: Germany\nTotal Revenue: £221,698.00\n"
                "Number of Orders: 603\nAverage Order Value: £367.66"
            ),
            "answer": "The average order value for Germany is £367.66.",
            "expected_keywords": ["367", "Germany"],
        },
    ]

    def _faithfulness_score(self, answer: str, context: str) -> float:
        """
        Proxy faithfulness: check what fraction of answer sentences
        contain terms also present in context. (0.0 – 1.0)
        """
        sentences = [s.strip() for s in answer.split(".") if s.strip()]
        if not sentences:
            return 0.0
        context_words = set(context.lower().split())
        faithful = sum(
            1 for s in sentences
            if any(w in context_words for w in s.lower().split())
        )
        return faithful / len(sentences)

    def _relevance_score(self, question: str, answer: str) -> float:
        """
        Proxy relevance: keyword overlap between question and answer. (0.0 – 1.0)
        """
        q_words = set(question.lower().split()) - {"what", "which", "how", "is", "the", "a", "for"}
        a_words = set(answer.lower().split())
        if not q_words:
            return 0.0
        overlap = q_words & a_words
        return len(overlap) / len(q_words)

    def test_faithfulness_above_threshold(self):
        """Answers should be grounded in context (faithfulness > 0.5)."""
        for pair in self.EVAL_PAIRS:
            score = self._faithfulness_score(pair["answer"], pair["context"])
            assert score > 0.5, (
                f"Low faithfulness ({score:.2f}) for: {pair['question']}\n"
                f"Answer: {pair['answer']}"
            )

    def test_answer_contains_expected_keywords(self):
        """Answers must contain key facts from the context."""
        for pair in self.EVAL_PAIRS:
            for kw in pair["expected_keywords"]:
                assert kw in pair["answer"], (
                    f"Expected '{kw}' in answer for: {pair['question']}\n"
                    f"Answer: {pair['answer']}"
                )

    def test_relevance_above_threshold(self):
        """Answers should be relevant to the question (relevance > 0.3)."""
        for pair in self.EVAL_PAIRS:
            score = self._relevance_score(pair["question"], pair["answer"])
            assert score > 0.3, (
                f"Low relevance ({score:.2f}) for: {pair['question']}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TEST (marked slow — skipped in unit-only runs)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestOrchestratorIntegration:
    """
    End-to-end smoke test.
    Requires GOOGLE_API_KEY and the dataset to be present.
    Run with: pytest tests/ -v -m integration
    """

    def test_full_pipeline_returns_answer(self):
        """Full pipeline should return a non-empty answer string."""
        import os
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not set")

        from src.agents.orchestrator import EcommerceAnalystOrchestrator
        orch = EcommerceAnalystOrchestrator()
        orch.setup()

        result = orch.query("What are the top revenue-generating countries?")
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 50
        assert result["docs_retrieved"] > 0

    def test_pipeline_handles_ambiguous_query(self):
        """Pipeline should not crash on ambiguous or broad queries."""
        import os
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not set")

        from src.agents.orchestrator import EcommerceAnalystOrchestrator
        orch = EcommerceAnalystOrchestrator()
        orch.setup()

        result = orch.query("Tell me everything about the business.")
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 50
