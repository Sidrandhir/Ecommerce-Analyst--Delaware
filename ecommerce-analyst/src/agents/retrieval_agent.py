"""
Retrieval Agent
===============
Responsibility: Analyse the user query, determine the BEST retrieval
strategy (what document types to fetch, any filters to apply), fetch
context from the vectorstore, and return structured context to the
Analyst Agent.

This agent does NOT answer the question — it only retrieves.
"""
from typing import Dict, List, Optional
from langchain.schema import Document

from src.rag.retriever import HybridRetriever
from src.utils.llm_client import call_llm
from src.utils.logger import get_logger

logger = get_logger(__name__)

ROUTING_SYSTEM_PROMPT = """You are a retrieval routing assistant for an e-commerce analytics platform.

Given a user question, output ONLY a JSON object (no markdown, no explanation) with:
{
  "filter_type": "<one of: product, country, monthly, customer, or null>",
  "sub_queries": ["<refined query 1>", "<refined query 2>"],
  "reasoning": "<one sentence>"
}

Rules:
- filter_type = "product" for questions about items, SKUs, descriptions, bestsellers
- filter_type = "country" for geographic or regional questions
- filter_type = "monthly" for time-based, trend, or period questions
- filter_type = "customer" for customer value, loyalty, or segmentation questions
- filter_type = null when the question spans multiple types
- sub_queries: decompose complex questions into 1-3 targeted retrieval queries
"""


class RetrievalAgent:
    """
    Agent 1: Smart retrieval with query routing.

    Workflow:
      1. Route the query → determine filter_type + sub_queries
      2. Execute hybrid retrieval for each sub_query
      3. Deduplicate and return top-K unique docs
    """

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def _route_query(self, query: str) -> Dict:
        """Ask the LLM how to best retrieve for this query."""
        import json

        raw = call_llm(
            system_prompt=ROUTING_SYSTEM_PROMPT,
            user_prompt=f"User question: {query}",
            cache_prefix="routing",
        )

        try:
            # Strip any accidental markdown fences
            cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            routing = json.loads(cleaned)
        except Exception as e:
            logger.warning(f"Routing parse failed ({e}), using defaults.")
            routing = {"filter_type": None, "sub_queries": [query], "reasoning": "fallback"}

        logger.info(f"Routing decision: {routing}")
        return routing

    def retrieve(self, query: str) -> Dict:
        """
        Main entry point.
        Returns:
          {
            "docs": List[Document],
            "context": str,          # formatted for LLM consumption
            "routing": dict,
          }
        """
        routing = self._route_query(query)
        filter_type: Optional[str] = routing.get("filter_type")
        sub_queries: List[str] = routing.get("sub_queries", [query])

        # Retrieve for each sub_query and deduplicate by page_content
        seen = set()
        all_docs: List[Document] = []

        for sq in sub_queries:
            docs = self.retriever.retrieve(sq, filter_type=filter_type)
            for doc in docs:
                key = doc.page_content[:100]
                if key not in seen:
                    seen.add(key)
                    all_docs.append(doc)

        context = self.retriever.format_context(all_docs)

        return {
            "docs": all_docs,
            "context": context,
            "routing": routing,
        }
