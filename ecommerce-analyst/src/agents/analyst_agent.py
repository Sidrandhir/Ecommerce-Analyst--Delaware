"""
Analyst Agent
=============
Responsibility: Receive retrieved context from the Retrieval Agent,
reason over it, perform any calculations or comparisons, and return
a structured business insight with source citations and confidence.

This agent does NOT retrieve data — it only reasons and answers.
"""
from typing import Dict, List
from langchain.schema import Document

from src.utils.llm_client import call_llm
from src.utils.logger import get_logger

logger = get_logger(__name__)

ANALYST_SYSTEM_PROMPT = """You are a senior e-commerce business analyst AI.

You will receive:
1. A business question from the user
2. Relevant data context retrieved from an e-commerce database (Online Retail II, UK-based)

Your task:
- Answer the question precisely using ONLY the provided context
- Perform calculations if needed (e.g. percentages, comparisons, rankings)
- Structure your response as:

## Answer
<Direct, clear answer to the question>

## Key Insights
<2-4 bullet points of the most important findings>

## Data Evidence
<Reference specific numbers or facts from the context>

## Confidence
<HIGH / MEDIUM / LOW — and a one-line reason>

## Recommended Actions
<1-2 concrete business recommendations based on the findings>

Rules:
- Never hallucinate numbers not present in the context
- If context is insufficient, say so clearly and explain what data is missing
- Be concise but precise — this is a business report, not an essay
- Monetary values are in British Pounds (£)
"""


class AnalystAgent:
    """
    Agent 2: Business reasoning and insight generation.

    Workflow:
      1. Receive context + query from Retrieval Agent
      2. Synthesise insights using structured reasoning prompt
      3. Return formatted business analysis
    """

    def analyse(
        self,
        query: str,
        context: str,
        routing: Dict = None,
    ) -> str:
        """
        Generate a business analysis answer.

        Args:
            query:   Original user question
            context: Retrieved and formatted context string
            routing: Routing metadata (for logging/debugging)

        Returns:
            Formatted markdown analysis string
        """
        if routing:
            logger.info(f"Analyst reasoning over {routing.get('filter_type','mixed')}-type context")

        user_prompt = f"""Business Question: {query}

--- RETRIEVED DATA CONTEXT ---
{context}
--- END CONTEXT ---

Please provide your analysis."""

        response = call_llm(
            system_prompt=ANALYST_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            cache_prefix="analyst",
        )
        return response

    def analyse_with_comparison(
        self,
        query: str,
        contexts: List[str],
        labels: List[str],
    ) -> str:
        """
        Multi-context comparison (e.g. Q1 vs Q2, UK vs Germany).
        Feeds multiple labelled contexts for side-by-side analysis.
        """
        combined = "\n\n".join(
            f"--- CONTEXT: {label} ---\n{ctx}" for label, ctx in zip(labels, contexts)
        )

        user_prompt = f"""Business Question: {query}

{combined}

Please compare the above data sets and provide a comparative analysis."""

        return call_llm(
            system_prompt=ANALYST_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            cache_prefix="analyst_compare",
        )
