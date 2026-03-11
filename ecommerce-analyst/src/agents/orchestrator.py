"""
Orchestrator — E-Commerce AI Analyst
"""
import os
import logging
from typing import Dict, List, Optional

from langchain.schema import Document
from langchain_chroma import Chroma

from src.agents.retrieval_agent import RetrievalAgent
from src.agents.analyst_agent import AnalystAgent
from src.rag.ingest import run_ingestion
from src.rag.retriever import build_retriever
from src.utils.logger import get_logger
from src.config import validate

# Kill ChromaDB telemetry noise entirely
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
os.environ["ANONYMIZED_TELEMETRY"] = "false"

logger = get_logger(__name__)


class EcommerceAnalystOrchestrator:

    def __init__(self):
        self.retrieval_agent: Optional[RetrievalAgent] = None
        self.analyst_agent: Optional[AnalystAgent] = None
        self._ready = False

    def _load_bm25_corpus(self, vectorstore: Chroma) -> Optional[List[Document]]:
        try:
            stored = vectorstore.get(include=["documents", "metadatas"])
            corpus = [
                Document(page_content=doc, metadata=meta or {})
                for doc, meta in zip(stored["documents"], stored["metadatas"])
                if doc and doc.strip()
            ]
            logger.info(f"BM25 corpus loaded: {len(corpus)} documents")
            return corpus if corpus else None
        except Exception as e:
            logger.warning(f"BM25 corpus load failed, using vector-only: {e}")
            return None

    def setup(self, force_ingest: bool = False) -> None:
        validate()
        logger.info("Setting up E-Commerce AI Analyst...")
        vectorstore: Chroma = run_ingestion(force=force_ingest)
        corpus = self._load_bm25_corpus(vectorstore)
        retriever = build_retriever(vectorstore, corpus)
        self.retrieval_agent = RetrievalAgent(retriever)
        self.analyst_agent = AnalystAgent()
        self._ready = True
        logger.info("System ready.")

    def _check_ready(self):
        if not self._ready:
            raise RuntimeError("Call orchestrator.setup() before querying.")

    def query(self, question: str) -> Dict:
        self._check_ready()
        logger.info(f"Query: {question}")
        retrieval_result = self.retrieval_agent.retrieve(question)
        answer = self.analyst_agent.analyse(
            query=question,
            context=retrieval_result["context"],
            routing=retrieval_result["routing"],
        )
        return {
            "question": question,
            "answer": answer,
            "docs_retrieved": len(retrieval_result["docs"]),
            "routing": retrieval_result["routing"],
            "context": retrieval_result["context"],
        }

    def compare(self, question: str, query_a: str, query_b: str,
                label_a: str = "A", label_b: str = "B") -> Dict:
        self._check_ready()
        result_a = self.retrieval_agent.retrieve(query_a)
        result_b = self.retrieval_agent.retrieve(query_b)
        answer = self.analyst_agent.analyse_with_comparison(
            query=question,
            contexts=[result_a["context"], result_b["context"]],
            labels=[label_a, label_b],
        )
        return {
            "question": question,
            "answer": answer,
            "docs_retrieved": len(result_a["docs"]) + len(result_b["docs"]),
        }