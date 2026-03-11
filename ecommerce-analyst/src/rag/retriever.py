"""
RAG Retriever — Hybrid Search (Dense + BM25 keyword fallback)
=============================================================
Combines semantic (vector) search with BM25 keyword retrieval,
then fuses results using Reciprocal Rank Fusion (RRF).
"""
from typing import List, Optional
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import (
    CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME,
    RETRIEVAL_TOP_K, GOOGLE_API_KEY
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    """
    Wraps ChromaDB vector search + BM25 keyword search.
    Falls back to vector-only if BM25 corpus is not loaded.
    """

    def __init__(self, vectorstore: Chroma, all_documents: Optional[List[Document]] = None):
        self.vectorstore = vectorstore
        self.vector_retriever = vectorstore.as_retriever(
            search_type="mmr",               # Max Marginal Relevance — reduces redundancy
            search_kwargs={"k": RETRIEVAL_TOP_K, "fetch_k": RETRIEVAL_TOP_K * 3}
        )

        if all_documents:
            bm25 = BM25Retriever.from_documents(all_documents)
            bm25.k = RETRIEVAL_TOP_K
            self.retriever = EnsembleRetriever(
                retrievers=[self.vector_retriever, bm25],
                weights=[0.6, 0.4],   # semantic weighted higher; BM25 boosts exact matches
            )
            logger.info("Hybrid retriever initialised (vector 60% + BM25 40%)")
        else:
            self.retriever = self.vector_retriever
            logger.info("Vector-only retriever initialised (no BM25 corpus)")

    def retrieve(self, query: str, filter_type: Optional[str] = None) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        Optionally filter by document type: 'product', 'country', 'monthly', 'customer'.
        """
        if filter_type:
            # ChromaDB metadata filter
            filtered = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": RETRIEVAL_TOP_K,
                    "fetch_k": RETRIEVAL_TOP_K * 3,
                    "filter": {"type": filter_type}
                }
            )
            docs = filtered.invoke(query)
        else:
            docs = self.retriever.invoke(query)

        logger.debug(f"Retrieved {len(docs)} docs for query: '{query[:60]}...'")
        return docs

    def format_context(self, docs: List[Document]) -> str:
        """Format retrieved docs into a single context string for the LLM."""
        if not docs:
            return "No relevant data found in the knowledge base."
        parts = []
        for i, doc in enumerate(docs, 1):
            doc_type = doc.metadata.get("type", "data")
            parts.append(f"[{i}] ({doc_type.upper()})\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)


def build_retriever(vectorstore: Chroma, corpus: Optional[List[Document]] = None) -> HybridRetriever:
    return HybridRetriever(vectorstore, corpus)