"""
Simple RAG module for Qdrant-based question answering
"""

from .qdrant_rag import QdrantRAG, RetrievalResult

# Reranker components are imported conditionally
try:
    from .reranker import DocumentReranker, RerankingConfig
    __all__ = ["QdrantRAG", "RetrievalResult", "DocumentReranker", "RerankingConfig"]
except ImportError:
    # Reranker dependencies not available
    __all__ = ["QdrantRAG", "RetrievalResult"]
