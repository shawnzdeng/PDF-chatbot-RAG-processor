"""
Simple RAG module for Qdrant-based question answering
"""

from .qdrant_rag import QdrantRAG, RetrievalResult

__all__ = ["QdrantRAG", "RetrievalResult"]
