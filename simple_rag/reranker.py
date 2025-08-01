"""
Reranking module for improving retrieval quality in RAG systems
"""

import logging
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

from sentence_transformers import CrossEncoder
from langchain_openai import OpenAIEmbeddings

from .qdrant_rag import RetrievalResult

logger = logging.getLogger(__name__)

class RerankingConfig:
    """Configuration for reranking loaded from parameters_config.json"""
    
    def __init__(self, 
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 top_k_before_rerank: int = 20,
                 top_k_after_rerank: int = 5,
                 enable_hybrid_scoring: bool = True,
                 cross_encoder_weight: float = 0.7,
                 embedding_weight: float = 0.3):
        """
        Initialize reranking configuration
        
        Args:
            model_name: Cross-encoder model name
            top_k_before_rerank: Number of docs to retrieve before reranking
            top_k_after_rerank: Final number of docs after reranking
            enable_hybrid_scoring: Whether to combine embedding + cross-encoder scores
            cross_encoder_weight: Weight for cross-encoder score
            embedding_weight: Weight for embedding similarity score
        """
        self.model_name = model_name
        self.top_k_before_rerank = top_k_before_rerank
        self.top_k_after_rerank = top_k_after_rerank
        self.enable_hybrid_scoring = enable_hybrid_scoring
        self.cross_encoder_weight = cross_encoder_weight
        self.embedding_weight = embedding_weight
    
    @classmethod
    def from_parameters_config(cls, config_path: str = None) -> 'RerankingConfig':
        """
        Load reranking configuration from parameters_config.json
        
        Args:
            config_path: Path to parameters_config.json file
            
        Returns:
            RerankingConfig instance with default values from config
        """
        if config_path is None:
            # Default path relative to this file
            config_path = Path(__file__).parent.parent / "parameters_config.json"
        
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            rerank_config = config_data.get("parameters", {}).get("rag_engine", {}).get("reranking", {})
            
            # Use first values from arrays as defaults
            return cls(
                model_name=rerank_config.get("model_names", ["cross-encoder/ms-marco-MiniLM-L-6-v2"])[0],
                top_k_before_rerank=rerank_config.get("top_k_before_rerank", [20])[0],
                enable_hybrid_scoring=rerank_config.get("enable_hybrid_scoring", [True])[0],
                cross_encoder_weight=rerank_config.get("cross_encoder_weights", [0.7])[0],
                embedding_weight=rerank_config.get("embedding_weights", [0.3])[0]
            )
            
        except (FileNotFoundError, KeyError, IndexError) as e:
            logger.warning(f"Could not load reranking config from {config_path}: {e}. Using defaults.")
            return cls()  # Return default configuration

class DocumentReranker:
    """
    Document reranker that improves retrieval quality by using cross-encoder models
    and hybrid scoring techniques
    """
    
    def __init__(self, config: RerankingConfig = None):
        """
        Initialize the reranker
        
        Args:
            config: Reranking configuration
        """
        self.config = config or RerankingConfig()
        
        # Initialize cross-encoder model
        try:
            self.cross_encoder = CrossEncoder(self.config.model_name)
            logger.info(f"Loaded cross-encoder model: {self.config.model_name}")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")
            self.cross_encoder = None
    
    def rerank_documents(self, 
                        query: str, 
                        documents: List[RetrievalResult],
                        top_k: int = None) -> List[RetrievalResult]:
        """
        Rerank documents using cross-encoder and hybrid scoring
        
        Args:
            query: User query
            documents: List of retrieved documents
            top_k: Number of documents to return (uses config default if None)
            
        Returns:
            Reranked list of documents
        """
        if not documents:
            return documents
            
        if self.cross_encoder is None:
            logger.warning("Cross-encoder not available, returning original order")
            return documents[:top_k or self.config.top_k_after_rerank]
        
        top_k = top_k or self.config.top_k_after_rerank
        
        try:
            # Prepare query-document pairs for cross-encoder
            query_doc_pairs = []
            for doc in documents:
                query_doc_pairs.append([query, doc.content])
            
            # Get cross-encoder scores
            cross_encoder_scores = self.cross_encoder.predict(query_doc_pairs)
            
            # Apply hybrid scoring if enabled
            if self.config.enable_hybrid_scoring:
                final_scores = self._compute_hybrid_scores(documents, cross_encoder_scores)
            else:
                final_scores = cross_encoder_scores
            
            # Create new RetrievalResult objects with updated scores
            reranked_documents = []
            for i, doc in enumerate(documents):
                new_doc = RetrievalResult(
                    content=doc.content,
                    score=float(final_scores[i]),
                    metadata={
                        **doc.metadata,
                        "original_score": doc.score,
                        "cross_encoder_score": float(cross_encoder_scores[i]),
                        "reranked": True
                    }
                )
                reranked_documents.append(new_doc)
            
            # Sort by final score (descending)
            reranked_documents.sort(key=lambda x: x.score, reverse=True)
            
            logger.info(f"Reranked {len(documents)} documents, returning top {top_k}")
            return reranked_documents[:top_k]
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            return documents[:top_k]
    
    def _compute_hybrid_scores(self, 
                              documents: List[RetrievalResult], 
                              cross_encoder_scores: np.ndarray) -> np.ndarray:
        """
        Compute hybrid scores combining embedding similarity and cross-encoder scores
        
        Args:
            documents: Original documents with embedding scores
            cross_encoder_scores: Cross-encoder relevance scores
            
        Returns:
            Combined scores
        """
        # Normalize embedding scores to [0, 1] range
        embedding_scores = np.array([doc.score for doc in documents])
        if embedding_scores.max() > 1.0:
            # If scores are similarity scores (0-1), keep as is
            # If scores are distance-based, may need different normalization
            pass
        
        # Normalize cross-encoder scores to [0, 1] range  
        ce_scores = np.array(cross_encoder_scores)
        ce_min, ce_max = ce_scores.min(), ce_scores.max()
        if ce_max > ce_min:
            ce_scores_norm = (ce_scores - ce_min) / (ce_max - ce_min)
        else:
            ce_scores_norm = np.ones_like(ce_scores)
        
        # Combine scores using weighted average
        hybrid_scores = (
            self.config.embedding_weight * embedding_scores + 
            self.config.cross_encoder_weight * ce_scores_norm
        )
        
        return hybrid_scores
    
    def get_reranking_stats(self, 
                           original_docs: List[RetrievalResult],
                           reranked_docs: List[RetrievalResult]) -> Dict[str, Any]:
        """
        Get statistics about the reranking process
        
        Args:
            original_docs: Documents before reranking
            reranked_docs: Documents after reranking
            
        Returns:
            Dictionary with reranking statistics
        """
        if not original_docs or not reranked_docs:
            return {}
        
        # Calculate rank changes
        original_ids = [id(doc) for doc in original_docs]
        reranked_ids = [id(doc) for doc in reranked_docs]
        
        rank_changes = []
        for new_rank, doc_id in enumerate(reranked_ids):
            if doc_id in original_ids:
                old_rank = original_ids.index(doc_id)
                rank_change = old_rank - new_rank  # Positive = moved up
                rank_changes.append(rank_change)
        
        return {
            "total_documents_reranked": len(original_docs),
            "final_documents_returned": len(reranked_docs),
            "average_rank_change": np.mean(rank_changes) if rank_changes else 0,
            "max_rank_improvement": max(rank_changes) if rank_changes else 0,
            "documents_reordered": len([x for x in rank_changes if x != 0]),
            "reranking_config": {
                "model": self.config.model_name,
                "hybrid_scoring": self.config.enable_hybrid_scoring,
                "cross_encoder_weight": self.config.cross_encoder_weight
            }
        }
