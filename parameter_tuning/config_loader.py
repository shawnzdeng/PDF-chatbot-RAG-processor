"""
Configuration utilities for parameter tuning with reranking support
"""

import json
import itertools
from typing import Dict, List, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ParameterConfigLoader:
    """Utility class to load and generate parameter combinations from config"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize parameter config loader
        
        Args:
            config_path: Path to parameters_config.json
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "parameters_config.json"
        
        self.config_path = config_path
        self.config_data = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            return {}
    
    def get_parameter_combinations(self, max_combinations: int = None) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations including reranking parameters
        
        Args:
            max_combinations: Maximum number of combinations to generate
            
        Returns:
            List of parameter dictionaries
        """
        params = self.config_data.get("parameters", {})
        
        # Extract parameter values
        chunking_params = params.get("chunkings", {})
        model_params = params.get("models", {})
        rag_params = params.get("rag_engine", {})
        rerank_params = rag_params.get("reranking", {})
        
        # Build parameter combinations
        param_combinations = []
        
        # Base parameters (existing)
        base_params = {
            "chunk_size": chunking_params.get("chunk_sizes", [1000]),
            "chunk_overlap": chunking_params.get("chunk_overlaps", [100]),
            "embedding_model": model_params.get("embedding_models", ["text-embedding-3-large"]),
            "llm_model": model_params.get("llm_models", ["gpt-4o"]),
            "temperature": rag_params.get("temperatures", [0.0]),
            "top_k": rag_params.get("top_k_retrieval", [5]),
            "score_threshold": rag_params.get("score_thresholds", [0.3]),
            "prompt_template": rag_params.get("prompt_templates", [self._get_default_prompt()])
        }
        
        # Reranking parameters
        rerank_param_dict = {
            "enable_reranking": rerank_params.get("enabled", [True, False]),
            "reranking_model": rerank_params.get("model_names", ["cross-encoder/ms-marco-MiniLM-L-6-v2"]),
            "rerank_top_k_before": rerank_params.get("top_k_before_rerank", [20]),
            "rerank_hybrid_scoring": rerank_params.get("enable_hybrid_scoring", [True]),
            "rerank_cross_encoder_weight": rerank_params.get("cross_encoder_weights", [0.7]),
            "rerank_embedding_weight": rerank_params.get("embedding_weights", [0.3])
        }
        
        # Combine all parameters
        all_params = {**base_params, **rerank_param_dict}
        
        # Generate combinations
        keys = list(all_params.keys())
        values = list(all_params.values())
        
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            
            # Add collection name
            param_dict["collection_name"] = self._generate_collection_name(param_dict)
            
            param_combinations.append(param_dict)
        
        # Limit combinations if specified
        if max_combinations and len(param_combinations) > max_combinations:
            # Use deterministic sampling to ensure reproducible results
            import random
            random.seed(42)  # Fixed seed for reproducibility
            param_combinations = random.sample(param_combinations, max_combinations)
        
        logger.info(f"Generated {len(param_combinations)} parameter combinations")
        return param_combinations
    
    def _generate_collection_name(self, params: Dict[str, Any]) -> str:
        """Generate unique collection name based on parameters"""
        prefix = self.config_data.get("input", {}).get("collection_prefix", "rag_")
        
        # Create a hash-like suffix based on key parameters
        key_params = [
            str(params.get("chunk_size", 1000)),
            str(params.get("chunk_overlap", 100)),
            params.get("embedding_model", "").replace("text-embedding-", "").replace("-", ""),
            str(params.get("enable_reranking", False))[:1].lower()  # t/f
        ]
        
        suffix = "_".join(key_params)
        return f"{prefix}{suffix}"
    
    def _get_default_prompt(self) -> str:
        """Get default prompt template"""
        return """You are a helpful assistant that answers questions based on the provided context from PDF documents.

Context from documents:
{context}

Question: {question}

Instructions:
1. Answer the question based ONLY on the information provided in the context
2. If the answer is not found in the context, clearly state "I don't have enough information in the provided documents to answer this question"
3. Be concise and accurate
4. If relevant, mention which part of the document the information comes from

Answer:"""
    
    def get_reranking_enabled_combinations(self) -> List[Dict[str, Any]]:
        """Get only parameter combinations with reranking enabled"""
        all_combinations = self.get_parameter_combinations()
        return [combo for combo in all_combinations if combo.get("enable_reranking", False)]
    
    def get_reranking_disabled_combinations(self) -> List[Dict[str, Any]]:
        """Get only parameter combinations with reranking disabled"""
        all_combinations = self.get_parameter_combinations()
        return [combo for combo in all_combinations if not combo.get("enable_reranking", False)]
    
    def export_sample_config(self, output_path: str = None):
        """Export a sample configuration showing all available parameters"""
        if output_path is None:
            output_path = Path(__file__).parent / "sample_parameters_with_reranking.json"
        
        sample_config = {
            "input": {
                "pdf_file_name": "your_document.pdf",
                "validate_file_name": "your_benchmark.csv",
                "collection_prefix": "rag_optimized_"
            },
            "parameters": {
                "chunkings": {
                    "chunk_sizes": [500, 1000, 1500],
                    "chunk_overlaps": [50, 100, 150]
                },
                "models": {
                    "embedding_models": ["text-embedding-3-large", "text-embedding-3-small"],
                    "llm_models": ["gpt-4o", "gpt-3.5-turbo"]
                },
                "rag_engine": {
                    "temperatures": [0.0, 0.1, 0.2],
                    "top_k_retrieval": [3, 5, 10],
                    "score_thresholds": [0.3, 0.5],
                    "prompt_templates": [self._get_default_prompt()],
                    "reranking": {
                        "enabled": [True, False],
                        "model_names": [
                            "cross-encoder/ms-marco-MiniLM-L-6-v2",
                            "cross-encoder/ms-marco-MiniLM-L-12-v2",
                            "cross-encoder/ms-marco-TinyBERT-L-2-v2"
                        ],
                        "top_k_before_rerank": [15, 20, 25],
                        "enable_hybrid_scoring": [True, False],
                        "cross_encoder_weights": [0.6, 0.7, 0.8],
                        "embedding_weights": [0.2, 0.3, 0.4]
                    }
                }
            },
            "experiment_settings": {
                "max_combinations": 50,
                "random_seed": 42,
                "parallel_runs": 4,
                "evaluation_timeout": 300
            },
            "ragas_metrics": [
                "faithfulness",
                "answer_relevancy", 
                "context_precision",
                "context_recall"
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(sample_config, f, indent=2)
        
        logger.info(f"Sample configuration exported to: {output_path}")
        return output_path
