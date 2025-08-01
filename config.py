import os
import json
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    """Configuration class for the RAG system"""
    
    # Environment variables
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL")
    
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    PROCESSOR_DIR = BASE_DIR / "processor"
    SIMPLE_RAG_DIR = BASE_DIR / "simple_rag"
    PARAMETER_TUNING_DIR = BASE_DIR / "parameter_tuning"
    RESULTS_DIR = BASE_DIR / "results"
    PREPARATION_RESULTS_DIR = RESULTS_DIR / "preparation"
    SCORING_RESULTS_DIR = RESULTS_DIR / "scoring"
    PRODUCTION_CONFIG_DIR = RESULTS_DIR / "production_config"
    
    # Default parameters (not defined in parameters_config.json)
    DEFAULT_COLLECTION_NAME = "pdf_documents"

    # MLflow settings
    MLFLOW_TRACKING_URI = "file:./mlruns"
    MLFLOW_EXPERIMENT_NAME = "rag_parameter_tuning"
    
    @classmethod
    def validate_env_vars(cls):
        """Validate that all required environment variables are set"""
        required_vars = ["OPENAI_API_KEY", "QDRANT_API_KEY", "QDRANT_URL"]
        missing_vars = [var for var in required_vars if not getattr(cls, var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist"""
        directories = [
            cls.DATA_DIR,
            cls.PARAMETER_TUNING_DIR,
            cls.RESULTS_DIR,
            cls.PREPARATION_RESULTS_DIR,
            cls.SCORING_RESULTS_DIR,
            cls.PRODUCTION_CONFIG_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
    
    @classmethod
    def get_timestamped_filename(cls, prefix: str, suffix: str = ".json") -> str:
        """Generate a timestamped filename"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}{suffix}"
    
    @classmethod
    def load_parameters_config(cls, config_path: str = None) -> Dict[str, Any]:
        """Load parameters configuration from JSON file"""
        if config_path is None:
            config_path = cls.BASE_DIR / "parameters_config.json"
        
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Parameters config file not found at {config_path}. This file is required.")
            raise FileNotFoundError(f"Required parameters_config.json not found at {config_path}")
    
    @classmethod
    def get_pdf_file_path(cls, config: Dict[str, Any] = None) -> str:
        """Get PDF file path from config or use default"""
        if config and "input" in config and "pdf_file_name" in config["input"]:
            return str(cls.DATA_DIR / config["input"]["pdf_file_name"])
        return str(cls.DATA_DIR / "random_machine_learning_pdf.pdf")
    
    @classmethod
    def get_benchmark_file_path(cls, config: Dict[str, Any] = None) -> str:
        """Get benchmark file path from config or use default"""
        if config and "input" in config and "validate_file_name" in config["input"]:
            return str(cls.DATA_DIR / config["input"]["validate_file_name"])
        return str(cls.DATA_DIR / "Unique_RAG_QA_Benchmark.csv")
    
    @classmethod
    def get_collection_prefix(cls, config: Dict[str, Any] = None) -> str:
        """Get collection prefix from config or use default"""
        if config and "input" in config and "collection_prefix" in config["input"]:
            return config["input"]["collection_prefix"]
        return "rag_test"
    
    @classmethod
    def get_chunking_parameters(cls, config: Dict[str, Any] = None) -> Dict[str, List]:
        """Get chunking parameters from config"""
        if not config:
            config = cls.load_parameters_config()
        
        return {
            "chunk_sizes": config.get("parameters", {}).get("chunkings", {}).get("chunk_sizes", [1000]),
            "chunk_overlaps": config.get("parameters", {}).get("chunkings", {}).get("chunk_overlaps", [100])
        }
    
    @classmethod
    def get_model_parameters(cls, config: Dict[str, Any] = None) -> Dict[str, List]:
        """Get model parameters from config"""
        if not config:
            config = cls.load_parameters_config()
        
        return {
            "embedding_models": config.get("parameters", {}).get("models", {}).get("embedding_models", ["text-embedding-3-large"]),
            "llm_models": config.get("parameters", {}).get("models", {}).get("llm_models", ["gpt-4o"])
        }
    
    @classmethod
    def get_rag_engine_parameters(cls, config: Dict[str, Any] = None) -> Dict[str, List]:
        """Get RAG engine parameters from config including reranking parameters"""
        if not config:
            config = cls.load_parameters_config()
        
        rag_engine = config.get("parameters", {}).get("rag_engine", {})
        
        base_params = {
            "temperatures": rag_engine.get("temperatures", [0.0]),
            "top_k_retrieval": rag_engine.get("top_k_retrieval", [5]),
            "score_thresholds": rag_engine.get("score_thresholds", [0.3]),
            "prompt_templates": rag_engine.get("prompt_templates", [
                "You are a helpful assistant that answers questions based on the provided context from PDF documents.\n\nContext from documents:\n{context}\n\nQuestion: {question}\n\nInstructions:\n1. Answer the question based ONLY on the information provided in the context\n2. If the answer is not found in the context, clearly state \"I don't have enough information in the provided documents to answer this question\"\n3. Be concise and accurate\n4. If relevant, mention which part of the document the information comes from\n\nAnswer:"
            ])
        }
        
        # Add reranking parameters if they exist
        reranking_config = rag_engine.get("reranking", {})
        if reranking_config:
            base_params["reranking"] = reranking_config
        
        return base_params
    
    @classmethod
    def get_experiment_settings(cls, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get experiment settings from config"""
        if not config:
            config = cls.load_parameters_config()
        
        return config.get("experiment_settings", {
            "max_combinations": 50,
            "random_seed": 42,
            "parallel_runs": 4,
            "evaluation_timeout": 300
        })
    
    @classmethod
    def get_ragas_metrics(cls, config: Dict[str, Any] = None) -> List[str]:
        """Get RAGAs metrics from config"""
        if not config:
            config = cls.load_parameters_config()
        
        return config.get("ragas_metrics", [
            "faithfulness",
            "answer_relevancy", 
            "context_precision",
            "context_recall"
        ])
    
    @classmethod
    def get_file_description(cls, config: Dict[str, Any] = None) -> str:
        """Get file description from config"""
        if not config:
            config = cls.load_parameters_config()
        
        return config.get("input", {}).get("file_description", "")
    
    @classmethod
    def get_example_questions(cls, config: Dict[str, Any] = None) -> List[str]:
        """Get example questions from config"""
        if not config:
            config = cls.load_parameters_config()
        
        return config.get("input", {}).get("example_questions", [])
