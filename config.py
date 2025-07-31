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
    
    # Default parameters
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 100
    DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
    DEFAULT_LLM_MODEL = "gpt-4o"
    DEFAULT_TEMPERATURE = 0.0
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
        
        logger.info("All required environment variables are set")
        return True
    
    @classmethod
    def load_parameters_config(cls, config_path: str = None) -> Dict[str, Any]:
        """Load parameters configuration from JSON file"""
        if config_path is None:
            config_path = cls.PARAMETER_TUNING_DIR / "parameters_config.json"
        
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Parameters config file not found at {config_path}. Using defaults.")
            return cls.get_default_parameters_config()
    
    @classmethod
    def get_default_parameters_config(cls) -> Dict[str, Any]:
        """Get default parameters configuration"""
        return {
            "chunk_sizes": [500, 1000, 1500],
            "chunk_overlaps": [50, 100, 150],
            "embedding_models": ["text-embedding-3-large"],
            "llm_models": ["gpt-4o"],
            "temperatures": [0.0, 0.1],
            "top_k_retrieval": [3, 5, 10]
        }
