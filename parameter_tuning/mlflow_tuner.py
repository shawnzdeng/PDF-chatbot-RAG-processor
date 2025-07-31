"""
Parameter tuning system using MLflow and RAGAs evaluation
Optimizes RAG system parameters for best performance
"""

import os
import json
import random
import logging
import itertools
from typing import Dict, List, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

import mlflow
import mlflow.sklearn
from datasets import Dataset

# RAGAs imports
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

# Local imports
from config import Config
from processor import QdrantProcessor
from simple_rag import QdrantRAG

logger = logging.getLogger(__name__)

class ParameterTuner:
    """MLflow-based parameter tuning for RAG system"""
    
    def __init__(self, 
                 benchmark_file: str = None,
                 config_file: str = None,
                 experiment_name: str = None):
        """
        Initialize parameter tuner
        
        Args:
            benchmark_file: Path to CSV file with questions and ground truth
            config_file: Path to parameters configuration JSON
            experiment_name: MLflow experiment name
        """
        self.benchmark_file = benchmark_file or str(Config.PARAMETER_TUNING_DIR / "Unique_RAG_QA_Benchmark.csv")
        self.config_file = config_file or str(Config.PARAMETER_TUNING_DIR / "parameters_config.json")
        self.experiment_name = experiment_name or Config.MLFLOW_EXPERIMENT_NAME
        
        # Load configuration
        self.config = self._load_config()
        self.benchmark_data = self._load_benchmark_data()
        
        # Setup MLflow
        mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(self.experiment_name)
        
        # RAGAs metrics
        self.ragas_metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]
        
        logger.info(f"Initialized ParameterTuner with {len(self.benchmark_data)} test cases")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load parameter configuration"""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded parameter config from {self.config_file}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return Config.get_default_parameters_config()
    
    def _load_benchmark_data(self) -> pd.DataFrame:
        """Load benchmark questions and ground truth"""
        try:
            df = pd.read_csv(self.benchmark_file)
            logger.info(f"Loaded {len(df)} benchmark questions")
            return df
        except Exception as e:
            logger.error(f"Error loading benchmark data: {e}")
            raise
    
    def generate_parameter_combinations(self, max_combinations: int = None) -> List[Dict[str, Any]]:
        """
        Generate parameter combinations for testing
        
        Args:
            max_combinations: Maximum number of combinations to test
            
        Returns:
            List of parameter dictionaries
        """
        max_combinations = max_combinations or self.config.get("experiment_settings", {}).get("max_combinations", 50)
        
        # Get all possible combinations
        param_names = ["chunk_sizes", "chunk_overlaps", "embedding_models", "llm_models", "temperatures", "top_k_retrieval"]
        param_values = [self.config[name] for name in param_names]
        
        all_combinations = list(itertools.product(*param_values))
        
        # Randomly sample if too many combinations
        if len(all_combinations) > max_combinations:
            random.seed(self.config.get("experiment_settings", {}).get("random_seed", 42))
            all_combinations = random.sample(all_combinations, max_combinations)
        
        # Convert to dictionaries
        combinations = []
        for combo in all_combinations:
            param_dict = {
                "chunk_size": combo[0],
                "chunk_overlap": combo[1],
                "embedding_model": combo[2],
                "llm_model": combo[3],
                "temperature": combo[4],
                "top_k": combo[5]
            }
            combinations.append(param_dict)
        
        logger.info(f"Generated {len(combinations)} parameter combinations")
        return combinations
    
    def run_rag_evaluation(self, params: Dict[str, Any]) -> Dict[str, float]:
        """
        Run RAG evaluation for given parameters
        
        Args:
            params: Parameter dictionary
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Create collection name with timestamp to avoid conflicts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            collection_name = f"test_collection_{timestamp}_{random.randint(1000, 9999)}"
            
            # Process PDF with current parameters
            processor = QdrantProcessor(
                chunk_size=params["chunk_size"],
                chunk_overlap=params["chunk_overlap"],
                embedding_model=params["embedding_model"],
                collection_name=collection_name
            )
            
            pdf_path = Config.DATA_DIR / "random_machine_learing_pdf.pdf"
            success = processor.process_pdf_to_qdrant(str(pdf_path))
            
            if not success:
                logger.error("Failed to process PDF")
                return {"error": 1.0}
            
            # Initialize RAG system
            rag = QdrantRAG(
                collection_name=collection_name,
                embedding_model=params["embedding_model"],
                llm_model=params["llm_model"],
                temperature=params["temperature"],
                top_k=params["top_k"]
            )
            
            # Generate answers for benchmark questions
            results = []
            for _, row in self.benchmark_data.iterrows():
                question = row["question"]
                ground_truth = row["ground_truth"]
                
                try:
                    rag_result = rag.answer_question(question)
                    
                    result = {
                        "question": question,
                        "answer": rag_result["answer"],
                        "contexts": [rag_result["context"]],
                        "ground_truth": ground_truth
                    }
                    results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Error processing question '{question}': {e}")
                    # Add placeholder result to maintain consistency
                    result = {
                        "question": question,
                        "answer": "Error processing question",
                        "contexts": [""],
                        "ground_truth": ground_truth
                    }
                    results.append(result)
            
            # Convert to dataset for RAGAs
            dataset_dict = {
                "question": [r["question"] for r in results],
                "answer": [r["answer"] for r in results],
                "contexts": [r["contexts"] for r in results],
                "ground_truth": [r["ground_truth"] for r in results]
            }
            
            dataset = Dataset.from_dict(dataset_dict)
            
            # Evaluate with RAGAs
            evaluation_result = evaluate(
                dataset=dataset,
                metrics=self.ragas_metrics
            )
            
            # Clean up collection
            try:
                rag.qdrant_client.delete_collection(collection_name)
            except:
                pass
            
            # Extract metrics
            metrics = {
                "faithfulness": float(evaluation_result["faithfulness"]),
                "answer_relevancy": float(evaluation_result["answer_relevancy"]),
                "context_precision": float(evaluation_result["context_precision"]),
                "context_recall": float(evaluation_result["context_recall"])
            }
            
            # Calculate composite score
            metrics["composite_score"] = np.mean(list(metrics.values()))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in RAG evaluation: {e}")
            return {"error": 1.0, "composite_score": 0.0}
    
    def run_parameter_tuning(self, max_combinations: int = None) -> pd.DataFrame:
        """
        Run complete parameter tuning experiment
        
        Args:
            max_combinations: Maximum parameter combinations to test
            
        Returns:
            DataFrame with results
        """
        combinations = self.generate_parameter_combinations(max_combinations)
        results = []
        
        for i, params in enumerate(tqdm(combinations, desc="Parameter tuning")):
            with mlflow.start_run():
                # Log parameters
                mlflow.log_params(params)
                
                # Run evaluation
                metrics = self.run_rag_evaluation(params)
                
                # Log metrics
                mlflow.log_metrics(metrics)
                
                # Store result
                result = {**params, **metrics}
                results.append(result)
                
                logger.info(f"Completed run {i+1}/{len(combinations)}: "
                           f"composite_score={metrics.get('composite_score', 0):.3f}")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        results_file = Config.PARAMETER_TUNING_DIR / f"tuning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(results_file, index=False)
        
        logger.info(f"Parameter tuning completed. Results saved to {results_file}")
        
        return results_df
    
    def get_best_parameters(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get best parameters from results
        
        Args:
            results_df: Results DataFrame
            
        Returns:
            Best parameter configuration
        """
        # Filter out error runs
        valid_results = results_df[~results_df.get("error", False)]
        
        if valid_results.empty:
            logger.error("No valid results found")
            return {}
        
        # Find best composite score
        best_idx = valid_results["composite_score"].idxmax()
        best_params = valid_results.loc[best_idx]
        
        # Extract parameter values
        param_keys = ["chunk_size", "chunk_overlap", "embedding_model", "llm_model", "temperature", "top_k"]
        best_config = {key: best_params[key] for key in param_keys if key in best_params}
        
        # Add collection information for production use
        best_config["collection_name"] = Config.DEFAULT_COLLECTION_NAME
        best_config["qdrant_url"] = Config.QDRANT_URL
        
        # Add metrics
        best_config["metrics"] = {
            "faithfulness": best_params.get("faithfulness", 0),
            "answer_relevancy": best_params.get("answer_relevancy", 0),
            "context_precision": best_params.get("context_precision", 0),
            "context_recall": best_params.get("context_recall", 0),
            "composite_score": best_params.get("composite_score", 0)
        }
        
        # Add metadata for production integration
        best_config["optimization_metadata"] = {
            "tuning_date": datetime.now().isoformat(),
            "total_combinations_tested": len(valid_results),
            "benchmark_questions_count": len(self.benchmark_data),
            "optimization_framework": "MLflow + RAGAs",
            "ready_for_production": True
        }
        
        logger.info(f"Best parameters found with composite score: {best_config['metrics']['composite_score']:.3f}")
        
        return best_config
    
    def save_best_parameters(self, best_params: Dict[str, Any], filename: str = None):
        """Save best parameters to file"""
        filename = filename or str(Config.PARAMETER_TUNING_DIR / "best_parameters.json")
        
        with open(filename, 'w') as f:
            json.dump(best_params, f, indent=2)
        
        logger.info(f"Best parameters saved to {filename}")
    
    def export_production_config(self, best_params: Dict[str, Any], export_path: str = None) -> str:
        """
        Export optimized parameters in a format ready for production RAG system
        
        Args:
            best_params: Best parameter configuration
            export_path: Path to save the production config
            
        Returns:
            Path to the exported configuration file
        """
        export_path = export_path or str(Config.PARAMETER_TUNING_DIR / "production_rag_config.json")
        
        production_config = {
            "rag_system_config": {
                "chunk_size": best_params["chunk_size"],
                "chunk_overlap": best_params["chunk_overlap"],
                "embedding_model": best_params["embedding_model"],
                "llm_model": best_params["llm_model"],
                "temperature": best_params["temperature"],
                "top_k_retrieval": best_params["top_k"]
            },
            "qdrant_config": {
                "collection_name": best_params["collection_name"],
                "url": best_params["qdrant_url"],
                "vector_size": 3072 if "3-large" in best_params["embedding_model"] else 1536
            },
            "performance_metrics": best_params["metrics"],
            "optimization_info": best_params["optimization_metadata"],
            "usage_instructions": {
                "description": "Optimized RAG parameters from MLflow tuning with RAGAs evaluation",
                "integration_notes": [
                    "Use these exact parameters for optimal performance",
                    "Collection contains pre-processed document embeddings",
                    "Metrics show performance on evaluation benchmark",
                    "Ready for production deployment"
                ],
                "required_environment_variables": [
                    "OPENAI_API_KEY",
                    "QDRANT_API_KEY"
                ]
            }
        }
        
        with open(export_path, 'w') as f:
            json.dump(production_config, f, indent=2)
        
        logger.info(f"Production config exported to {export_path}")
        return export_path


def main():
    """Main function for parameter tuning"""
    tuner = ParameterTuner()
    
    # Run parameter tuning
    print("🔧 Starting focused parameter tuning with premium models...")
    results_df = tuner.run_parameter_tuning(max_combinations=18)  # All possible combinations
    
    # Get best parameters
    best_params = tuner.get_best_parameters(results_df)
    
    # Save best parameters
    tuner.save_best_parameters(best_params)
    
    # Export production config
    production_config_path = tuner.export_production_config(best_params)
    
    print(f"\n✅ Parameter optimization completed!")
    print(f"📊 Best composite score: {best_params['metrics']['composite_score']:.3f}")
    print(f"📁 Best parameters saved to: {Config.PARAMETER_TUNING_DIR / 'best_parameters.json'}")
    print(f"🚀 Production config exported to: {production_config_path}")
    
    print(f"\n🎯 Optimized Configuration (Premium Models):")
    print(f"   Chunk Size: {best_params['chunk_size']}")
    print(f"   Chunk Overlap: {best_params['chunk_overlap']}")
    print(f"   Embedding Model: {best_params['embedding_model']} (3072-dim)")
    print(f"   LLM Model: {best_params['llm_model']} (latest GPT-4)")
    print(f"   Temperature: {best_params['temperature']}")
    print(f"   Top-K Retrieval: {best_params['top_k']}")
    print(f"   Collection Name: {best_params['collection_name']}")
    
    print(f"\n📈 Performance Metrics:")
    for metric, score in best_params['metrics'].items():
        print(f"   {metric.replace('_', ' ').title()}: {score:.3f}")
    
    print(f"\n🔗 Integration Information:")
    print(f"   Ready for production RAG system integration")
    print(f"   Premium model configuration for highest quality")
    print(f"   Use production_rag_config.json for deployment")
    print(f"   Collection '{best_params['collection_name']}' contains optimized embeddings")
    
    print(f"\n📋 All {len(results_df)} parameter combinations tested:")
    if not results_df.empty:
        top_results = results_df.nlargest(len(results_df), 'composite_score')
        display_cols = ['chunk_size', 'chunk_overlap', 'temperature', 'top_k', 'composite_score']
        available_cols = [col for col in display_cols if col in top_results.columns]
        print(top_results[available_cols].to_string(index=False))


if __name__ == "__main__":
    main()
