"""
RAGAs evaluation system for RAG parameter tuning
Handles running RAGAs metrics on prepared collections with different scoring parameters
"""

import os
import json
import random
import logging
import time
from typing import Dict, List, Any
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
from simple_rag import QdrantRAG
from .collection_preparer import CollectionPreparer

logger = logging.getLogger(__name__)

class RAGAsScorer:
    """Handles RAGAs evaluation of prepared collections"""
    
    def __init__(self, 
                 benchmark_file: str = None,
                 config_file: str = None,
                 experiment_name: str = None):
        """
        Initialize RAGAs scorer
        
        Args:
            benchmark_file: Path to CSV file with questions and ground truth
            config_file: Path to parameters configuration JSON
            experiment_name: MLflow experiment name
        """
        self.config_file = config_file or str(Config.BASE_DIR / "parameters_config.json")
        self.experiment_name = experiment_name or Config.MLFLOW_EXPERIMENT_NAME
        
        # Load configuration and benchmark data
        self.config = self._load_config()
        self.benchmark_file = benchmark_file or Config.get_benchmark_file_path(self.config)
        self.benchmark_data = self._load_benchmark_data()
        
        # Initialize collection preparer for metadata loading
        self.collection_preparer = CollectionPreparer(self.config_file)
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
        
        try:
            # Set or create experiment
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"Using MLflow experiment: {self.experiment_name}")
        except Exception as e:
            logger.warning(f"Could not set MLflow experiment: {e}")
        
        # RAGAs metrics configuration
        self.ragas_metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]
        
        logger.info(f"Initialized RAGAsScorer with {len(self.benchmark_data)} benchmark questions")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load parameter configuration"""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded parameter config from {self.config_file}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def _load_benchmark_data(self) -> pd.DataFrame:
        """Load benchmark questions and ground truth"""
        try:
            df = pd.read_csv(self.benchmark_file)
            logger.info(f"Loaded {len(df)} benchmark questions from {self.benchmark_file}")
            return df
        except Exception as e:
            logger.error(f"Error loading benchmark data: {e}")
            raise
    
    def generate_scoring_combinations(self, max_combinations: int = None) -> List[Dict[str, Any]]:
        """
        Generate combinations for scoring parameters (excluding chunking parameters)
        Includes reranking configurations from parameters_config.json
        
        Args:
            max_combinations: Maximum combinations to generate
            
        Returns:
            List of scoring parameter dictionaries including reranking configs
        """
        rag_params = Config.get_rag_engine_parameters(self.config)
        model_params = Config.get_model_parameters(self.config)
        
        llm_models = model_params["llm_models"]
        temperatures = rag_params["temperatures"]
        top_k_retrieval = rag_params["top_k_retrieval"]
        score_thresholds = rag_params["score_thresholds"]
        prompt_templates = rag_params["prompt_templates"]
        
        # Extract reranking parameters
        reranking_params = rag_params.get("reranking", {})
        reranking_enabled = reranking_params.get("enabled", [False])
        reranking_models = reranking_params.get("model_names", ["cross-encoder/ms-marco-MiniLM-L-6-v2"])
        top_k_before_rerank = reranking_params.get("top_k_before_rerank", [20])
        hybrid_scoring = reranking_params.get("enable_hybrid_scoring", [True])
        cross_encoder_weights = reranking_params.get("cross_encoder_weights", [0.7])
        embedding_weights = reranking_params.get("embedding_weights", [0.3])
        
        # Create combinations of scoring parameters including reranking
        scoring_combinations = []
        for llm_model in llm_models:
            for temperature in temperatures:
                for top_k in top_k_retrieval:
                    for score_threshold in score_thresholds:
                        for prompt_template in prompt_templates:
                            for rerank_enabled in reranking_enabled:
                                if rerank_enabled:
                                    # If reranking is enabled, test different reranking configurations
                                    for rerank_model in reranking_models:
                                        for top_k_before in top_k_before_rerank:
                                            for hybrid_score in hybrid_scoring:
                                                for ce_weight in cross_encoder_weights:
                                                    for emb_weight in embedding_weights:
                                                        scoring_combinations.append({
                                                            "llm_model": llm_model,
                                                            "temperature": temperature,
                                                            "top_k": top_k,
                                                            "score_threshold": score_threshold,
                                                            "prompt_template": prompt_template,
                                                            "enable_reranking": rerank_enabled,
                                                            "reranking_model": rerank_model,
                                                            "rerank_top_k_before": top_k_before,
                                                            "rerank_hybrid_scoring": hybrid_score,
                                                            "rerank_cross_encoder_weight": ce_weight,
                                                            "rerank_embedding_weight": emb_weight
                                                        })
                                else:
                                    # If reranking is disabled, add configuration without reranking params
                                    scoring_combinations.append({
                                        "llm_model": llm_model,
                                        "temperature": temperature,
                                        "top_k": top_k,
                                        "score_threshold": score_threshold,
                                        "prompt_template": prompt_template,
                                        "enable_reranking": rerank_enabled,
                                        "reranking_model": None,
                                        "rerank_top_k_before": None,
                                        "rerank_hybrid_scoring": None,
                                        "rerank_cross_encoder_weight": None,
                                        "rerank_embedding_weight": None
                                    })
        
        # Limit combinations if specified
        if max_combinations and len(scoring_combinations) > max_combinations:
            experiment_settings = Config.get_experiment_settings(self.config)
            random.seed(experiment_settings.get("random_seed", 42))
            scoring_combinations = random.sample(scoring_combinations, max_combinations)
        
        reranking_enabled_count = sum(1 for combo in scoring_combinations if combo.get("enable_reranking", False))
        reranking_disabled_count = len(scoring_combinations) - reranking_enabled_count
        
        logger.info(f"Generated {len(scoring_combinations)} scoring combinations:")
        logger.info(f"  - {reranking_enabled_count} with reranking enabled")
        logger.info(f"  - {reranking_disabled_count} with reranking disabled")
        
        return scoring_combinations
    
    def run_scoring_on_collections(self, max_combinations: int = None, 
                                  metadata_file_path: str = None) -> pd.DataFrame:
        """
        Run scoring on pre-prepared collections
        
        Args:
            max_combinations: Maximum scoring combinations to test per collection
            metadata_file_path: Path to specific collections metadata file
            
        Returns:
            DataFrame with all results
        """
        # Ensure results directories exist
        Config.ensure_directories()
        
        # Load collections metadata
        collections_metadata = self.collection_preparer.load_collections_metadata(metadata_file_path)
        
        # Generate scoring combinations
        scoring_combinations = self.generate_scoring_combinations(max_combinations)
        
        print(f"🎯 Testing {len(scoring_combinations)} scoring combinations on {len(collections_metadata)} collections")
        print(f"📊 Total experiments: {len(collections_metadata) * len(scoring_combinations)}")
        
        all_results = []
        
        with tqdm(total=len(collections_metadata) * len(scoring_combinations), desc="Running scoring experiments") as pbar:
            for collection_meta in collections_metadata:
                collection_name = collection_meta["collection_name"]
                
                for scoring_combo in scoring_combinations:
                    try:
                        # Record start time for this experiment
                        experiment_start_time = time.time()
                        
                        # Create full parameter set
                        full_params = {
                            "chunk_size": collection_meta["chunk_size"],
                            "chunk_overlap": collection_meta["chunk_overlap"],
                            "embedding_model": collection_meta["embedding_model"],
                            **scoring_combo,
                            "collection_name": collection_name
                        }
                        
                        # Run evaluation on existing collection
                        metrics = self.run_rag_evaluation_on_existing_collection(full_params)
                        
                        # Calculate experiment runtime
                        experiment_runtime = time.time() - experiment_start_time
                        
                        if "error" not in metrics:
                            # Add runtime to metrics
                            metrics["experiment_runtime_seconds"] = experiment_runtime
                            metrics["experiment_runtime_minutes"] = experiment_runtime / 60.0
                            
                            # Log to MLflow
                            with mlflow.start_run():
                                # Log parameters
                                mlflow.log_params(full_params)
                                
                                # Log metrics (including runtime)
                                for metric_name, metric_value in metrics.items():
                                    mlflow.log_metric(metric_name, metric_value)
                                
                                # Store result
                                result = {**full_params, **metrics}
                                all_results.append(result)
                        else:
                            # Even for errors, record the runtime
                            metrics["experiment_runtime_seconds"] = experiment_runtime
                            metrics["experiment_runtime_minutes"] = experiment_runtime / 60.0
                            logger.warning(f"Experiment failed in {experiment_runtime:.2f}s: {metrics.get('error_message', 'Unknown error')}")
                        
                        pbar.update(1)
                        
                    except Exception as e:
                        experiment_runtime = time.time() - experiment_start_time if 'experiment_start_time' in locals() else 0
                        logger.error(f"Error in experiment for collection {collection_name} (runtime: {experiment_runtime:.2f}s): {e}")
                        pbar.update(1)
                        continue
        
        # Convert to DataFrame and save
        results_df = pd.DataFrame(all_results)
        
        if not results_df.empty:
            # Save to scoring results directory with timestamped filename
            results_file_path = self.save_scoring_results(results_df, collections_metadata)
            logger.info(f"Scoring results saved to {results_file_path}")
        
        return results_df
    
    def save_scoring_results(self, results_df: pd.DataFrame, 
                           collections_metadata: List[Dict[str, Any]]) -> str:
        """
        Save scoring results to timestamped file
        
        Args:
            results_df: Results DataFrame
            collections_metadata: Original collections metadata
            
        Returns:
            Path to saved results file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Extract prefix from collections metadata if available
        collection_prefix = "unknown"
        if collections_metadata and len(collections_metadata) > 0:
            first_collection = collections_metadata[0]
            collection_prefix = first_collection.get("collection_prefix", "unknown")
        
        results_filename = f"{collection_prefix}_scoring_results_{timestamp}.csv"
        results_file_path = Config.SCORING_RESULTS_DIR / results_filename
        
        # Save the results
        results_df.to_csv(results_file_path, index=False)
        
        # Also save a summary JSON file with basic info
        summary_filename = f"{collection_prefix}_scoring_summary_{timestamp}.json"
        summary_file_path = Config.SCORING_RESULTS_DIR / summary_filename
        
        summary_data = {
            "summary": {
                "total_experiments": len(results_df),
                "collections_tested": len(collections_metadata),
                "scoring_combinations": len(results_df) // len(collections_metadata) if collections_metadata else 0,
                "reranking_experiments": len(results_df[results_df.get("reranking_enabled", False)]) if "reranking_enabled" in results_df.columns else 0,
                "non_reranking_experiments": len(results_df[~results_df.get("reranking_enabled", False)]) if "reranking_enabled" in results_df.columns else len(results_df),
                "average_runtime_seconds": float(results_df["total_evaluation_time_seconds"].mean()) if "total_evaluation_time_seconds" in results_df.columns else None,
                "total_runtime_hours": float(results_df["total_evaluation_time_seconds"].sum() / 3600) if "total_evaluation_time_seconds" in results_df.columns else None,
                "created_at": datetime.now().isoformat(),
                "results_file": results_filename
            },
            "experiment_metadata": {
                "collection_prefix": collection_prefix,
                "benchmark_file": self.benchmark_file,
                "mlflow_experiment": self.experiment_name,
                "ragas_metrics": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
                "reranking_models_tested": list(results_df["reranking_model_used"].dropna().unique()) if "reranking_model_used" in results_df.columns else [],
                "reranking_configurations_count": len(results_df[results_df.get("reranking_enabled", False)].drop_duplicates(subset=[
                    "reranking_model_used", "rerank_hybrid_scoring", "rerank_cross_encoder_weight", "rerank_embedding_weight"
                ])) if "reranking_enabled" in results_df.columns else 0
            }
        }
        
        with open(summary_file_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info(f"Scoring summary saved to {summary_file_path}")
        return str(results_file_path)
    
    def run_rag_evaluation_on_existing_collection(self, params: Dict[str, Any]) -> Dict[str, float]:
        """
        Run RAG evaluation on an existing collection
        
        Args:
            params: Parameter dictionary including collection_name
            
        Returns:
            Dictionary of evaluation metrics including detailed runtime breakdown
        """
        # Track detailed timing for different phases
        timing_start = time.time()
        phase_timings = {}
        
        try:
            collection_name = params["collection_name"]
            
            # Phase 1: Collection verification
            verification_start = time.time()
            try:
                from qdrant_client import QdrantClient
                client = QdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY)
                collection_info = client.get_collection(collection_name)
                
                if collection_info.points_count == 0:
                    logger.error(f"Collection {collection_name} is empty (0 points)")
                    return {"error": 1.0, "composite_score": 0.0, "error_message": "Empty collection"}
                
                logger.info(f"Collection {collection_name} has {collection_info.points_count} documents")
                
            except Exception as e:
                logger.error(f"Could not verify collection {collection_name}: {e}")
                return {"error": 1.0, "composite_score": 0.0, "error_message": f"Collection access error: {str(e)}"}
            
            phase_timings["collection_verification_seconds"] = time.time() - verification_start
            
            # Phase 2: RAG system initialization
            init_start = time.time()
            rag_init_params = {
                "collection_name": collection_name,
                "embedding_model": params["embedding_model"],
                "llm_model": params["llm_model"],
                "temperature": params["temperature"],
                "top_k": params["top_k"],
                "prompt_template": params["prompt_template"],
                "score_threshold": params["score_threshold"],
                "enable_reranking": params.get("enable_reranking", False)
            }
            
            # Add reranking model if reranking is enabled
            if params.get("enable_reranking", False) and params.get("reranking_model"):
                rag_init_params["reranking_model"] = params["reranking_model"]
            
            rag = QdrantRAG(**rag_init_params)
            
            # Apply reranking configuration if enabled
            if params.get("enable_reranking", False):
                rerank_config_params = {}
                if params.get("rerank_top_k_before") is not None:
                    rerank_config_params["top_k_before_rerank"] = params["rerank_top_k_before"]
                if params.get("rerank_hybrid_scoring") is not None:
                    rerank_config_params["enable_hybrid_scoring"] = params["rerank_hybrid_scoring"]
                if params.get("rerank_cross_encoder_weight") is not None:
                    rerank_config_params["cross_encoder_weight"] = params["rerank_cross_encoder_weight"]
                if params.get("rerank_embedding_weight") is not None:
                    rerank_config_params["embedding_weight"] = params["rerank_embedding_weight"]
                
                if rerank_config_params:
                    rag.update_reranking_config(rerank_config_params)
                    logger.debug(f"Applied reranking config: {rerank_config_params}")
            
            phase_timings["rag_initialization_seconds"] = time.time() - init_start
            
            # Phase 3: Question processing and RAG evaluation
            rag_processing_start = time.time()
            results = []
            zero_context_count = 0
            question_timings = []
            
            for _, row in self.benchmark_data.iterrows():
                question = row["question"]
                ground_truth = row["ground_truth"]
                
                question_start = time.time()
                try:
                    rag_result = rag.answer_question(question)
                    question_time = time.time() - question_start
                    question_timings.append(question_time)
                    
                    # Check if we retrieved any documents
                    if rag_result["retrieved_documents"] == 0:
                        zero_context_count += 1
                        logger.warning(f"Zero documents retrieved for question: '{question[:50]}...'")
                    
                    result = {
                        "question": question,
                        "answer": rag_result["answer"],
                        "contexts": [rag_result["context"]],
                        "ground_truth": ground_truth
                    }
                    results.append(result)
                    
                except Exception as e:
                    question_time = time.time() - question_start
                    question_timings.append(question_time)
                    logger.warning(f"Error processing question '{question}': {e}")
                    # Add placeholder result to maintain consistency
                    result = {
                        "question": question,
                        "answer": "Error processing question",
                        "contexts": [""],
                        "ground_truth": ground_truth
                    }
                    results.append(result)
            
            phase_timings["rag_processing_seconds"] = time.time() - rag_processing_start
            phase_timings["average_question_time_seconds"] = np.mean(question_timings) if question_timings else 0
            phase_timings["total_questions_processed"] = len(question_timings)
            
            if zero_context_count > 0:
                logger.warning(f"Collection {collection_name}: {zero_context_count}/{len(self.benchmark_data)} questions had zero document retrieval")
            
            # Phase 4: RAGAs evaluation
            ragas_start = time.time()
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
            phase_timings["ragas_evaluation_seconds"] = time.time() - ragas_start
            
            # Extract metrics
            metrics = {
                "faithfulness": float(evaluation_result["faithfulness"]),
                "answer_relevancy": float(evaluation_result["answer_relevancy"]),
                "context_precision": float(evaluation_result["context_precision"]),
                "context_recall": float(evaluation_result["context_recall"])
            }
            
            # Calculate composite score
            metrics["composite_score"] = np.mean(list(metrics.values()))
            
            # Add all timing information
            total_runtime = time.time() - timing_start
            metrics.update(phase_timings)
            metrics["total_evaluation_time_seconds"] = total_runtime
            metrics["total_evaluation_time_minutes"] = total_runtime / 60.0
            
            # Add reranking statistics if reranking was enabled
            if params.get("enable_reranking", False):
                metrics["reranking_enabled"] = True
                metrics["reranking_model_used"] = params.get("reranking_model", "unknown")
                
                # Add reranking configuration metrics
                if params.get("rerank_hybrid_scoring") is not None:
                    metrics["rerank_hybrid_scoring"] = bool(params["rerank_hybrid_scoring"])
                if params.get("rerank_cross_encoder_weight") is not None:
                    metrics["rerank_cross_encoder_weight"] = float(params["rerank_cross_encoder_weight"])
                if params.get("rerank_embedding_weight") is not None:
                    metrics["rerank_embedding_weight"] = float(params["rerank_embedding_weight"])
                if params.get("rerank_top_k_before") is not None:
                    metrics["rerank_top_k_before"] = int(params["rerank_top_k_before"])
                
                # Try to get reranking statistics from the RAG system if available
                if hasattr(rag, 'reranker') and rag.reranker is not None:
                    metrics["reranker_available"] = True
                    if hasattr(rag.reranker, 'config'):
                        metrics["reranker_model_loaded"] = rag.reranker.config.model_name
                else:
                    metrics["reranker_available"] = False
            else:
                metrics["reranking_enabled"] = False
                metrics["reranking_model_used"] = None
            
            return metrics
            
        except Exception as e:
            error_msg = str(e)
            total_runtime = time.time() - timing_start
            logger.error(f"Error in RAG evaluation on existing collection {params.get('collection_name', 'unknown')} (runtime: {total_runtime:.2f}s): {error_msg}")
            
            # Include timing even for errors
            error_result = {
                "error": 1.0, 
                "composite_score": 0.0, 
                "total_evaluation_time_seconds": total_runtime,
                "total_evaluation_time_minutes": total_runtime / 60.0
            }
            
            # Add any phase timings that were completed
            error_result.update(phase_timings)
            
            # Provide more specific error information
            if "collection" in error_msg.lower() and "not found" in error_msg.lower():
                error_result["error_message"] = "Collection not found"
            elif "openai" in error_msg.lower() or "api" in error_msg.lower():
                error_result["error_message"] = "API error"
            else:
                error_result["error_message"] = f"Evaluation error: {error_msg[:100]}"
                
            return error_result
    
    def run_single_evaluation(self, collection_name: str, scoring_params: Dict[str, Any]) -> Dict[str, float]:
        """
        Run RAGAs evaluation on a single collection with specific parameters
        
        Args:
            collection_name: Name of the collection to evaluate
            scoring_params: Dictionary with llm_model, temperature, top_k, prompt_template, embedding_model
            
        Returns:
            Dictionary of evaluation metrics
        """
        full_params = {
            "collection_name": collection_name,
            **scoring_params
        }
        
        return self.run_rag_evaluation_on_existing_collection(full_params)
    
    def list_available_results(self) -> List[Dict[str, Any]]:
        """
        List all available scoring results files
        
        Returns:
            List of results file information
        """
        results_files = []
        
        # Check scoring results directory
        csv_files = list(Config.SCORING_RESULTS_DIR.glob("*_scoring_results_*.csv"))
        
        for file_path in csv_files:
            try:
                # Try to find corresponding summary file
                summary_file = file_path.with_suffix('').with_suffix('').parent / f"{file_path.stem.replace('_results_', '_summary_')}.json"
                
                if summary_file.exists():
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                    
                    results_files.append({
                        "results_file": str(file_path),
                        "summary_file": str(summary_file),
                        "created_at": summary.get("summary", {}).get("created_at"),
                        "total_experiments": summary.get("summary", {}).get("total_experiments"),
                        "collections_tested": summary.get("summary", {}).get("collections_tested"),
                        "file_size": file_path.stat().st_size
                    })
                else:
                    # No summary file, basic info only
                    results_files.append({
                        "results_file": str(file_path),
                        "summary_file": None,
                        "created_at": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                        "total_experiments": None,
                        "collections_tested": None,
                        "file_size": file_path.stat().st_size
                    })
            except Exception as e:
                logger.warning(f"Could not read results file {file_path}: {e}")
        
        # Sort by creation time, newest first
        results_files.sort(key=lambda x: x["created_at"], reverse=True)
        
        return results_files
    
    def get_scoring_stats(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics about scoring results
        
        Args:
            results_df: Results DataFrame
            
        Returns:
            Dictionary with scoring statistics
        """
        if results_df.empty:
            return {"error": "No results to analyze"}
        
        # Filter out error runs
        valid_results = results_df[~results_df.get("error", False)]
        
        if valid_results.empty:
            return {"error": "No valid results found"}
        
        stats = {
            "total_experiments": len(results_df),
            "valid_experiments": len(valid_results),
            "error_rate": (len(results_df) - len(valid_results)) / len(results_df),
            "composite_score_stats": {
                "mean": float(valid_results["composite_score"].mean()),
                "std": float(valid_results["composite_score"].std()),
                "min": float(valid_results["composite_score"].min()),
                "max": float(valid_results["composite_score"].max()),
                "median": float(valid_results["composite_score"].median())
            },
            "metric_correlations": {},
            "reranking_analysis": {}
        }
        
        # Add reranking analysis if reranking data is available
        if "reranking_enabled" in valid_results.columns:
            rerank_enabled = valid_results[valid_results["reranking_enabled"] == True]
            rerank_disabled = valid_results[valid_results["reranking_enabled"] == False]
            
            stats["reranking_analysis"] = {
                "total_reranking_experiments": len(rerank_enabled),
                "total_non_reranking_experiments": len(rerank_disabled),
                "reranking_performance": {},
                "reranking_models_tested": list(valid_results["reranking_model_used"].dropna().unique()) if "reranking_model_used" in valid_results.columns else []
            }
            
            # Compare performance between reranking enabled/disabled
            if len(rerank_enabled) > 0 and len(rerank_disabled) > 0:
                stats["reranking_analysis"]["reranking_performance"] = {
                    "reranking_enabled_mean_score": float(rerank_enabled["composite_score"].mean()),
                    "reranking_disabled_mean_score": float(rerank_disabled["composite_score"].mean()),
                    "reranking_improvement": float(rerank_enabled["composite_score"].mean() - rerank_disabled["composite_score"].mean()),
                    "reranking_enabled_best_score": float(rerank_enabled["composite_score"].max()),
                    "reranking_disabled_best_score": float(rerank_disabled["composite_score"].max())
                }
                
                # Metric-specific improvements
                metric_columns = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
                for metric in metric_columns:
                    if metric in valid_results.columns:
                        rerank_mean = rerank_enabled[metric].mean()
                        no_rerank_mean = rerank_disabled[metric].mean()
                        improvement = rerank_mean - no_rerank_mean
                        stats["reranking_analysis"]["reranking_performance"][f"{metric}_improvement"] = float(improvement)
            
            # Analyze different reranking models if multiple were tested
            if "reranking_model_used" in valid_results.columns:
                model_performance = {}
                for model in rerank_enabled["reranking_model_used"].unique():
                    if pd.notna(model):
                        model_results = rerank_enabled[rerank_enabled["reranking_model_used"] == model]
                        if len(model_results) > 0:
                            model_performance[model] = {
                                "experiments_count": len(model_results),
                                "mean_composite_score": float(model_results["composite_score"].mean()),
                                "best_composite_score": float(model_results["composite_score"].max())
                            }
                
                stats["reranking_analysis"]["model_performance"] = model_performance
        
        # Calculate correlations between individual metrics and composite score
        metric_columns = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        for metric in metric_columns:
            if metric in valid_results.columns:
                correlation = valid_results[metric].corr(valid_results["composite_score"])
                stats["metric_correlations"][metric] = float(correlation)
        
        return stats
    
    def analyze_reranking_impact(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the impact of reranking on performance
        
        Args:
            results_df: Results DataFrame with reranking experiments
            
        Returns:
            Dictionary with detailed reranking analysis
        """
        if results_df.empty or "reranking_enabled" not in results_df.columns:
            return {"error": "No reranking data available for analysis"}
        
        # Filter out error runs
        valid_results = results_df[~results_df.get("error", False)]
        
        if valid_results.empty:
            return {"error": "No valid results found"}
        
        rerank_enabled = valid_results[valid_results["reranking_enabled"] == True]
        rerank_disabled = valid_results[valid_results["reranking_enabled"] == False]
        
        analysis = {
            "summary": {
                "total_experiments": len(valid_results),
                "reranking_experiments": len(rerank_enabled),
                "non_reranking_experiments": len(rerank_disabled)
            },
            "performance_comparison": {},
            "reranking_configurations": {},
            "best_configurations": {}
        }
        
        # Performance comparison
        if len(rerank_enabled) > 0 and len(rerank_disabled) > 0:
            metrics = ["composite_score", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]
            
            performance_comparison = {}
            for metric in metrics:
                if metric in valid_results.columns:
                    rerank_mean = rerank_enabled[metric].mean()
                    no_rerank_mean = rerank_disabled[metric].mean()
                    improvement = rerank_mean - no_rerank_mean
                    improvement_pct = (improvement / no_rerank_mean) * 100 if no_rerank_mean != 0 else 0
                    
                    performance_comparison[metric] = {
                        "reranking_enabled_mean": float(rerank_mean),
                        "reranking_disabled_mean": float(no_rerank_mean),
                        "absolute_improvement": float(improvement),
                        "percentage_improvement": float(improvement_pct),
                        "reranking_enabled_best": float(rerank_enabled[metric].max()),
                        "reranking_disabled_best": float(rerank_disabled[metric].max())
                    }
            
            analysis["performance_comparison"] = performance_comparison
        
        # Analyze different reranking configurations
        if len(rerank_enabled) > 0:
            config_columns = [
                "reranking_model_used", "rerank_hybrid_scoring", 
                "rerank_cross_encoder_weight", "rerank_embedding_weight"
            ]
            
            # Group by reranking configuration
            available_config_cols = [col for col in config_columns if col in rerank_enabled.columns]
            
            if available_config_cols:
                config_groups = rerank_enabled.groupby(available_config_cols)
                
                config_performance = {}
                for name, group in config_groups:
                    config_key = str(name) if len(available_config_cols) > 1 else str(name)
                    config_performance[config_key] = {
                        "experiments_count": len(group),
                        "mean_composite_score": float(group["composite_score"].mean()),
                        "best_composite_score": float(group["composite_score"].max()),
                        "configuration": dict(zip(available_config_cols, name if isinstance(name, tuple) else [name]))
                    }
                
                analysis["reranking_configurations"] = config_performance
        
        # Find best overall configurations
        if len(valid_results) > 0:
            # Best reranking configuration
            if len(rerank_enabled) > 0:
                best_rerank = rerank_enabled.loc[rerank_enabled["composite_score"].idxmax()]
                analysis["best_configurations"]["best_with_reranking"] = {
                    "composite_score": float(best_rerank["composite_score"]),
                    "configuration": {
                        "reranking_model": best_rerank.get("reranking_model_used"),
                        "hybrid_scoring": best_rerank.get("rerank_hybrid_scoring"),
                        "cross_encoder_weight": best_rerank.get("rerank_cross_encoder_weight"),
                        "embedding_weight": best_rerank.get("rerank_embedding_weight"),
                        "llm_model": best_rerank.get("llm_model"),
                        "temperature": best_rerank.get("temperature"),
                        "top_k": best_rerank.get("top_k")
                    }
                }
            
            # Best non-reranking configuration
            if len(rerank_disabled) > 0:
                best_no_rerank = rerank_disabled.loc[rerank_disabled["composite_score"].idxmax()]
                analysis["best_configurations"]["best_without_reranking"] = {
                    "composite_score": float(best_no_rerank["composite_score"]),
                    "configuration": {
                        "llm_model": best_no_rerank.get("llm_model"),
                        "temperature": best_no_rerank.get("temperature"),
                        "top_k": best_no_rerank.get("top_k"),
                        "score_threshold": best_no_rerank.get("score_threshold")
                    }
                }
            
            # Overall best configuration
            best_overall = valid_results.loc[valid_results["composite_score"].idxmax()]
            analysis["best_configurations"]["best_overall"] = {
                "composite_score": float(best_overall["composite_score"]),
                "uses_reranking": bool(best_overall.get("reranking_enabled", False)),
                "configuration": {
                    "reranking_enabled": bool(best_overall.get("reranking_enabled", False)),
                    "reranking_model": best_overall.get("reranking_model_used") if best_overall.get("reranking_enabled") else None,
                    "llm_model": best_overall.get("llm_model"),
                    "temperature": best_overall.get("temperature"),
                    "top_k": best_overall.get("top_k")
                }
            }
        
        return analysis
