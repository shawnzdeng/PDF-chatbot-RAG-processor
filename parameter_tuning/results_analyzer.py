"""
Results analysis and production config generation for RAG parameter tuning
Handles analyzing scoring results and generating optimized production configurations
"""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Local imports
from config import Config

logger = logging.getLogger(__name__)

class ResultsAnalyzer:
    """Analyzes scoring results and generates production configurations"""
    
    def __init__(self, config_file: str = None):
        """
        Initialize results analyzer
        
        Args:
            config_file: Path to parameters configuration JSON
        """
        self.config_file = config_file or str(Config.BASE_DIR / "parameters_config.json")
        self.config = self._load_config()
        
        logger.info("Initialized ResultsAnalyzer")
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj
    
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
    
    def load_results_from_file(self, results_file_path: str) -> pd.DataFrame:
        """
        Load scoring results from CSV file
        
        Args:
            results_file_path: Path to results CSV file
            
        Returns:
            Results DataFrame
        """
        try:
            results_df = pd.read_csv(results_file_path)
            logger.info(f"Loaded {len(results_df)} results from {results_file_path}")
            return results_df
        except Exception as e:
            logger.error(f"Error loading results from {results_file_path}: {e}")
            raise
    
    def load_results_from_mlflow(self, experiment_name: str = None) -> pd.DataFrame:
        """
        Load scoring results directly from MLflow experiments
        
        Args:
            experiment_name: Name of MLflow experiment (uses default if None)
            
        Returns:
            Results DataFrame
        """
        try:
            import mlflow
            
            # Set MLflow tracking URI
            mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
            
            # Get experiment
            if experiment_name:
                try:
                    experiment = mlflow.get_experiment_by_name(experiment_name)
                    if not experiment:
                        logger.error(f"Experiment '{experiment_name}' not found")
                        return pd.DataFrame()
                except:
                    logger.error(f"Error finding experiment '{experiment_name}'")
                    return pd.DataFrame()
            else:
                # Get the default experiment or the first available
                experiments = mlflow.search_experiments()
                if not experiments:
                    logger.error("No MLflow experiments found")
                    return pd.DataFrame()
                experiment = experiments[0]
                logger.info(f"Using experiment: {experiment.name}")
            
            # Get all runs
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            
            if runs.empty:
                logger.error("No runs found in MLflow experiment")
                return pd.DataFrame()
            
            logger.info(f"Found {len(runs)} runs in MLflow experiment")
            
            # Debug: print available columns
            logger.info(f"Available columns in MLflow runs: {list(runs.columns)}")
            
            # Convert MLflow runs to the format expected by analyzer
            csv_data = []
            
            for idx, run in runs.iterrows():
                # Extract parameters
                params = {}
                param_cols = [col for col in runs.columns if col.startswith('params.')]
                for param_col in param_cols:
                    param_name = param_col.replace('params.', '')
                    param_value = run.get(param_col)
                    if pd.notna(param_value) and param_value != 'None':
                        # Convert string numbers to appropriate types
                        try:
                            if param_name in ['chunk_size', 'chunk_overlap', 'top_k', 'rerank_top_k_before']:
                                params[param_name] = int(param_value)
                            elif param_name in ['temperature', 'rerank_embedding_weight', 'rerank_cross_encoder_weight', 'score_threshold']:
                                params[param_name] = float(param_value)
                            elif param_name in ['enable_reranking', 'rerank_hybrid_scoring']:
                                params[param_name] = param_value.lower() == 'true' if isinstance(param_value, str) else bool(param_value)
                            else:
                                params[param_name] = param_value
                        except:
                            params[param_name] = param_value
                
                # Extract metrics
                metrics = {}
                metric_cols = [col for col in runs.columns if col.startswith('metrics.')]
                for metric_col in metric_cols:
                    metric_name = metric_col.replace('metrics.', '')
                    metric_value = run.get(metric_col)
                    if pd.notna(metric_value):
                        metrics[metric_name] = float(metric_value)
                
                # Combine parameters and metrics
                row_data = {**params, **metrics}
                
                # Add collection name if not present (required by analyzer)
                if 'collection_name' not in row_data:
                    row_data['collection_name'] = Config.DEFAULT_COLLECTION_NAME
                
                csv_data.append(row_data)
            
            # Create DataFrame
            results_df = pd.DataFrame(csv_data)
            
            logger.info(f"Successfully loaded {len(results_df)} results from MLflow")
            return results_df
            
        except ImportError:
            logger.error("MLflow not available. Please install mlflow: pip install mlflow")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading results from MLflow: {e}")
            return pd.DataFrame()
    
    def get_best_parameters(self, results_df: pd.DataFrame, 
                           metric: str = "composite_score") -> Dict[str, Any]:
        """
        Get best parameters from results based on specified metric
        
        Args:
            results_df: Results DataFrame
            metric: Metric to optimize for (default: composite_score)
            
        Returns:
            Best parameter configuration
        """
        if results_df.empty:
            logger.error("No valid results found")
            return {}

        if metric not in results_df.columns:
            logger.error(f"Metric '{metric}' not found in results")
            return {}

        # Find best score
        best_idx = results_df[metric].idxmax()
        best_params = results_df.loc[best_idx]
        
        # Extract parameter values - safely handle missing keys
        param_keys = ["chunk_size", "chunk_overlap", "embedding_model", "llm_model", 
                     "temperature", "top_k", "prompt_template", "enable_reranking", "rerank_embedding_weight", 
                     "rerank_cross_encoder_weight", "rerank_top_k_before", "rerank_hybrid_scoring", "reranking_model", "score_threshold"]
        best_config = {}
        for key in param_keys:
            if key in best_params and pd.notna(best_params[key]):
                best_config[key] = best_params[key]
        
        # Add collection information for production use
        best_config["collection_name"] = best_params.get("collection_name", Config.DEFAULT_COLLECTION_NAME)
        best_config["qdrant_url"] = Config.QDRANT_URL
        
        # Add all metrics - safely handle missing keys
        metric_keys = ["faithfulness", "answer_relevancy", "context_precision", 
                      "context_recall", "composite_score"]
        best_config["metrics"] = {}
        for key in metric_keys:
            if key in best_params and pd.notna(best_params[key]):
                try:
                    best_config["metrics"][key] = float(best_params[key])
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert metric '{key}' to float: {best_params[key]}")
                    best_config["metrics"][key] = 0.0
        
        # Add metadata for production integration
        best_config["optimization_metadata"] = {
            "tuning_date": datetime.now().isoformat(),
            "total_combinations_tested": len(results_df),
            "optimization_metric": metric,
            "optimization_framework": "MLflow + RAGAs",
            "ready_for_production": True
        }
        
        logger.info(f"Best parameters found with {metric}: {best_config['metrics'].get(metric, 0):.3f}")
        
        return best_config
    
    def get_top_n_parameters(self, results_df: pd.DataFrame, n: int = 5, 
                            metric: str = "composite_score") -> List[Dict[str, Any]]:
        """
        Get top N parameter configurations
        
        Args:
            results_df: Results DataFrame
            n: Number of top configurations to return
            metric: Metric to optimize for
            
        Returns:
            List of top parameter configurations
        """
        if results_df.empty:
            logger.error("No valid results found")
            return []

        if metric not in results_df.columns:
            logger.error(f"Metric '{metric}' not found in results")
            return []

        # Get top N results
        top_results = results_df.nlargest(n, metric)
        
        top_configs = []
        for idx, row in top_results.iterrows():
            param_keys = ["chunk_size", "chunk_overlap", "embedding_model", "llm_model", 
                         "temperature", "top_k", "prompt_template", "enable_reranking", "rerank_embedding_weight", 
                         "rerank_cross_encoder_weight", "rerank_top_k_before", "rerank_hybrid_scoring", "reranking_model", "score_threshold"]
            config = {}
            for key in param_keys:
                if key in row and pd.notna(row[key]):
                    config[key] = row[key]
            
            # Add collection and metrics info
            config["collection_name"] = row.get("collection_name", Config.DEFAULT_COLLECTION_NAME)
            config["rank"] = len(top_configs) + 1
            
            metric_keys = ["faithfulness", "answer_relevancy", "context_precision", 
                          "context_recall", "composite_score"]
            config["metrics"] = {}
            for key in metric_keys:
                if key in row and pd.notna(row[key]):
                    try:
                        config["metrics"][key] = float(row[key])
                    except (ValueError, TypeError):
                        config["metrics"][key] = 0.0
            
            top_configs.append(config)
        
        logger.info(f"Retrieved top {len(top_configs)} configurations based on {metric}")
        return top_configs
    
    def analyze_parameter_impact(self, results_df: pd.DataFrame, 
                                metric: str = "composite_score") -> Dict[str, Any]:
        """
        Analyze the impact of different parameters on performance
        
        Args:
            results_df: Results DataFrame
            metric: Metric to analyze
            
        Returns:
            Analysis of parameter impacts
        """
        if results_df.empty or metric not in results_df.columns:
            return {"error": "No valid results to analyze"}

        analysis = {
            "metric_analyzed": metric,
            "total_valid_experiments": len(results_df),
            "parameter_impacts": {}
        }
        
        # Analyze impact of each parameter
        categorical_params = ["embedding_model", "llm_model", "prompt_template", "enable_reranking", "rerank_hybrid_scoring", "reranking_model"]
        numerical_params = ["chunk_size", "chunk_overlap", "temperature", "top_k", "rerank_embedding_weight", 
                           "rerank_cross_encoder_weight", "rerank_top_k_before", "score_threshold"]
        
        # Categorical parameters - group by value and calculate statistics
        for param in categorical_params:
            if param in results_df.columns:
                grouped = results_df.groupby(param)[metric].agg(['mean', 'std', 'count', 'min', 'max'])
                analysis["parameter_impacts"][param] = {
                    "type": "categorical",
                    "values": {
                        str(value): {
                            "mean": float(stats['mean']),
                            "std": float(stats['std']),
                            "count": int(stats['count']),
                            "min": float(stats['min']),
                            "max": float(stats['max'])
                        } for value, stats in grouped.iterrows()
                    }
                }
        
        # Numerical parameters - calculate correlations and statistics
        for param in numerical_params:
            if param in results_df.columns:
                try:
                    # Check if the column has numeric data only
                    numeric_series = pd.to_numeric(results_df[param], errors='coerce')
                    if not numeric_series.isna().all():
                        correlation = numeric_series.corr(results_df[metric])
                        analysis["parameter_impacts"][param] = {
                            "type": "numerical",
                            "correlation": float(correlation) if not pd.isna(correlation) else 0.0,
                            "statistics": {
                                "mean": float(numeric_series.mean()),
                                "std": float(numeric_series.std()),
                                "min": float(numeric_series.min()),
                                "max": float(numeric_series.max())
                            }
                        }
                except Exception as e:
                    logger.warning(f"Could not calculate correlation for parameter '{param}': {e}")
                    # Skip this parameter if correlation calculation fails
        
        return analysis
    
    def generate_performance_summary(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive performance summary
        
        Args:
            results_df: Results DataFrame
            
        Returns:
            Performance summary
        """
        summary = {
            "experiment_overview": {
                "total_experiments": len(results_df),
                "valid_experiments": len(results_df),
                "error_rate": 0,
                "analysis_date": datetime.now().isoformat()
            },
            "metric_statistics": {},
            "best_performance": {},
            "parameter_variety": {}
        }
        
        if results_df.empty:
            summary["error"] = "No valid results to analyze"
            return summary
        
        # Metric statistics
        metric_columns = ["faithfulness", "answer_relevancy", "context_precision", 
                         "context_recall", "composite_score"]
        
        for metric in metric_columns:
            if metric in results_df.columns:
                summary["metric_statistics"][metric] = {
                    "mean": float(results_df[metric].mean()),
                    "std": float(results_df[metric].std()),
                    "min": float(results_df[metric].min()),
                    "max": float(results_df[metric].max()),
                    "median": float(results_df[metric].median()),
                    "q25": float(results_df[metric].quantile(0.25)),
                    "q75": float(results_df[metric].quantile(0.75))
                }
        
        # Best performance for each metric
        for metric in metric_columns:
            if metric in results_df.columns:
                best_idx = results_df[metric].idxmax()
                best_row = results_df.loc[best_idx]
                best_performance = {
                    "value": float(best_row[metric]),
                    "chunk_size": int(best_row.get("chunk_size", 0)),
                    "chunk_overlap": int(best_row.get("chunk_overlap", 0)),
                    "temperature": float(best_row.get("temperature", 0)),
                    "top_k": int(best_row.get("top_k", 0))
                }
                
                # Only add reranker fields if they exist in the data
                if "enable_reranking" in best_row:
                    best_performance["enable_reranking"] = best_row.get("enable_reranking", False)
                if "rerank_embedding_weight" in best_row:
                    best_performance["rerank_embedding_weight"] = float(best_row.get("rerank_embedding_weight", 0))
                if "rerank_cross_encoder_weight" in best_row:
                    best_performance["rerank_cross_encoder_weight"] = float(best_row.get("rerank_cross_encoder_weight", 0))
                if "rerank_top_k_before" in best_row:
                    best_performance["rerank_top_k_before"] = int(best_row.get("rerank_top_k_before", 0))
                if "rerank_hybrid_scoring" in best_row:
                    best_performance["rerank_hybrid_scoring"] = best_row.get("rerank_hybrid_scoring", False)
                if "reranking_model" in best_row:
                    best_performance["reranking_model"] = best_row.get("reranking_model", "")
                if "score_threshold" in best_row:
                    best_performance["score_threshold"] = float(best_row.get("score_threshold", 0))
                
                summary["best_performance"][metric] = best_performance
        
        # Parameter variety
        param_columns = ["chunk_size", "chunk_overlap", "embedding_model", "llm_model", 
                        "temperature", "top_k", "enable_reranking", "rerank_embedding_weight", 
                        "rerank_cross_encoder_weight", "rerank_top_k_before", "rerank_hybrid_scoring", "reranking_model", "score_threshold"]
        
        for param in param_columns:
            if param in results_df.columns:
                unique_values = results_df[param].nunique()
                summary["parameter_variety"][param] = {
                    "unique_values": int(unique_values),
                    "total_experiments": len(results_df)
                }
        
        return summary
    
    def save_best_parameters(self, best_params: Dict[str, Any], filename: str = None) -> str:
        """
        Save best parameters to file in scoring results directory
        
        Args:
            best_params: Best parameter configuration
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        Config.ensure_directories()
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"best_parameters_{timestamp}.json"
        
        file_path = Config.SCORING_RESULTS_DIR / filename
        
        # Convert numpy types to native Python types for JSON serialization
        json_safe_params = self._convert_numpy_types(best_params)
        
        with open(file_path, 'w') as f:
            json.dump(json_safe_params, f, indent=2)
        
        logger.info(f"Best parameters saved to {file_path}")
        return str(file_path)
    
    def export_production_config(self, best_params: Dict[str, Any], export_path: str = None) -> str:
        """
        Export optimized parameters in a format ready for production RAG system
        
        Args:
            best_params: Best parameter configuration
            export_path: Path to save the production config
            
        Returns:
            Path to the exported configuration file
        """
        Config.ensure_directories()
        
        if export_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_filename = f"production_rag_config_{timestamp}.json"
            export_path = str(Config.PRODUCTION_CONFIG_DIR / export_filename)
        
        production_config = {
            "document_metadata": {
                "file_description": Config.get_file_description(self.config),
                "example_questions": Config.get_example_questions(self.config)
            },
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
        }
        
        # Add reranker config only if reranker parameters exist in the data
        if "enable_reranking" in best_params:
            reranker_config = {"enabled": best_params["enable_reranking"]}
            
            # Add other reranker parameters if they exist
            if "rerank_embedding_weight" in best_params:
                reranker_config["embedding_weight"] = best_params["rerank_embedding_weight"]
            if "rerank_cross_encoder_weight" in best_params:
                reranker_config["cross_encoder_weight"] = best_params["rerank_cross_encoder_weight"]
            if "rerank_top_k_before" in best_params:
                reranker_config["top_k_before_rerank"] = best_params["rerank_top_k_before"]
            if "rerank_hybrid_scoring" in best_params:
                reranker_config["hybrid_scoring"] = best_params["rerank_hybrid_scoring"]
            if "reranking_model" in best_params:
                reranker_config["model"] = best_params["reranking_model"]
            if "score_threshold" in best_params:
                reranker_config["score_threshold"] = best_params["score_threshold"]
                
            production_config["reranker_config"] = reranker_config
        
        # Build integration notes based on available data
        integration_notes = [
            "Use these exact parameters for optimal performance",
            "Collection contains pre-processed document embeddings",
            "Metrics show performance on evaluation benchmark",
        ]
        
        if "enable_reranking" in best_params:
            integration_notes.append("Reranker configuration optimized for document relevance")
        
        integration_notes.append("Ready for production deployment")
        
        # Build deployment checklist based on available data
        deployment_checklist = [
            "Verify collection exists in Qdrant",
            "Confirm environment variables are set",
        ]
        
        if best_params.get("enable_reranking", False):
            deployment_checklist.append("Install reranker dependencies if reranking is enabled")
        
        deployment_checklist.extend([
            "Test with sample queries",
            "Monitor performance metrics"
        ])
        
        production_config["usage_instructions"] = {
            "description": "Optimized RAG parameters from MLflow tuning with RAGAs evaluation",
            "integration_notes": integration_notes,
            "required_environment_variables": [
                "OPENAI_API_KEY",
                "QDRANT_API_KEY"
            ],
            "deployment_checklist": deployment_checklist
        }
        
        # Convert numpy types to native Python types for JSON serialization
        json_safe_config = self._convert_numpy_types(production_config)
        
        with open(export_path, 'w') as f:
            json.dump(json_safe_config, f, indent=2)
        
        logger.info(f"Production config exported to {export_path}")
        return export_path
    
    def compare_configurations(self, configs: List[Dict[str, Any]], 
                              metric: str = "composite_score") -> Dict[str, Any]:
        """
        Compare multiple configurations
        
        Args:
            configs: List of configuration dictionaries
            metric: Metric to compare
            
        Returns:
            Comparison analysis
        """
        if not configs:
            return {"error": "No configurations to compare"}
        
        comparison = {
            "metric_compared": metric,
            "total_configurations": len(configs),
            "comparison_date": datetime.now().isoformat(),
            "configurations": []
        }
        
        for i, config in enumerate(configs):
            config_info = {
                "rank": i + 1,
                "score": config.get("metrics", {}).get(metric, 0),
                "parameters": {
                    "chunk_size": config.get("chunk_size"),
                    "chunk_overlap": config.get("chunk_overlap"),
                    "embedding_model": config.get("embedding_model"),
                    "llm_model": config.get("llm_model"),
                    "temperature": config.get("temperature"),
                    "top_k": config.get("top_k"),
                    "enable_reranking": config.get("enable_reranking"),
                    "rerank_embedding_weight": config.get("rerank_embedding_weight"),
                    "rerank_cross_encoder_weight": config.get("rerank_cross_encoder_weight"),
                    "rerank_top_k_before": config.get("rerank_top_k_before"),
                    "rerank_hybrid_scoring": config.get("rerank_hybrid_scoring"),
                    "reranking_model": config.get("reranking_model"),
                    "score_threshold": config.get("score_threshold")
                },
                "all_metrics": config.get("metrics", {})
            }
            comparison["configurations"].append(config_info)
        
        # Add summary statistics
        scores = [config.get("metrics", {}).get(metric, 0) for config in configs]
        comparison["summary"] = {
            "best_score": max(scores),
            "worst_score": min(scores),
            "score_range": max(scores) - min(scores),
            "mean_score": sum(scores) / len(scores),
            "score_improvement": max(scores) - min(scores)
        }
        
        return comparison
    
    def save_analysis_report(self, results_df: pd.DataFrame, 
                           report_name: str = None) -> str:
        """
        Generate and save a comprehensive analysis report
        
        Args:
            results_df: Results DataFrame
            report_name: Optional custom report name
            
        Returns:
            Path to saved report
        """
        Config.ensure_directories()
        
        if report_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"analysis_report_{timestamp}.json"
        
        report_path = Config.SCORING_RESULTS_DIR / report_name
        
        # Generate comprehensive report
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_results": len(results_df),
                "analysis_framework": "RAG Parameter Tuning Analysis"
            },
            "performance_summary": self.generate_performance_summary(results_df),
            "best_parameters": self.get_best_parameters(results_df),
            "top_5_configurations": self.get_top_n_parameters(results_df, n=5),
            "parameter_impact_analysis": self.analyze_parameter_impact(results_df)
        }
        
        # Convert numpy types to native Python types for JSON serialization
        json_safe_report = self._convert_numpy_types(report)
        
        with open(report_path, 'w') as f:
            json.dump(json_safe_report, f, indent=2)
        
        logger.info(f"Analysis report saved to {report_path}")
        return str(report_path)
    
    def list_available_results(self) -> List[Dict[str, Any]]:
        """
        List all available scoring results files and MLflow experiments
        
        Returns:
            List of results file information
        """
        results_files = []
        
        # Check scoring results directory for CSV files
        csv_files = list(Config.SCORING_RESULTS_DIR.glob("*_scoring_results_*.csv"))
        
        for file_path in csv_files:
            try:
                # Try to find corresponding summary file
                summary_file = file_path.with_suffix('').with_suffix('').parent / f"{file_path.stem.replace('_results_', '_summary_')}.json"
                
                if summary_file.exists():
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                    
                    results_files.append({
                        "source_type": "csv",
                        "results_file": str(file_path),
                        "summary_file": str(summary_file),
                        "created_at": summary.get("summary", {}).get("created_at"),
                        "total_experiments": summary.get("summary", {}).get("total_experiments"),
                        "collections_tested": summary.get("summary", {}).get("collections_tested"),
                        "best_score": summary.get("best_result", {}).get("composite_score"),
                        "file_size": file_path.stat().st_size
                    })
                else:
                    # No summary file, basic info only
                    results_files.append({
                        "source_type": "csv",
                        "results_file": str(file_path),
                        "summary_file": None,
                        "created_at": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                        "total_experiments": None,
                        "collections_tested": None,
                        "best_score": None,
                        "file_size": file_path.stat().st_size
                    })
            except Exception as e:
                logger.warning(f"Could not read results file {file_path}: {e}")
        
        # Check for MLflow experiments
        try:
            import mlflow
            
            # Set MLflow tracking URI
            mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
            
            # Get all experiments
            experiments = mlflow.search_experiments()
            
            for exp in experiments:
                try:
                    # Get runs for this experiment
                    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
                    
                    if not runs.empty:
                        # Calculate best score if available
                        best_score = None
                        if 'metrics.composite_score' in runs.columns:
                            best_score = runs['metrics.composite_score'].max()
                        
                        # Get creation time from experiment
                        creation_time = None
                        if hasattr(exp, 'creation_time') and exp.creation_time:
                            creation_time = datetime.fromtimestamp(exp.creation_time / 1000).isoformat()
                        else:
                            # Use the latest run's start time
                            if 'start_time' in runs.columns:
                                latest_start = runs['start_time'].max()
                                if pd.notna(latest_start):
                                    creation_time = latest_start.isoformat()
                        
                        results_files.append({
                            "source_type": "mlflow",
                            "experiment_name": exp.name,
                            "experiment_id": exp.experiment_id,
                            "created_at": creation_time or datetime.now().isoformat(),
                            "total_experiments": len(runs),
                            "collections_tested": 1,  # Assuming single collection per experiment
                            "best_score": best_score,
                            "run_count": len(runs)
                        })
                        
                except Exception as e:
                    logger.warning(f"Could not read MLflow experiment {exp.name}: {e}")
                    
        except ImportError:
            logger.info("MLflow not available - skipping MLflow experiment discovery")
        except Exception as e:
            logger.warning(f"Error accessing MLflow experiments: {e}")
        
        # Sort by creation time, newest first
        results_files.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return results_files
