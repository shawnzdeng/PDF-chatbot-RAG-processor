"""
Main entry point for the PDF RAG system
Provides CLI interface for all system components
"""

import argparse
import sys
import logging
from pathlib import Path

from config import Config
from processor import QdrantProcessor
from simple_rag import QdrantRAG
from parameter_tuning import CollectionPreparer, RAGAsScorer, ResultsAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_collections(args):
    """Prepare collections for all chunk_size and chunk_overlap combinations"""
    # Load configuration
    config = Config.load_parameters_config()
    
    # Determine PDF path - use from args if provided, otherwise from config
    pdf_path = None
    if args.pdf_path:
        # Construct full path from filename in data folder
        pdf_path = str(Config.DATA_DIR / args.pdf_path)
    # If pdf_path is None, CollectionPreparer will use the config default
    
    # Get collection prefix from config or command line
    collection_prefix = args.collection_prefix or None  # Let CollectionPreparer handle config default
    
    # Use the specialized CollectionPreparer
    preparer = CollectionPreparer()
    
    print(f"📦 Starting collection preparation...")
    if pdf_path:
        print(f"📄 Using PDF: {pdf_path}")
    else:
        config_pdf_path = Config.get_pdf_file_path(config)
        print(f"📄 Using PDF from config: {config_pdf_path}")
    
    if collection_prefix:
        print(f"🏷️  Collection prefix: {collection_prefix}")
    else:
        config_prefix = Config.get_collection_prefix(config)
        print(f"🏷️  Collection prefix from config: {config_prefix}")
    
    collections_metadata = preparer.prepare_collections(
        pdf_path=pdf_path, 
        collection_prefix=collection_prefix
    )
    
    # Save collections metadata (preparer handles this internally but we get the path)
    metadata_file_path = preparer.save_collections_metadata(collections_metadata, collection_prefix, pdf_path)
    
    print(f"\n✅ Collection preparation completed!")
    print(f"📁 Collections metadata saved to: {metadata_file_path}")
    print(f"🔢 Total collections created: {len(collections_metadata)}")
    
    if args.show_details:
        print(f"\n📦 Created Collections:")
        for collection in collections_metadata:
            print(f"   {collection['collection_name']}: chunk_size={collection['chunk_size']}, chunk_overlap={collection['chunk_overlap']}, embedding_model={collection['embedding_model']}")
    
    return metadata_file_path

def run_ragas_scoring(args, metadata_file_path=None):
    """
    Run RAGAs scoring on prepared collections
    
    Args:
        args: Command line arguments
        metadata_file_path: Optional path to collections metadata file from previous step
    """
    # Load configuration
    config = Config.load_parameters_config()
    
    # Get max combinations from config or command line
    max_combinations = args.max_combinations or config.get("experiment_settings", {}).get("max_combinations", 18)
    
    # Use the specialized RAGAsScorer
    scorer = RAGAsScorer()
    
    print(f"🎯 Starting RAGAs scoring phase...")
    print(f"🔧 Max combinations per collection: {max_combinations}")
    
    # Determine metadata file path (priority order):
    # 1. Provided from previous step (metadata_file_path parameter)
    # 2. Command line argument (--metadata-file)
    # 3. Auto-discover latest file (handled by RAGAsScorer)
    final_metadata_file_path = None
    if metadata_file_path:
        final_metadata_file_path = metadata_file_path
        print(f"📁 Using metadata from previous step: {metadata_file_path}")
    elif args.metadata_file:
        final_metadata_file_path = args.metadata_file
        print(f"📁 Using metadata from command line: {args.metadata_file}")
    else:
        print(f"📁 Auto-discovering latest metadata file...")
    
    results_df = scorer.run_scoring_on_collections(
        max_combinations=max_combinations,
        metadata_file_path=final_metadata_file_path
    )
    
    if results_df.empty:
        print("❌ No scoring results generated")
        return None
    
    # Save scoring results - reuse the same metadata file path that was used for scoring
    collections_metadata = scorer.collection_preparer.load_collections_metadata(final_metadata_file_path)
    results_file_path = scorer.save_scoring_results(results_df, collections_metadata)
    scoring_stats = scorer.get_scoring_stats(results_df)
    
    print(f"\n✅ RAGAs scoring completed!")
    print(f"📊 Total experiments run: {len(results_df)}")
    print(f"📁 Scoring results saved to: {results_file_path}")
    print(f"🎯 Best composite score: {scoring_stats['best_score']:.3f}")
    print(f"📈 Average composite score: {scoring_stats['average_score']:.3f}")
    
    if args.show_details:
        print(f"\n🎯 Scoring Statistics:")
        print(f"   Successful experiments: {scoring_stats['successful_experiments']}")
        print(f"   Failed experiments: {scoring_stats['failed_experiments']}")
        print(f"   Collections tested: {scoring_stats['unique_collections']}")
        print(f"   Parameter combinations per collection: {scoring_stats['combinations_per_collection']}")
    
    return results_file_path

def analyze_results(args, results_file_path=None):
    """
    Analyze scoring results and generate production configuration
    
    Args:
        args: Command line arguments  
        results_file_path: Optional path to results file from previous step
    """
    # Load configuration
    config = Config.load_parameters_config()
    
    # Use the specialized ResultsAnalyzer
    analyzer = ResultsAnalyzer()
    
    print(f"📊 Starting results analysis...")
    
    # Determine results file path (priority order):
    # 1. Provided from previous step (results_file_path parameter)
    # 2. Command line argument (--results-file)
    # 3. Command line MLflow experiment (--mlflow-experiment)
    # 4. Auto-discover latest file (CSV or MLflow)
    if results_file_path:
        results_df = analyzer.load_results_from_file(results_file_path)
        print(f"📁 Using results from previous step: {results_file_path}")
    elif args.results_file:
        results_df = analyzer.load_results_from_file(args.results_file)
        print(f"📁 Using results file from command line: {args.results_file}")
    elif hasattr(args, 'mlflow_experiment') and args.mlflow_experiment:
        results_df = analyzer.load_results_from_mlflow(args.mlflow_experiment)
        print(f"📁 Using MLflow experiment from command line: {args.mlflow_experiment}")
    else:
        # Find the latest scoring results file or MLflow experiment
        available_results = analyzer.list_available_results()
        if not available_results:
            print("❌ No scoring results found. Please run scoring first or check MLflow experiments.")
            return None
        
        latest_results = available_results[0]  # Already sorted by date, newest first
        
        # Check if it's MLflow or CSV
        if latest_results.get('source_type') == 'mlflow':
            experiment_name = latest_results['experiment_name']
            results_df = analyzer.load_results_from_mlflow(experiment_name)
            print(f"📁 Using latest MLflow experiment: {experiment_name}")
            print(f"📊 Found {latest_results.get('total_experiments', 0)} runs in experiment")
        else:
            results_df = analyzer.load_results_from_file(latest_results['results_file'])
            print(f"📁 Using latest results file: {latest_results['results_file']}")
    
    if results_df.empty:
        print("❌ No valid results found for analysis")
        return None
    
    # Get best parameters
    best_params = analyzer.get_best_parameters(results_df)
    top_5_params = analyzer.get_top_n_parameters(results_df, n=5)
    
    # Perform parameter impact analysis
    parameter_impact = analyzer.analyze_parameter_impact(results_df)
    
    # Generate performance summary
    performance_summary = analyzer.generate_performance_summary(results_df)
    
    # Save best parameters
    best_params_file = analyzer.save_best_parameters(best_params)
    
    # Export production config
    production_config_path = analyzer.export_production_config(best_params)
    
    # Generate comprehensive analysis report
    analysis_report_path = analyzer.save_analysis_report(results_df)
    
    print(f"\n✅ Results analysis completed!")
    print(f"📊 Best composite score: {best_params.get('metrics', {}).get('composite_score', 0):.3f}")
    print(f"📁 Best parameters saved to: {best_params_file}")
    print(f"🚀 Production config exported to: {production_config_path}")
    print(f"📋 Analysis report saved to: {analysis_report_path}")
    print(f"🔗 Optimal collection name: {best_params.get('collection_name', 'N/A')}")
    
    if args.show_details:
        print(f"\n🎯 Optimized Parameters:")
        # Core parameters
        core_param_keys = ['chunk_size', 'chunk_overlap', 'embedding_model', 'llm_model', 'temperature', 'top_k']
        for key in core_param_keys:
            if key in best_params:
                print(f"   {key.replace('_', ' ').title()}: {best_params[key]}")
        
        # Reranker parameters (only if they exist)
        reranker_param_keys = ['enable_reranking', 'rerank_embedding_weight', 'rerank_cross_encoder_weight', 
                              'rerank_top_k_before', 'rerank_hybrid_scoring', 'reranking_model', 'score_threshold']
        reranker_params_exist = any(key in best_params for key in reranker_param_keys)
        if reranker_params_exist:
            print(f"\n🔄 Reranker Parameters:")
            for key in reranker_param_keys:
                if key in best_params:
                    print(f"   {key.replace('_', ' ').title()}: {best_params[key]}")
        
        print(f"\n📈 Performance Metrics:")
        for metric, score in best_params.get('metrics', {}).items():
            print(f"   {metric.replace('_', ' ').title()}: {score:.3f}")
        
        print(f"\n🔍 Parameter Impact Analysis:")
        for param, impact in parameter_impact.items():
            if isinstance(impact, dict) and 'correlation' in impact:
                print(f"   {param.replace('_', ' ').title()}: {impact['correlation']:.3f} correlation")
        
        print(f"\n🏆 Top 5 Configurations by Score:")
        for i, config in enumerate(top_5_params[:5], 1):
            score = config.get('metrics', {}).get('composite_score', 0)
            display_parts = [f"chunk_size={config.get('chunk_size')}", f"llm_model={config.get('llm_model')}"]
            
            # Only show reranker status if reranker data exists
            if 'enable_reranking' in config:
                rerank_status = "✅" if config.get('enable_reranking', False) else "❌"
                display_parts.append(f"rerank={rerank_status}")
            
            display_str = ", ".join(display_parts)
            print(f"   #{i}: {score:.3f} ({display_str})")
    
    return {
        'best_params_file': best_params_file,
        'production_config_path': production_config_path,
        'analysis_report_path': analysis_report_path,
        'best_params': best_params
    }

def run_full_pipeline(args):
    """
    Run the complete pipeline: prepare collections -> score -> analyze
    
    Args:
        args: Command line arguments
    """
    print("🚀 Starting full RAG optimization pipeline...")
    print("="*60)
    
    # Step 1: Prepare collections
    print("\n📦 STEP 1: Preparing Collections")
    print("-" * 40)
    metadata_file_path = prepare_collections(args)
    if not metadata_file_path:
        print("❌ Collection preparation failed. Stopping pipeline.")
        return None
    
    # Step 2: Run scoring  
    print("\n🎯 STEP 2: Running RAGAs Scoring")
    print("-" * 40)
    results_file_path = run_ragas_scoring(args, metadata_file_path)
    if not results_file_path:
        print("❌ RAGAs scoring failed. Stopping pipeline.")
        return None
    
    # Step 3: Analyze results
    print("\n📊 STEP 3: Analyzing Results")
    print("-" * 40)
    analysis_results = analyze_results(args, results_file_path)
    if not analysis_results:
        print("❌ Results analysis failed. Stopping pipeline.")
        return None
    
    # Pipeline completion summary
    print("\n" + "="*60)
    print("✅ FULL PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"📁 Collections metadata: {metadata_file_path}")
    print(f"📊 Scoring results: {results_file_path}")
    print(f"🎯 Best parameters: {analysis_results['best_params_file']}")
    print(f"🚀 Production config: {analysis_results['production_config_path']}")
    print(f"📋 Analysis report: {analysis_results['analysis_report_path']}")
    
    best_score = analysis_results['best_params'].get('metrics', {}).get('composite_score', 0)
    print(f"🏆 Best composite score achieved: {best_score:.3f}")
    
    return analysis_results

def main():
    parser = argparse.ArgumentParser(description="PDF RAG System with Qdrant")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Load config to get defaults
    try:
        config = Config.load_parameters_config()
        default_collection_prefix = Config.get_collection_prefix(config)
        default_max_combinations = config.get("experiment_settings", {}).get("max_combinations", 18)
    except Exception:
        default_collection_prefix = "rag_test"
        default_max_combinations = 18

    # Collection preparation command
    prepare_parser = subparsers.add_parser("prepare-collections", help="Prepare collections for all chunk size and overlap combinations")
    prepare_parser.add_argument("--pdf-path", help="PDF filename in the data folder (e.g., 'document.pdf')")
    prepare_parser.add_argument("--collection-prefix", default=default_collection_prefix, help=f"Prefix for collection names (default from config: '{default_collection_prefix}')")
    prepare_parser.add_argument("--show-details", action="store_true", help="Show detailed collection information")

    # RAGAs scoring command
    scoring_parser = subparsers.add_parser("score", help="Run RAGAs scoring on prepared collections")
    scoring_parser.add_argument("--max-combinations", type=int, default=default_max_combinations, help=f"Maximum parameter combinations to test per collection (default from config: {default_max_combinations})")
    scoring_parser.add_argument("--metadata-file", help="Path to specific collections metadata file")
    scoring_parser.add_argument("--show-details", action="store_true", help="Show detailed scoring statistics")

    # Results analysis command
    analysis_parser = subparsers.add_parser("analyze", help="Analyze scoring results and generate production configuration")
    analysis_parser.add_argument("--results-file", help="Path to specific scoring results CSV file (uses latest if not specified)")
    analysis_parser.add_argument("--mlflow-experiment", help="Name of specific MLflow experiment to analyze")
    analysis_parser.add_argument("--show-details", action="store_true", help="Show detailed analysis results")

    # Full pipeline command
    pipeline_parser = subparsers.add_parser("run-pipeline", help="Run complete pipeline: prepare collections -> score -> analyze")
    pipeline_parser.add_argument("--pdf-path", help="PDF filename in the data folder (e.g., 'document.pdf')")
    pipeline_parser.add_argument("--collection-prefix", default=default_collection_prefix, help=f"Prefix for collection names (default from config: '{default_collection_prefix}')")
    pipeline_parser.add_argument("--max-combinations", type=int, default=default_max_combinations, help=f"Maximum parameter combinations to test per collection (default from config: {default_max_combinations})")
    pipeline_parser.add_argument("--show-details", action="store_true", help="Show detailed information for all steps")

    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        Config.validate_env_vars()
    except ValueError as e:
        print(f"❌ Environment validation failed: {e}")
        sys.exit(1)
    
    if args.command == "prepare-collections":
        prepare_collections(args)
    elif args.command == "score":
        run_ragas_scoring(args)
    elif args.command == "analyze":
        analyze_results(args)
    elif args.command == "run-pipeline":
        run_full_pipeline(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
