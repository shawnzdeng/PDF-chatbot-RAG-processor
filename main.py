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
from parameter_tuning import ParameterTuner

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_pdf(args):
    """Process PDF and upload to Qdrant"""
    pdf_path = args.pdf_path or str(Config.DATA_DIR / "random_machine_learing_pdf.pdf")
    
    processor = QdrantProcessor(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
        collection_name=args.collection_name
    )
    
    success = processor.process_pdf_to_qdrant(pdf_path)
    
    if success:
        print(f"✅ Successfully processed PDF: {pdf_path}")
        print(f"Collection info: {processor.get_collection_info()}")
    else:
        print(f"❌ Failed to process PDF: {pdf_path}")
        sys.exit(1)

def query_rag(args):
    """Query the RAG system"""
    rag = QdrantRAG(
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        temperature=args.temperature,
        top_k=args.top_k
    )
    
    if args.interactive:
        print("🤖 Interactive RAG Mode (type 'quit' to exit)")
        print("=" * 50)
        
        while True:
            question = input("\nQuestion: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            try:
                result = rag.answer_question(question)
                print(f"\nAnswer: {result['answer']}")
                print(f"Retrieved docs: {result['retrieved_documents']}")
                print(f"Avg relevance: {result['average_relevance_score']:.3f}")
            except Exception as e:
                print(f"Error: {e}")
    
    else:
        if not args.question:
            print("❌ Please provide a question with --question or use --interactive mode")
            sys.exit(1)
        
        result = rag.answer_question(args.question)
        print(f"Question: {args.question}")
        print(f"Answer: {result['answer']}")
        print(f"Retrieved docs: {result['retrieved_documents']}")
        print(f"Avg relevance: {result['average_relevance_score']:.3f}")

def run_tuning(args):
    """Run parameter tuning"""
    tuner = ParameterTuner()
    
    print(f"🔧 Starting parameter tuning with max {args.max_combinations} combinations...")
    
    results_df = tuner.run_parameter_tuning(max_combinations=args.max_combinations)
    best_params = tuner.get_best_parameters(results_df)
    tuner.save_best_parameters(best_params)
    
    # Export production config
    production_config_path = tuner.export_production_config(best_params)
    
    print(f"\n✅ Parameter tuning completed!")
    print(f"📊 Best composite score: {best_params.get('metrics', {}).get('composite_score', 0):.3f}")
    print(f"📁 Best parameters: {Config.PARAMETER_TUNING_DIR / 'best_parameters.json'}")
    print(f"🚀 Production config: {production_config_path}")
    print(f"🔗 Collection name: {best_params.get('collection_name', 'N/A')}")
    
    if args.show_details:
        print(f"\n🎯 Optimized Parameters:")
        param_keys = ['chunk_size', 'chunk_overlap', 'embedding_model', 'llm_model', 'temperature', 'top_k']
        for key in param_keys:
            if key in best_params:
                print(f"   {key.replace('_', ' ').title()}: {best_params[key]}")
        
        print(f"\n📈 Performance Metrics:")
        for metric, score in best_params.get('metrics', {}).items():
            print(f"   {metric.replace('_', ' ').title()}: {score:.3f}")

def main():
    parser = argparse.ArgumentParser(description="PDF RAG System with Qdrant")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Process PDF command
    process_parser = subparsers.add_parser("process", help="Process PDF and upload to Qdrant")
    process_parser.add_argument("--pdf-path", help="Path to PDF file")
    process_parser.add_argument("--chunk-size", type=int, default=Config.DEFAULT_CHUNK_SIZE, help="Chunk size")
    process_parser.add_argument("--chunk-overlap", type=int, default=Config.DEFAULT_CHUNK_OVERLAP, help="Chunk overlap")
    process_parser.add_argument("--embedding-model", default=Config.DEFAULT_EMBEDDING_MODEL, help="Embedding model")
    process_parser.add_argument("--collection-name", default=Config.DEFAULT_COLLECTION_NAME, help="Qdrant collection name")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("--question", help="Question to ask")
    query_parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    query_parser.add_argument("--collection-name", default=Config.DEFAULT_COLLECTION_NAME, help="Qdrant collection name")
    query_parser.add_argument("--embedding-model", default=Config.DEFAULT_EMBEDDING_MODEL, help="Embedding model")
    query_parser.add_argument("--llm-model", default=Config.DEFAULT_LLM_MODEL, help="LLM model")
    query_parser.add_argument("--temperature", type=float, default=Config.DEFAULT_TEMPERATURE, help="LLM temperature")
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of documents to retrieve")
    
    # Parameter tuning command
    tuning_parser = subparsers.add_parser("tune", help="Run parameter tuning and export optimized config")
    tuning_parser.add_argument("--max-combinations", type=int, default=18, help="Maximum parameter combinations to test (18 total possible combinations)")
    tuning_parser.add_argument("--show-details", action="store_true", help="Show detailed parameter and metric information")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        Config.validate_env_vars()
    except ValueError as e:
        print(f"❌ Environment validation failed: {e}")
        sys.exit(1)
    
    if args.command == "process":
        process_pdf(args)
    elif args.command == "query":
        query_rag(args)
    elif args.command == "tune":
        run_tuning(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
