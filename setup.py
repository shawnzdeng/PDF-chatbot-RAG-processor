"""
Quick setup and demo script for PDF RAG system
"""

import os
import sys
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from config import Config
from processor import QdrantProcessor
from simple_rag import QdrantRAG

def setup_demo():
    """Run a complete demo of the system"""
    
    print("🚀 PDF RAG System Demo")
    print("=" * 50)
    
    # 1. Validate environment
    print("\n1. Validating environment...")
    try:
        Config.validate_env_vars()
        print("✅ Environment variables validated")
    except ValueError as e:
        print(f"❌ Environment validation failed: {e}")
        print("\nPlease check your .env file contains:")
        print("- OPENAI_API_KEY")
        print("- QDRANT_API_KEY") 
        print("- QDRANT_URL")
        return False
    
    # 2. Process PDF
    print("\n2. Processing PDF document...")
    pdf_path = Config.DATA_DIR / "random_machine_learing_pdf.pdf"
    
    if not pdf_path.exists():
        print(f"❌ PDF file not found: {pdf_path}")
        return False
    
    processor = QdrantProcessor()
    success = processor.process_pdf_to_qdrant(str(pdf_path))
    
    if success:
        print("✅ PDF processed and uploaded to Qdrant")
        collection_info = processor.get_collection_info()
        print(f"   Documents in collection: {collection_info.get('points_count', 'N/A')}")
    else:
        print("❌ Failed to process PDF")
        return False
    
    # 3. Test RAG system
    print("\n3. Testing RAG system...")
    rag = QdrantRAG()
    
    test_questions = [
        "What is machine learning?",
        "What are the types of learning?",
        "What is PAC learning?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n   Question {i}: {question}")
        
        try:
            result = rag.answer_question(question)
            print(f"   Answer: {result['answer'][:100]}...")
            print(f"   Retrieved docs: {result['retrieved_documents']}")
            print(f"   Relevance score: {result['average_relevance_score']:.3f}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n✅ Demo completed successfully!")
    print(f"\nNext steps:")
    print(f"1. Run parameter tuning: python main.py tune")
    print(f"2. Try interactive mode: python main.py query --interactive")
    print(f"3. View MLflow results: mlflow ui")
    print(f"4. Export best parameters for production RAG system")
    
    return True

def quick_test():
    """Quick functionality test"""
    print("🧪 Quick Test")
    print("=" * 30)
    
    try:
        # Test configuration
        Config.validate_env_vars()
        print("✅ Configuration OK")
        
        # Test Qdrant connection
        from qdrant_client import QdrantClient
        client = QdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY)
        collections = client.get_collections()
        print(f"✅ Qdrant connection OK ({len(collections.collections)} collections)")
        
        # Test OpenAI connection
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)
        test_embedding = embeddings.embed_query("test")
        print(f"✅ OpenAI connection OK (embedding size: {len(test_embedding)})")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup and demo for PDF RAG system")
    parser.add_argument("--demo", action="store_true", help="Run full demo")
    parser.add_argument("--test", action="store_true", help="Run quick test")
    
    args = parser.parse_args()
    
    if args.demo:
        setup_demo()
    elif args.test:
        quick_test()
    else:
        print("Usage: python setup.py --demo   (full demo)")
        print("       python setup.py --test   (quick test)")
