# PDF RAG Parameter Optimization System

A specialized system for finding optimal RAG (Retrieval-Augmented Generation) parameters using Qdrant Cloud storage, MLflow experiment tracking, and RAGAs evaluation. This system processes PDF documents, tests various parameter combinations, and exports optimized configurations for production RAG systems.

See this repo for an example to use this project's output to build a RAG service: https://github.com/shawnzdeng/PDF-chatbot-RAG-service

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.8+
- OpenAI API key
- Qdrant Cloud account and API key

### 2. Installation
```bash
git clone <repository-url>
cd PDF-chatbot-RAG-processor
pip install -r requirements.txt
```

### 3. Environment Setup
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_URL=your_qdrant_cloud_url_here
```

### 4. Add Your PDF Document
Place your PDF file in the `data/` folder (or use the provided example).

### 5. Update the `parameters_config.json` file
Update the `input` section based on the file you uploaded.
```bash
"input": {
    "pdf_file_name": "random_machine_learning_pdf.pdf",
    "validate_file_name": "Unique_RAG_QA_Benchmark.csv",
    "collection_prefix": "rag_random_ml_pdf_",
    "file_description": "The file is a set of lecture notes titled Machine Learning from Malla Reddy College of Engineering & Technology, designed for a final-year B.Tech course. It provides a comprehensive overview of machine learning fundamentals, including supervised, unsupervised, and reinforcement learning; common models like decision trees, neural networks, and support vector machines; ensemble methods; probabilistic models; and genetic algorithms. The material emphasizes both theoretical foundations (e.g., PAC learning, VC dimension) and practical applications.",
    "example_questions": [
      "What is the difference between supervised and unsupervised learning?",
      "Can you explain how decision trees work?",
      "What are the key components of a neural network?",
      "How does reinforcement learning differ from other types of machine learning?",
      "What is the purpose of ensemble methods in machine learning?"
    ]
  }
```
You may also update the parameter tunning and other section as needed.

## 🏃‍♂️ How to Run

### Option 1: Step-by-Step Approach (Recommended, More Control)
```bash
# Step 1: Prepare collections for all combinations
python main.py prepare-collections --show-details

# Step 2: Score all combinations  
python main.py score --max-combinations 9 --show-details

# Step 3: Analyze results and generate production config
python main.py analyze --max-combinations 9 --show-details
```

### Option 2: Complete Pipeline
```bash
# Run complete optimization pipeline with default settings
python main.py run-pipeline --max-combinations 9 --show-details
```

### View Results
```bash
# Launch MLflow UI to view experiment tracking
mlflow ui
# Then open: http://localhost:5000

# Check your optimized configuration
ls results/production_config/
```

## 📊 Key Output Files

After optimization, you'll get:

1. **`production_rag_config.json`** - Ready-to-use configuration:
```json
{
  "rag_system_config": {
    "chunk_size": 500,
    "chunk_overlap": 100,
    "embedding_model": "text-embedding-3-large",
    "llm_model": "gpt-4o",
    "temperature": 0.0,
    "top_k_retrieval": 5
  },
  "qdrant_config": {
    "collection_name": "optimized_collection_name",
    "url": "your-qdrant-cloud-url",
    "vector_size": 3072
  },
  "performance_metrics": {
    "composite_score": 0.847,
    "faithfulness": 0.893,
    "answer_relevancy": 0.876
  }
}
```

2. **Analysis Report** - Detailed performance analysis
3. **Collection Metadata** - Qdrant collection information for production use

## 📁 Project Structure

```
PDF-chatbot-RAG-processor/
├── main.py                   # 🎯 Main CLI entry point
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── parameters_config.json    # Parameter combinations to test
├── LICENSE                   # MIT License
├── .env                      # Environment variables (create this)
├── .gitignore               # Git ignore rules
├── data/                     # Place your PDF documents here
│   ├── random_machine_learning_pdf.pdf
│   └── Unique_RAG_QA_Benchmark.csv
├── simple_rag/              # Core RAG implementation
│   ├── __init__.py
│   ├── qdrant_rag.py        # Main RAG system
│   └── reranker.py          # Result reranking logic
├── processor/               # Document processing modules
│   ├── __init__.py
│   └── qdrant_processor.py  # Qdrant vector database processor
├── parameter_tuning/        # Parameter optimization modules
│   ├── __init__.py
│   ├── collection_preparer.py  # Collection preparation logic
│   ├── config_loader.py        # Configuration loading utilities
│   ├── ragas_scorer.py         # RAGAs evaluation scorer
│   └── results_analyzer.py     # Results analysis and reporting
├── results/                 # Generated optimization results
│   ├── preparation/         # Collection metadata
│   ├── scoring/            # Scoring results
│   └── production_config/  # Final optimized configuration
├── mlruns/                 # MLflow tracking (auto-generated)
└── __pycache__/           # Python bytecode cache (auto-generated)
```

### Performance Metrics Explained

The system uses the **RAGAs evaluation framework** with a benchmark dataset:

- **Faithfulness (0-1)**: How grounded the answer is in the context
- **Answer Relevancy (0-1)**: How relevant the answer is to the question  
- **Context Precision (0-1)**: Precision of retrieved context
- **Context Recall (0-1)**: Recall of retrieved context
- **Composite Score**: Average of all metrics (optimization target)


## 🔍 Troubleshooting

### Common Issues

1. **Qdrant Connection Error**
   - Verify QDRANT_URL and QDRANT_API_KEY in `.env`
   - Check network connectivity

2. **OpenAI API Error**
   - Verify OPENAI_API_KEY in `.env`
   - Check API quota and billing

3. **PDF Processing Fails**
   - Ensure PDF is readable and not corrupted
   - Check file permissions

4. **MLflow UI Not Loading**
```bash
mlflow ui --host 0.0.0.0 --port 5000
```

### Debug Mode
Enable detailed logging for troubleshooting:
```bash
# Set environment variable for detailed logging
python main.py tune --max-combinations 3 --show-details
```

```python
# Or enable in Python code
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain**: For RAG framework
- **Qdrant**: For vector database
- **OpenAI**: For embeddings and LLM
- **RAGAs**: For evaluation framework
- **MLflow**: For experiment tracking