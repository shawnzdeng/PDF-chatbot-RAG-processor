# PDF RAG Parameter Optimization System

A specialized system for finding optimal RAG (Retrieval-Augmented Generation) parameters using Qdrant Cloud storage, MLflow experiment tracking, and RAGAs evaluation. This system processes PDF documents, tests various parameter combinations, and exports optimized configurations for production RAG systems.

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
Place your PDF file in the `data/` folder (or use the provided example):
```bash
# Example PDF is already included at data/random_machine_learning_pdf.pdf
```

## 🏃‍♂️ How to Run

### Option 1: Complete Pipeline (Recommended for Beginners)
```bash
# Run complete optimization pipeline with default settings
python main.py run-pipeline --max-combinations 9 --show-details

# Run with your own PDF
python main.py run-pipeline --pdf-path "your_document.pdf" --collection-prefix "my_project" --max-combinations 9 --show-details

# Quick test run (fewer combinations)
python main.py run-pipeline --max-combinations 3 --show-details
```

### Option 2: Step-by-Step Approach (More Control)
```bash
# Step 1: Prepare collections for all combinations
python main.py prepare-collections --show-details

# Step 2: Score all combinations  
python main.py score --max-combinations 9 --show-details

# Step 3: Analyze results and generate production config
## 📄 License

MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain**: RAG framework
- **Qdrant**: Vector database  
- **OpenAI**: Embeddings and LLM
- **RAGAs**: Evaluation framework
- **MLflow**: Experiment tracking
```

### View Results
```bash
# Launch MLflow UI to view experiment tracking
mlflow ui
# Then open: http://localhost:5000

# Check your optimized configuration
ls results/production_config/
```

## 🎯 What This System Does

This project optimizes RAG parameters using premium AI models:
- **PDF Processing**: Uses text-embedding-3-large for high-quality embeddings
- **Parameter Testing**: Tests combinations of chunk sizes, overlaps, models, temperatures, and top-k values
- **Evaluation**: Uses RAGAs framework for comprehensive metrics (faithfulness, relevancy, precision, recall)
- **Export**: Generates production-ready configuration files
- **Tracking**: MLflow experiment management for all optimization runs

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

## ⚙️ Configuration

### Custom Parameters
Edit `parameters_config.json` to test different combinations:
```json
{
  "chunk_sizes": [500, 1000, 1500],
  "chunk_overlaps": [50, 100, 150],
  "embedding_models": ["text-embedding-3-large"],
  "llm_models": ["gpt-4o"],
  "temperatures": [0.0, 0.1],
  "top_k_values": [3, 5, 10]
}
```

### Custom Evaluation Questions
Replace `data/Unique_RAG_QA_Benchmark.csv` with your own questions:
```csv
question,ground_truth
"What is machine learning?","Machine learning is a method..."
"How do neural networks work?","Neural networks are computing..."
```

## 🏗️ Production Integration

Load and use your optimized configuration:
```python
import json
from qdrant_client import QdrantClient

# Load optimized configuration
with open('results/production_config/production_rag_config_*.json', 'r') as f:
    config = json.load(f)

# Connect to optimized Qdrant collection
client = QdrantClient(
    url=config['qdrant_config']['url'],
    api_key="your_qdrant_api_key"
)

# Use the optimized parameters in your production RAG system
rag_params = config['rag_system_config']
collection_name = config['qdrant_config']['collection_name']
```

## 📁 Project Structure

```
PDF-chatbot-RAG-processor/
├── main.py                   # 🎯 Main CLI entry point
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (create this)
├── parameters_config.json    # Parameter combinations to test
├── data/                     # Place your PDF documents here
│   ├── random_machine_learning_pdf.pdf
│   └── Unique_RAG_QA_Benchmark.csv
├── results/                  # Generated optimization results
│   ├── preparation/          # Collection metadata
│   ├── scoring/             # Scoring results
│   └── production_config/   # Final optimized configuration
└── mlruns/                  # MLflow tracking (auto-generated)
```

## 🔧 Available Commands

```bash
# Main workflows
python main.py run-pipeline         # Complete end-to-end optimization
python main.py prepare-collections  # Phase 1: Create collections
python main.py score                # Phase 2: Score combinations
python main.py analyze              # Phase 3: Generate production config

# Testing
python main.py query                # Test optimized system
python main.py query --interactive  # Interactive testing

# Help
python main.py --help
python main.py run-pipeline --help
```

#### 1. Production Configuration
`results/production_config/production_rag_config_YYYYMMDD_HHMMSS.json`
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
    "collection_name": "rag_random_ml_pdf__chunk500_overlap100_emb3large",
    "url": "https://your-qdrant-cloud-url:6333",
    "vector_size": 3072
  },
  "performance_metrics": {
    "faithfulness": 0.737,
    "answer_relevancy": 0.704,
    "context_precision": 0.890,
    "context_recall": 0.579,
    "composite_score": 0.728
  }
}
```

#### 2. Analysis Report
`results/scoring/analysis_report_YYYYMMDD_HHMMSS.json`
- Detailed performance analysis
- Best parameter combinations
- Metric comparisons

#### 3. Collection Metadata
`results/preparation/rag_*_collections_metadata_YYYYMMDD_HHMMSS.json`
- Information about created Qdrant collections
- Reusable for future scoring runs

### Performance Metrics Explained

The system uses the **RAGAs evaluation framework** with a benchmark dataset:

- **Benchmark File**: `data/Unique_RAG_QA_Benchmark.csv` contains evaluation questions and ground truth answers
- **Faithfulness (0-1)**: How grounded the answer is in the retrieved context
- **Answer Relevancy (0-1)**: How relevant the answer is to the question  
- **Context Precision (0-1)**: Precision of retrieved context chunks
- **Context Recall (0-1)**: Recall of retrieved context chunks
- **Composite Score**: Average of all metrics (optimization target)

The benchmark ensures consistent, objective evaluation across all parameter combinations.

## 🔧 Advanced Commands

```bash
# Custom PDF processing
python main.py prepare-collections --pdf-path "my_document.pdf" --collection-prefix "my_project"

# Score with specific metadata
python main.py score --metadata-file "path/to/metadata.json" --max-combinations 18

# Analyze specific results
python main.py analyze --results-file "path/to/results.csv" --show-details
```

#### Run Parameter Optimization

```python
from parameter_tuning import ParameterTuner

tuner = ParameterTuner()
results_df = tuner.run_parameter_tuning(max_combinations=18)
best_params = tuner.get_best_parameters(results_df)

# Export for production use
production_config_path = tuner.export_production_config(best_params)
```

#### Test RAG Configuration

```python
from simple_rag import QdrantRAG

# For testing with defaults
rag = QdrantRAG.create_with_defaults(
    llm_model="gpt-4o",
    temperature=0.0,
    top_k=5,
    score_threshold=0.3
)

# Or for production with all parameters specified
rag = QdrantRAG(
    collection_name="my_collection",
    embedding_model="text-embedding-3-large",
    llm_model="gpt-4o",
    temperature=0.0,
    top_k=5,
    prompt_template="Your custom prompt template with {context} and {question}",
    score_threshold=0.3  # Adjust based on your similarity requirements
)

result = rag.answer_question("What is machine learning?")
print(result["answer"])
```

## � Troubleshooting

This system's core functionality is finding optimal parameter combinations:

### Tunable Parameters

- **Chunk Size**: 500, 1000, 1500 tokens
- **Chunk Overlap**: 50, 100, 150 tokens
- **Embedding Models**: text-embedding-3-large (high-quality embeddings)
- **LLM Models**: gpt-4o (latest and most capable model)
- **Temperature**: 0.0, 0.1 (focused on deterministic, accurate responses)
- **Top K Retrieval**: 3, 5, 10 documents

### RAGAs Evaluation Metrics

- **Faithfulness (0-1)**: How grounded the answer is in the context
- **Answer Relevancy (0-1)**: How relevant the answer is to the question  
- **Context Precision (0-1)**: Precision of retrieved context
- **Context Recall (0-1)**: Recall of retrieved context
- **Composite Score**: Average of all metrics (optimization target)

### Output Files

After optimization, the system generates:

1. **`best_parameters.json`** - Complete optimization results
2. **`production_rag_config.json`** - Production-ready configuration with:
   - Optimized RAG parameters
   - Qdrant collection information
   - Performance metrics
   - Integration instructions

### MLflow Experiment Tracking

```bash
# View optimization experiments
mlflow ui --backend-store-uri file:./mlruns
```

Access at: `http://localhost:5000`

## � Production Integration

### Using Optimized Parameters

The `production_rag_config.json` file contains everything needed for production deployment:

```json
{
  "rag_system_config": {
    "chunk_size": 1000,
    "chunk_overlap": 100,
    "embedding_model": "text-embedding-3-large",
    "llm_model": "gpt-4o",
    "temperature": 0.0,
    "top_k_retrieval": 5
  },
  "qdrant_config": {
    "collection_name": "pdf_documents_optimized",
    "url": "your-qdrant-cloud-url",
    "vector_size": 3072
  },
  "performance_metrics": {
    "composite_score": 0.847,
    "faithfulness": 0.893,
    "answer_relevancy": 0.876
  },
  "optimization_info": {
    "tuning_date": "2025-01-30T...",
    "total_combinations_tested": 18,
    "ready_for_production": true
  }
}
```

### Integration Steps for Production RAG System

1. **Load optimized configuration**:
   ```python
   import json
   with open('production_rag_config.json', 'r') as f:
       config = json.load(f)
   ```

2. **Connect to optimized Qdrant collection**:
   ```python
   qdrant_client = QdrantClient(
       url=config['qdrant_config']['url'],
       api_key=your_qdrant_api_key
   )
   collection_name = config['qdrant_config']['collection_name']
   ```

3. **Use optimized RAG parameters**:
   ```python
   rag_params = config['rag_system_config']
   # Apply these exact parameters for optimal performance
   ```

### Collection Information

- Collection contains pre-processed embeddings from PDF documents
- Embeddings are optimized for the best-performing parameter combination
- Ready for immediate use in production RAG systems
- No need to reprocess documents - use existing collection

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