# PDF RAG Parameter Optimization System

A specialized system for finding optimal RAG (Retrieval-Augmented Generation) parameters using Qdrant Cloud storage, MLflow experiment tracking, and RAGAs evaluation. This system processes PDF documents, tests various parameter combinations, and exports optimized configurations for production RAG systems.

## 🎯 Primary Purpose

This project focuses on **parameter optimization** for RAG systems using **premium AI models**. It:
- Processes PDFs and stores embeddings in Qdrant Cloud using **text-embedding-3-large** (highest quality)
- Tests focused parameter combinations with **GPT-4o** (latest and most capable model)
- Uses low temperature settings (0.0-0.1) for **deterministic, accurate responses**
- Employs RAGAs framework for comprehensive evaluation
- Exports optimized parameters and collection info for production use
- Provides MLflow tracking for experiment management

The configuration focuses on **quality over quantity** - testing 18 carefully selected parameter combinations using the best available models rather than testing many combinations with varied model quality.

The optimized parameters and Qdrant collection name are designed to be used by **separate production RAG systems** with user interfaces.

## 🚀 Features

- **PDF Processing**: Parse PDF documents and convert to optimized embeddings
- **Qdrant Cloud Integration**: Store document embeddings with configurable parameters
- **Systematic Parameter Testing**: Test multiple combinations of RAG parameters
- **MLflow Experiment Tracking**: Track and compare optimization experiments
- **RAGAs Evaluation**: Comprehensive evaluation using faithfulness, relevancy, precision, and recall metrics
- **Production Export**: Export optimized parameters and collection info for production RAG systems
- **CLI Interface**: Command-line tools for processing, testing, and optimization

## 📁 Project Structure

```
PDF-chatbot-RAG-processor/
├── config.py                 # Configuration and environment settings
├── main.py                   # CLI entry point for optimization pipeline
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (API keys)
├── data/                     # PDF documents to process
│   └── random_machine_learing_pdf.pdf
├── processor/                # PDF processing and Qdrant upload
│   ├── __init__.py
│   └── qdrant_processor.py
├── simple_rag/              # RAG implementation for testing
│   ├── __init__.py
│   └── qdrant_rag.py
├── parameter_tuning/        # MLflow optimization system
│   ├── __init__.py
│   ├── parameters_config.json      # Parameter combinations to test
│   ├── mlflow_tuner.py             # Optimization engine
│   ├── Unique_RAG_QA_Benchmark.csv # Evaluation dataset
│   ├── best_parameters.json        # Generated: best parameters
│   └── production_rag_config.json  # Generated: production-ready config
└── mlruns/                  # MLflow tracking (generated)
```

## ⚙️ Setup

### 1. Environment Variables

Create a `.env` file with your API keys:

```env
OPENAI_API_KEY=your_openai_api_key
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_URL=your_qdrant_cloud_url
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Setup

```bash
python main.py process --help
```

## 🔧 Usage

### Primary Workflow: Parameter Optimization

#### 1. Process PDF Documents
```bash
# Process default PDF from data folder
python main.py process

# Process specific PDF with custom parameters
python main.py process --pdf-path "path/to/your.pdf" --chunk-size 1000 --chunk-overlap 200
```

#### 2. Run Parameter Optimization
```bash
# Run comprehensive parameter tuning (optimized for quality)
python main.py tune --max-combinations 18 --show-details

# Quick optimization for testing
python main.py tune --max-combinations 9
```

#### 3. Export Results for Production
After optimization, you'll get:
- `parameter_tuning/best_parameters.json` - Raw optimization results
- `parameter_tuning/production_rag_config.json` - Production-ready configuration
- Qdrant collection name and URL for integration

### Testing and Validation

#### Query System for Testing
```bash
# Test a single question
python main.py query --question "What is machine learning?"

# Interactive testing mode
python main.py query --interactive

# Test with specific parameters
python main.py query --question "What are the types of learning?" --llm-model "gpt-4o" --top-k 5
```

### Direct Module Usage

#### Process PDF for Optimization

```python
from processor import QdrantProcessor

processor = QdrantProcessor(
    chunk_size=1000,
    chunk_overlap=100,
    embedding_model="text-embedding-3-large"
)

success = processor.process_pdf_to_qdrant("data/your_document.pdf")
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

rag = QdrantRAG(
    llm_model="gpt-4o",
    temperature=0.0,
    top_k=5
)

result = rag.answer_question("What is machine learning?")
print(result["answer"])
```

## 📊 Parameter Optimization

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

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔬 Advanced Configuration

### Custom Prompt Template

Modify the prompt in `simple_rag/qdrant_rag.py`:

```python
def _create_prompt_template(self):
    template = """Your custom prompt here...
    Context: {context}
    Question: {question}
    Answer:"""
    return ChatPromptTemplate.from_template(template)
```

### Custom Evaluation Metrics

Add custom metrics in `parameter_tuning/mlflow_tuner.py`:

```python
def custom_metric(prediction, reference):
    # Your custom evaluation logic
    return score
```


## 🎯 Summary

This system is designed specifically for **RAG parameter optimization** and **production integration preparation**. It:

1. **Processes PDFs** and stores embeddings in Qdrant Cloud
2. **Tests parameter combinations** systematically using MLflow
3. **Evaluates performance** with RAGAs metrics
4. **Exports optimized configurations** for production RAG systems
5. **Provides collection information** for seamless integration

### Key Outputs
- **Optimized Parameters**: Best chunk size, overlap, models, temperature, top-k
- **Performance Metrics**: RAGAs evaluation scores for each configuration
- **Production Config**: Ready-to-use configuration file
- **Qdrant Collection**: Pre-processed embeddings ready for production queries

### Integration Workflow
1. Run this optimization system to find best parameters
2. Export `production_rag_config.json` 
3. Use the configuration and collection name in your production RAG system
4. Deploy with confidence knowing parameters are optimized

This approach separates **parameter optimization** (this project) from **production deployment** (your UI-enabled RAG system), allowing for focused optimization and clean production code.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain**: For RAG framework
- **Qdrant**: For vector database
- **OpenAI**: For embeddings and LLM
- **RAGAs**: For evaluation framework
- **MLflow**: For experiment tracking