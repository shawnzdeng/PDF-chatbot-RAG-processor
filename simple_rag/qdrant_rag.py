"""
Simple RAG implementation using Qdrant and OpenAI
Provides question-answering capabilities over PDF documents
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from config import Config

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Data class for retrieval results"""
    content: str
    score: float
    metadata: Dict[str, Any]

class QdrantRAG:
    """RAG system using Qdrant for retrieval and OpenAI for generation"""
    
    def __init__(self,
                 collection_name: str = None,
                 embedding_model: str = None,
                 llm_model: str = None,
                 temperature: float = None,
                 top_k: int = 5):
        """
        Initialize RAG system
        
        Args:
            collection_name: Qdrant collection name
            embedding_model: OpenAI embedding model
            llm_model: OpenAI LLM model
            temperature: LLM temperature
            top_k: Number of documents to retrieve
        """
        Config.validate_env_vars()
        
        self.collection_name = collection_name or Config.DEFAULT_COLLECTION_NAME
        self.embedding_model = embedding_model or Config.DEFAULT_EMBEDDING_MODEL
        self.llm_model = llm_model or Config.DEFAULT_LLM_MODEL
        self.temperature = temperature if temperature is not None else Config.DEFAULT_TEMPERATURE
        self.top_k = top_k
        
        # Initialize clients
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model,
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        self.llm = ChatOpenAI(
            model=self.llm_model,
            temperature=self.temperature,
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        self.qdrant_client = QdrantClient(
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_API_KEY
        )
        
        # Setup prompt template
        self.prompt_template = self._create_prompt_template()
        
        logger.info(f"Initialized QdrantRAG with model={self.llm_model}, "
                   f"temp={self.temperature}, top_k={self.top_k}")
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the prompt template for RAG"""
        template = """You are a helpful assistant that answers questions based on the provided context from PDF documents.

Context from documents:
{context}

Question: {question}

Instructions:
1. Answer the question based ONLY on the information provided in the context
2. If the answer is not found in the context, clearly state "I don't have enough information in the provided documents to answer this question"
3. Be concise and accurate
4. If relevant, mention which part of the document the information comes from

Answer:"""
        
        return ChatPromptTemplate.from_template(template)
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for query
        
        Args:
            query: User question
            
        Returns:
            Query embedding vector
        """
        try:
            embedding = self.embeddings.embed_query(query)
            return embedding
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise
    
    def retrieve_documents(self, query: str, top_k: int = None) -> List[RetrievalResult]:
        """
        Retrieve relevant documents from Qdrant
        
        Args:
            query: User question
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents with scores
        """
        top_k = top_k or self.top_k
        
        try:
            # Generate query embedding
            query_embedding = self.embed_query(query)
            
            # Search in Qdrant
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=0.7  # Minimum similarity threshold
            )
            
            # Convert to RetrievalResult objects
            results = []
            for result in search_result:
                retrieval_result = RetrievalResult(
                    content=result.payload.get("content", ""),
                    score=result.score,
                    metadata={
                        "source": result.payload.get("source", ""),
                        "page": result.payload.get("page", 0),
                        "chunk_index": result.payload.get("chunk_index", 0)
                    }
                )
                results.append(retrieval_result)
            
            logger.info(f"Retrieved {len(results)} documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def format_context(self, results: List[RetrievalResult]) -> str:
        """
        Format retrieved documents into context string
        
        Args:
            results: Retrieved documents
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant documents found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            source_info = f"Document {i} (Score: {result.score:.3f})"
            if result.metadata.get("source"):
                source_info += f" from {result.metadata['source']}"
            if result.metadata.get("page"):
                source_info += f", Page {result.metadata['page']}"
            
            context_parts.append(f"{source_info}:\n{result.content}\n")
        
        return "\n".join(context_parts)
    
    def generate_answer(self, query: str, context: str) -> str:
        """
        Generate answer using LLM
        
        Args:
            query: User question
            context: Retrieved context
            
        Returns:
            Generated answer
        """
        try:
            # Create the chain
            chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | self.prompt_template
                | self.llm
                | StrOutputParser()
            )
            
            # Generate answer
            answer = chain.invoke({"context": context, "question": query})
            return answer.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    def answer_question(self, question: str, top_k: int = None) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve and generate answer
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary with answer, context, and metadata
        """
        try:
            # Retrieve relevant documents
            retrieved_docs = self.retrieve_documents(question, top_k)
            
            # Format context
            context = self.format_context(retrieved_docs)
            
            # Generate answer
            answer = self.generate_answer(question, context)
            
            # Calculate average relevance score
            avg_score = sum(doc.score for doc in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0
            
            result = {
                "question": question,
                "answer": answer,
                "context": context,
                "retrieved_documents": len(retrieved_docs),
                "average_relevance_score": avg_score,
                "sources": [doc.metadata.get("source", "") for doc in retrieved_docs],
                "model_params": {
                    "llm_model": self.llm_model,
                    "embedding_model": self.embedding_model,
                    "temperature": self.temperature,
                    "top_k": top_k or self.top_k
                }
            }
            
            logger.info(f"Generated answer for question with {len(retrieved_docs)} retrieved docs")
            return result
            
        except Exception as e:
            logger.error(f"Error in answer_question: {e}")
            return {
                "question": question,
                "answer": f"Error processing question: {str(e)}",
                "context": "",
                "retrieved_documents": 0,
                "average_relevance_score": 0,
                "sources": [],
                "model_params": {}
            }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the Qdrant collection"""
        try:
            info = self.qdrant_client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "total_points": info.points_count,
                "vectors_count": info.vectors_count,
                "indexed_vectors": info.indexed_vectors_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}


def main():
    """Main function for testing"""
    rag = QdrantRAG()
    
    # Test questions
    test_questions = [
        "What is machine learning?",
        "What are the types of learning?",
        "What is PAC learning?",
        "Describe the goal of reinforcement learning."
    ]
    
    print(f"Collection stats: {rag.get_collection_stats()}")
    print("\n" + "="*50)
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        print("-" * 50)
        
        result = rag.answer_question(question)
        print(f"Answer: {result['answer']}")
        print(f"Retrieved docs: {result['retrieved_documents']}")
        print(f"Avg relevance: {result['average_relevance_score']:.3f}")


if __name__ == "__main__":
    main()
