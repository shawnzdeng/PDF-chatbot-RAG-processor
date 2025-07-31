"""
PDF Document Processor for Qdrant Cloud
Parses PDF files and uploads embeddings to Qdrant vector database
"""

import os
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from config import Config

logger = logging.getLogger(__name__)

class QdrantProcessor:
    """Handles PDF processing and Qdrant operations"""
    
    def __init__(self, 
                 chunk_size: int = None,
                 chunk_overlap: int = None,
                 embedding_model: str = None,
                 collection_name: str = None):
        """
        Initialize the processor
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            embedding_model: OpenAI embedding model name
            collection_name: Qdrant collection name
        """
        Config.validate_env_vars()
        
        self.chunk_size = chunk_size or Config.DEFAULT_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.DEFAULT_CHUNK_OVERLAP
        self.embedding_model = embedding_model or Config.DEFAULT_EMBEDDING_MODEL
        self.collection_name = collection_name or Config.DEFAULT_COLLECTION_NAME
        
        # Initialize clients
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model,
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        self.qdrant_client = QdrantClient(
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_API_KEY
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        logger.info(f"Initialized QdrantProcessor with chunk_size={self.chunk_size}, "
                   f"overlap={self.chunk_overlap}, model={self.embedding_model}")
    
    def load_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Load and parse PDF document
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of document chunks with metadata
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Loading PDF: {pdf_path}")
        
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Prepare documents with metadata
        processed_docs = []
        for i, chunk in enumerate(chunks):
            doc_data = {
                "id": str(uuid.uuid4()),
                "content": chunk.page_content,
                "metadata": {
                    "source": pdf_path,
                    "page": chunk.metadata.get("page", 0),
                    "chunk_index": i,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "embedding_model": self.embedding_model
                }
            }
            processed_docs.append(doc_data)
        
        logger.info(f"Processed {len(processed_docs)} chunks from PDF")
        return processed_docs
    
    def create_collection(self, vector_size: int = 1536) -> bool:
        """
        Create Qdrant collection if it doesn't exist
        
        Args:
            vector_size: Dimension of embedding vectors
            
        Returns:
            True if collection created or exists
        """
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name in collection_names:
                logger.info(f"Collection '{self.collection_name}' already exists")
                return True
            
            # Create collection
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            
            logger.info(f"Created collection '{self.collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            return False
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for text chunks
        
        Args:
            texts: List of text chunks
            
        Returns:
            List of embedding vectors
        """
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        try:
            embeddings = self.embeddings.embed_documents(texts)
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def upload_to_qdrant(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Upload documents with embeddings to Qdrant
        
        Args:
            documents: List of processed documents
            
        Returns:
            True if upload successful
        """
        if not documents:
            logger.warning("No documents to upload")
            return False
        
        try:
            # Extract texts for embedding
            texts = [doc["content"] for doc in documents]
            
            # Generate embeddings
            embeddings = self.generate_embeddings(texts)
            
            # Get vector size and create collection
            vector_size = len(embeddings[0]) if embeddings else 1536
            self.create_collection(vector_size)
            
            # Prepare points for upload
            points = []
            for doc, embedding in zip(documents, embeddings):
                point = PointStruct(
                    id=doc["id"],
                    vector=embedding,
                    payload={
                        "content": doc["content"],
                        **doc["metadata"]
                    }
                )
                points.append(point)
            
            # Upload to Qdrant in batches
            batch_size = 100
            total_batches = (len(points) + batch_size - 1) // batch_size
            
            for i in tqdm(range(0, len(points), batch_size), 
                         desc="Uploading to Qdrant", total=total_batches):
                batch = points[i:i + batch_size]
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
            
            logger.info(f"Successfully uploaded {len(points)} points to Qdrant")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading to Qdrant: {e}")
            return False
    
    def process_pdf_to_qdrant(self, pdf_path: str) -> bool:
        """
        Complete pipeline: load PDF, process, and upload to Qdrant
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            True if successful
        """
        try:
            # Load and process PDF
            documents = self.load_pdf(pdf_path)
            
            # Upload to Qdrant
            success = self.upload_to_qdrant(documents)
            
            if success:
                logger.info(f"Successfully processed and uploaded PDF: {pdf_path}")
            else:
                logger.error(f"Failed to upload PDF: {pdf_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the Qdrant collection"""
        try:
            info = self.qdrant_client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}


def main():
    """Main function for testing"""
    processor = QdrantProcessor()
    
    # Process the PDF in data folder
    pdf_path = Config.DATA_DIR / "random_machine_learing_pdf.pdf"
    
    if pdf_path.exists():
        success = processor.process_pdf_to_qdrant(str(pdf_path))
        if success:
            print(f"✅ Successfully processed {pdf_path}")
            print(f"Collection info: {processor.get_collection_info()}")
        else:
            print(f"❌ Failed to process {pdf_path}")
    else:
        print(f"❌ PDF file not found: {pdf_path}")


if __name__ == "__main__":
    main()
