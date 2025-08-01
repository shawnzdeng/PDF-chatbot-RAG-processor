"""
PDF Document Processor for Qdrant Cloud
Parses PDF files and uploads embeddings to Qdrant vector database
"""

import os
import uuid
import time
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
        
        # Initialize OpenAI embeddings with proper error handling
        try:
            self.embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                api_key=Config.OPENAI_API_KEY
            )
        except Exception as e:
            logger.warning(f"Failed with api_key parameter: {e}")
            # Try alternative initialization method
            try:
                self.embeddings = OpenAIEmbeddings(
                    openai_api_key=Config.OPENAI_API_KEY,
                    model=self.embedding_model
                )
            except Exception as e2:
                logger.error(f"Both initialization methods failed: {e2}")
                # Use basic initialization without explicit parameters
                import os
                os.environ['OPENAI_API_KEY'] = Config.OPENAI_API_KEY
                self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
        
        self.qdrant_client = QdrantClient(
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_API_KEY,
            timeout=60  # Increase timeout to 60 seconds
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
    
    def _upload_batch_with_retry(self, batch: List[PointStruct], max_retries: int = 3) -> bool:
        """
        Upload a batch of points with retry logic
        
        Args:
            batch: List of PointStruct objects to upload
            max_retries: Maximum number of retry attempts
            
        Returns:
            True if successful, False otherwise
        """
        for attempt in range(max_retries):
            try:
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                return True
            except Exception as e:
                wait_time = (2 ** attempt) * 1  # Exponential backoff: 1, 2, 4 seconds
                logger.warning(f"Upload attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {max_retries} upload attempts failed")
                    return False
        return False

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
            self.create_collection(vector_size=vector_size)
            
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
            
            # Upload to Qdrant in smaller batches with retry logic
            batch_size = 10  # Reduced batch size for better reliability
            total_batches = (len(points) + batch_size - 1) // batch_size
            failed_batches = 0
            
            for i in tqdm(range(0, len(points), batch_size), 
                         desc="Uploading to Qdrant", total=total_batches):
                batch = points[i:i + batch_size]
                success = self._upload_batch_with_retry(batch)
                if not success:
                    failed_batches += 1
                    logger.error(f"Failed to upload batch {i//batch_size + 1}")
            
            if failed_batches > 0:
                logger.warning(f"Upload completed with {failed_batches} failed batches out of {total_batches}")
                return failed_batches == 0  # Return False if any batches failed
            
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
    # Get first PDF file in data folder
    pdf_files = list(Config.DATA_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ No PDF files found in {Config.DATA_DIR}")
        print("Please add a PDF file to the data folder and try again")
        return
    
    pdf_path = pdf_files[0]
    print(f"📄 Processing PDF: {pdf_path.name}")
    
    processor = QdrantProcessor(
        chunk_size=1000,
        chunk_overlap=100,
        embedding_model="text-embedding-3-large",
        collection_name="pdf_documents_test"
    )
    
    success = processor.process_pdf_to_qdrant(str(pdf_path))
    
    if success:
        print(f"✅ Successfully processed {pdf_path.name}")
        collection_info = processor.get_collection_info()
        print(f"📊 Collection info: {collection_info}")
    else:
        print(f"❌ Failed to process {pdf_path.name}")


if __name__ == "__main__":
    main()
