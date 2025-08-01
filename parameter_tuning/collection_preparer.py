"""
Collection preparation system for RAG parameter tuning
Handles creating and organizing collections with different chunking parameters
"""

import os
import json
import logging
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Local imports
from config import Config
from processor import QdrantProcessor

logger = logging.getLogger(__name__)

class CollectionPreparer:
    """Handles preparation of collections for different chunking parameters"""
    
    def __init__(self, config_file: str = None):
        """
        Initialize collection preparer
        
        Args:
            config_file: Path to parameters configuration JSON
        """
        self.config_file = config_file or str(Config.BASE_DIR / "parameters_config.json")
        
        # Load configuration
        self.config = self._load_config()
        
        logger.info(f"Initialized CollectionPreparer with config from {self.config_file}")
    
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
    
    def generate_chunking_combinations(self) -> List[Dict[str, Any]]:
        """
        Generate combinations for chunk_size and chunk_overlap only
        
        Returns:
            List of chunking parameter dictionaries
        """
        chunk_sizes = Config.get_chunking_parameters(self.config)["chunk_sizes"]
        chunk_overlaps = Config.get_chunking_parameters(self.config)["chunk_overlaps"]
        embedding_models = Config.get_model_parameters(self.config)["embedding_models"]
        
        # Create combinations of chunk_size, chunk_overlap, and embedding_model
        chunking_combinations = []
        for chunk_size in chunk_sizes:
            for chunk_overlap in chunk_overlaps:
                for embedding_model in embedding_models:
                    chunking_combinations.append({
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "embedding_model": embedding_model
                    })
        
        logger.info(f"Generated {len(chunking_combinations)} chunking combinations")
        return chunking_combinations
    
    def prepare_collections(self, pdf_path: str = None, collection_prefix: str = None) -> List[Dict[str, Any]]:
        """
        Prepare collections for all chunk_size and chunk_overlap combinations
        
        Args:
            pdf_path: Optional path to PDF file (uses config default if None)
            collection_prefix: Prefix for collection names (uses config default if None)
            
        Returns:
            List of collection metadata
        """
        # Ensure results directories exist
        Config.ensure_directories()
        
        # Use PDF path from config if not provided
        if pdf_path is None:
            pdf_path = Config.get_pdf_file_path(self.config)
        
        # Use collection prefix from config if not provided
        if collection_prefix is None:
            collection_prefix = Config.get_collection_prefix(self.config)
        
        chunking_combinations = self.generate_chunking_combinations()
        collections_metadata = []
        
        print(f"🔄 Processing {len(chunking_combinations)} chunking combinations...")
        print(f"📄 PDF: {pdf_path}")
        print(f"🏷️  Collection prefix: {collection_prefix}")
        
        for i, combo in enumerate(tqdm(chunking_combinations, desc="Creating collections")):
            try:
                # Create unique collection name
                collection_name = f"{collection_prefix}_chunk{combo['chunk_size']}_overlap{combo['chunk_overlap']}_emb{combo['embedding_model'].replace('text-embedding-', '').replace('-', '')}"
                
                print(f"  📦 Creating collection {i+1}/{len(chunking_combinations)}: {collection_name}")
                
                # Process PDF with current chunking parameters
                processor = QdrantProcessor(
                    chunk_size=combo["chunk_size"],
                    chunk_overlap=combo["chunk_overlap"],
                    embedding_model=combo["embedding_model"],
                    collection_name=collection_name
                )
                
                success = processor.process_pdf_to_qdrant(pdf_path)
                
                if success:
                    # Store collection metadata
                    metadata = {
                        "collection_name": collection_name,
                        "chunk_size": combo["chunk_size"],
                        "chunk_overlap": combo["chunk_overlap"],
                        "embedding_model": combo["embedding_model"],
                        "pdf_path": pdf_path,
                        "collection_prefix": collection_prefix,
                        "created_at": datetime.now().isoformat()
                    }
                    collections_metadata.append(metadata)
                    print(f"    ✅ Successfully created collection: {collection_name}")
                else:
                    print(f"    ❌ Failed to create collection: {collection_name}")
                    
            except Exception as e:
                logger.error(f"Error creating collection for combo {combo}: {e}")
                print(f"    ❌ Error creating collection: {e}")
        
        # Save collections metadata to results/preparation directory with timestamped filename
        metadata_file_path = self.save_collections_metadata(collections_metadata, collection_prefix, pdf_path)
        
        print(f"📁 Collections metadata saved to: {metadata_file_path}")
        print(f"🔢 Total collections created: {len(collections_metadata)}")
        
        return collections_metadata
    
    def save_collections_metadata(self, collections_metadata: List[Dict[str, Any]], 
                                 collection_prefix: str = None, pdf_path: str = None) -> str:
        """
        Save collections metadata to timestamped file
        
        Args:
            collections_metadata: List of collection metadata
            collection_prefix: Collection prefix used (uses config default if None)
            pdf_path: Path to source PDF file (uses config default if None)
            
        Returns:
            Path to saved metadata file
        """
        # Use config defaults if not provided
        if collection_prefix is None:
            collection_prefix = Config.get_collection_prefix(self.config)
        if pdf_path is None:
            pdf_path = Config.get_pdf_file_path(self.config)
        
        pdf_basename = Path(pdf_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata_filename = f"{collection_prefix}_{pdf_basename}_collections_metadata_{timestamp}.json"
        metadata_file_path = Config.PREPARATION_RESULTS_DIR / metadata_filename
        
        # Add summary information to metadata
        metadata_with_summary = {
            "summary": {
                "total_collections": len(collections_metadata),
                "pdf_source": pdf_path,
                "collection_prefix": collection_prefix,
                "created_at": datetime.now().isoformat(),
                "chunking_combinations_count": len(collections_metadata)
            },
            "collections": collections_metadata
        }
        
        with open(metadata_file_path, 'w') as f:
            json.dump(metadata_with_summary, f, indent=2)
        
        logger.info(f"Collections metadata saved to {metadata_file_path}")
        return str(metadata_file_path)
    
    def load_collections_metadata(self, metadata_file_path: str = None) -> List[Dict[str, Any]]:
        """
        Load collections metadata from file
        
        Args:
            metadata_file_path: Optional specific path to metadata file.
                               If not provided, will try to find the most recent file
                               
        Returns:
            List of collection metadata
        """
        if metadata_file_path and Path(metadata_file_path).exists():
            # Use the specific file provided
            metadata_file = Path(metadata_file_path)
        else:
            # Try to find the most recent metadata file in preparation results
            preparation_files = list(Config.PREPARATION_RESULTS_DIR.glob("*_collections_metadata_*.json"))
            if preparation_files:
                # Sort by modification time, newest first
                metadata_file = max(preparation_files, key=lambda p: p.stat().st_mtime)
                logger.info(f"Using most recent collections metadata: {metadata_file}")
            else:
                # Fallback to legacy location
                legacy_file = Config.PARAMETER_TUNING_DIR / "collections_metadata.json"
                if legacy_file.exists():
                    metadata_file = legacy_file
                    logger.warning(f"Using legacy metadata file: {metadata_file}")
                else:
                    raise FileNotFoundError(
                        f"No collections metadata found. "
                        f"Please run 'prepare-collections' first or specify a valid metadata file path."
                    )
        
        with open(metadata_file, 'r') as f:
            metadata_content = json.load(f)
        
        # Handle both old and new format
        if isinstance(metadata_content, dict) and "collections" in metadata_content:
            # New format with summary
            collections_metadata = metadata_content["collections"]
            logger.info(f"Loaded metadata for {len(collections_metadata)} collections from {metadata_file}")
            logger.info(f"Summary: {metadata_content.get('summary', {})}")
        else:
            # Legacy format - direct list
            collections_metadata = metadata_content
            logger.info(f"Loaded metadata for {len(collections_metadata)} collections from {metadata_file}")
        
        return collections_metadata
    
    def list_available_metadata_files(self) -> List[Dict[str, Any]]:
        """
        List all available collections metadata files
        
        Returns:
            List of metadata file information
        """
        metadata_files = []
        
        # Check preparation results directory
        preparation_files = list(Config.PREPARATION_RESULTS_DIR.glob("*_collections_metadata_*.json"))
        for file_path in preparation_files:
            try:
                with open(file_path, 'r') as f:
                    content = json.load(f)
                
                if isinstance(content, dict) and "summary" in content:
                    summary = content["summary"]
                    metadata_files.append({
                        "file_path": str(file_path),
                        "created_at": summary.get("created_at"),
                        "collection_prefix": summary.get("collection_prefix"),
                        "pdf_source": summary.get("pdf_source"),
                        "total_collections": summary.get("total_collections"),
                        "file_size": file_path.stat().st_size
                    })
                else:
                    # Legacy format
                    metadata_files.append({
                        "file_path": str(file_path),
                        "created_at": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                        "collection_prefix": "unknown",
                        "pdf_source": "unknown",
                        "total_collections": len(content) if isinstance(content, list) else 0,
                        "file_size": file_path.stat().st_size
                    })
            except Exception as e:
                logger.warning(f"Could not read metadata file {file_path}: {e}")
        
        # Sort by creation time, newest first
        metadata_files.sort(key=lambda x: x["created_at"], reverse=True)
        
        return metadata_files
    
    def get_collection_stats(self, metadata_file_path: str = None) -> Dict[str, Any]:
        """
        Get statistics about prepared collections
        
        Args:
            metadata_file_path: Optional path to specific metadata file
            
        Returns:
            Dictionary with collection statistics
        """
        collections_metadata = self.load_collections_metadata(metadata_file_path)
        
        # Calculate statistics
        chunk_sizes = [col["chunk_size"] for col in collections_metadata]
        chunk_overlaps = [col["chunk_overlap"] for col in collections_metadata]
        embedding_models = [col["embedding_model"] for col in collections_metadata]
        
        stats = {
            "total_collections": len(collections_metadata),
            "unique_chunk_sizes": sorted(list(set(chunk_sizes))),
            "unique_chunk_overlaps": sorted(list(set(chunk_overlaps))),
            "unique_embedding_models": sorted(list(set(embedding_models))),
            "chunk_size_counts": {size: chunk_sizes.count(size) for size in set(chunk_sizes)},
            "chunk_overlap_counts": {overlap: chunk_overlaps.count(overlap) for overlap in set(chunk_overlaps)},
            "embedding_model_counts": {model: embedding_models.count(model) for model in set(embedding_models)},
            "creation_dates": [col["created_at"] for col in collections_metadata]
        }
        
        return stats
