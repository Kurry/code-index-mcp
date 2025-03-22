#!/usr/bin/env python3
import os
import pickle
import time
import glob
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging

# Try to import Qdrant client
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qdrant_models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

from .schema import Document, SearchMetadata, SearchResult, EmbeddingData

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingStorage:
    """
    Handles storage and retrieval of embeddings.
    """
    
    def __init__(self, base_dir: str = "embeddings"):
        """
        Initialize the storage.
        
        Args:
            base_dir: Base directory for storing embeddings
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    def save_embeddings(self, embedding_data: EmbeddingData, filepath: str) -> str:
        """
        Save embeddings to a file.
        
        Args:
            embedding_data: Embedding data to save
            filepath: Path to save the embeddings
            
        Returns:
            Path to the saved file
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            # Save embeddings
            with open(filepath, 'wb') as f:
                pickle.dump(embedding_data, f)
            
            logger.info(f"Embeddings saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
            raise
    
    def save_metadata(self, metadata: Dict[str, SearchMetadata], filepath: str) -> str:
        """
        Save only metadata to a file.
        
        Args:
            metadata: Metadata to save
            filepath: Path to save the metadata
            
        Returns:
            Path to the saved file
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            # Save metadata
            with open(filepath, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Metadata saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            raise
    
    def load_embeddings(self, filepath: str) -> EmbeddingData:
        """
        Load embeddings from a file.
        
        Args:
            filepath: Path to the embeddings file
            
        Returns:
            Loaded embedding data
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Embeddings file not found: {filepath}")
        
        try:
            with open(filepath, 'rb') as f:
                embedding_data = pickle.load(f)
            
            # Validate that it's an EmbeddingData object
            if not isinstance(embedding_data, EmbeddingData):
                raise ValueError(f"File does not contain valid embedding data: {filepath}")
            
            return embedding_data
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            raise
    
    def load_metadata(self, filepath: str) -> Dict[str, SearchMetadata]:
        """
        Load metadata from a file.
        
        Args:
            filepath: Path to the metadata file
            
        Returns:
            Loaded metadata
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Metadata file not found: {filepath}")
        
        try:
            with open(filepath, 'rb') as f:
                metadata = pickle.load(f)
            
            # Validate the metadata format
            if not isinstance(metadata, dict):
                raise ValueError(f"File does not contain valid metadata: {filepath}")
            
            return metadata
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            raise
    
    def get_latest_embeddings(self, base_dir: Optional[str] = None) -> Optional[str]:
        """
        Get the path to the latest embeddings file.
        
        Args:
            base_dir: Base directory to search for embeddings
            
        Returns:
            Path to the latest embeddings file, or None if no files found
        """
        if base_dir is None:
            base_dir = self.base_dir
        
        # Look for embedding files
        embedding_files = glob.glob(os.path.join(base_dir, "embeddings_*.pkl"))
        
        if not embedding_files:
            return None
        
        # Return the most recent file
        return max(embedding_files, key=os.path.getmtime)
    
    def store_in_qdrant(
        self,
        embedding_data: EmbeddingData,
        collection_name: str = "code_embeddings",
        url: str = "http://localhost:6333",
        batch_size: int = 100,
    ) -> bool:
        """
        Store embeddings in Qdrant.
        
        Args:
            embedding_data: Embedding data to store
            collection_name: Qdrant collection name
            url: Qdrant server URL
            batch_size: Batch size for uploading points
            
        Returns:
            True if successful, False otherwise
        """
        if not QDRANT_AVAILABLE:
            logger.error("Qdrant client not available. Install with 'pip install qdrant-client'")
            return False
        
        try:
            # Connect to Qdrant
            client = QdrantClient(url=url)
            
            # Get vector dimensions
            text_dim = len(embedding_data.text_embeddings[0]) if embedding_data.text_embeddings else 0
            code_dim = len(embedding_data.code_embeddings[0]) if embedding_data.code_embeddings else 0
            
            # Check if collection exists
            collections = client.get_collections().collections
            collection_exists = any(c.name == collection_name for c in collections)
            
            # Create collection if it doesn't exist
            if not collection_exists:
                vectors_config = {}
                
                if text_dim > 0:
                    vectors_config["text"] = qdrant_models.VectorParams(
                        size=text_dim,
                        distance=qdrant_models.Distance.COSINE,
                    )
                
                if code_dim > 0:
                    vectors_config["code"] = qdrant_models.VectorParams(
                        size=code_dim,
                        distance=qdrant_models.Distance.COSINE,
                    )
                
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=vectors_config,
                )
                logger.info(f"Created Qdrant collection: {collection_name}")
            
            # Upload points in batches
            documents = embedding_data.documents
            text_embeddings = embedding_data.text_embeddings
            code_embeddings = embedding_data.code_embeddings
            
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_text = text_embeddings[i:i+batch_size] if text_embeddings else None
                batch_code = code_embeddings[i:i+batch_size] if code_embeddings else None
                
                points = []
                
                for j, doc in enumerate(batch_docs):
                    vectors = {}
                    
                    if batch_text is not None:
                        vectors["text"] = batch_text[j].tolist()
                    
                    if batch_code is not None:
                        vectors["code"] = batch_code[j].tolist()
                    
                    # Convert metadata to payload
                    metadata_dict = doc.metadata.dict()
                    
                    points.append(qdrant_models.PointStruct(
                        id=doc.id,
                        vectors=vectors,
                        payload={
                            "content": doc.content,
                            **metadata_dict
                        }
                    ))
                
                client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                
                logger.info(f"Uploaded batch {i//batch_size + 1} ({len(points)} points) to Qdrant")
            
            return True
        except Exception as e:
            logger.error(f"Error storing in Qdrant: {e}")
            return False
    
    def search_qdrant(
        self,
        collection_name: str,
        text_query_vector: Optional[List[float]] = None,
        code_query_vector: Optional[List[float]] = None,
        top_k: int = 10,
        group_by: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search for similar documents in Qdrant.
        
        Args:
            collection_name: Qdrant collection name
            text_query_vector: Text query vector
            code_query_vector: Code query vector
            top_k: Number of top results to return
            group_by: Field to group results by
            
        Returns:
            List of search results
        """
        if not QDRANT_AVAILABLE:
            logger.error("Qdrant client not available. Install with 'pip install qdrant-client'")
            return []
        
        try:
            # Connect to Qdrant
            client = QdrantClient(url="http://localhost:6333")
            
            # Create search vectors
            search_queries = []
            
            if text_query_vector is not None:
                search_queries.append(
                    qdrant_models.SearchRequest(
                        vector_name="text",
                        vector=text_query_vector,
                        limit=top_k * 2,  # Get more results than needed for RRF
                    )
                )
            
            if code_query_vector is not None:
                search_queries.append(
                    qdrant_models.SearchRequest(
                        vector_name="code",
                        vector=code_query_vector,
                        limit=top_k * 2,  # Get more results than needed for RRF
                    )
                )
            
            if not search_queries:
                logger.error("No query vectors provided")
                return []
            
            # Execute search
            results = client.search_batch(
                collection_name=collection_name,
                requests=search_queries,
            )
            
            # Process results
            text_results = results[0] if text_query_vector is not None else []
            code_results = results[1] if (code_query_vector is not None and text_query_vector is not None) else \
                          results[0] if code_query_vector is not None else []
            
            # Merge results
            all_results = {}
            
            # Add text results
            for res in text_results:
                doc_id = res.id
                all_results[doc_id] = {
                    "id": doc_id,
                    "text_score": res.score,
                    "code_score": 0.0,
                    "content": res.payload.get("content", ""),
                    "metadata": res.payload
                }
            
            # Add or update with code results
            for res in code_results:
                doc_id = res.id
                if doc_id in all_results:
                    all_results[doc_id]["code_score"] = res.score
                else:
                    all_results[doc_id] = {
                        "id": doc_id,
                        "text_score": 0.0,
                        "code_score": res.score,
                        "content": res.payload.get("content", ""),
                        "metadata": res.payload
                    }
            
            # Calculate combined scores
            for doc_id, res in all_results.items():
                text_score = res["text_score"]
                code_score = res["code_score"]
                
                # Simple weighted average for combined score
                if text_query_vector is not None and code_query_vector is not None:
                    combined_score = (text_score + code_score) / 2.0
                elif text_query_vector is not None:
                    combined_score = text_score
                else:
                    combined_score = code_score
                
                res["combined_score"] = combined_score
            
            # Convert to SearchResult objects
            search_results = []
            for doc_id, res in all_results.items():
                metadata_dict = {k: v for k, v in res["metadata"].items() if k != "content"}
                
                # Create metadata object
                try:
                    metadata = SearchMetadata(**metadata_dict)
                except Exception as e:
                    logger.warning(f"Error creating metadata for document {doc_id}: {e}")
                    metadata = SearchMetadata(
                        file_path="unknown",
                        file_name="unknown",
                        file_type="unknown",
                        document_type="unknown"
                    )
                
                search_results.append(SearchResult(
                    id=doc_id,
                    content=res["content"],
                    metadata=metadata,
                    text_score=res["text_score"],
                    code_score=res["code_score"],
                    combined_score=res["combined_score"]
                ))
            
            # Sort by combined score
            search_results.sort(key=lambda x: x.combined_score, reverse=True)
            
            # Group results if requested
            if group_by is not None and hasattr(SearchMetadata, group_by):
                from .search import group_by_field
                search_results = group_by_field(search_results, group_by, top_k)
            
            # Limit to top_k
            return search_results[:top_k]
        
        except Exception as e:
            logger.error(f"Error searching in Qdrant: {e}")
            return [] 