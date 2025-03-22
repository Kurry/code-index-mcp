"""
Embedding Manager for Code Index MCP

This module manages code embeddings using the dual-embedding document processor
with text and code models for enhanced semantic search capabilities.
"""
import os
import sys
import json
import hashlib
import time
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Generator
from contextlib import contextmanager
from dotenv import load_dotenv
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Import Qdrant client
try:
    from qdrant_client import QdrantClient, models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

# Import embedder components
from ..embedder import process_directory, generate_embeddings, search_documents
from ..embedder.schema import ProcessingStats, EmbeddingStats, Document, SearchResult
from ..embedder.text_processor import textify_code, extract_code_structure
from ..embedder.embedding_models import get_embedding_models, DualEmbeddingModel
from ..embedder.storage import EmbeddingStorage
from ..embedder.search import search_embeddings, group_by_field

# Collection configuration
COLLECTION_NAME = "code-search"
DEFAULT_TEXT_MODEL = "text-embedding-3-small"
DEFAULT_CODE_MODEL = "text-embedding-3-small"
BATCH_SIZE = 64  # Batch size for embedding generation

class EmbeddingManager:
    """
    Manages code embeddings and vector search for the project.
    Uses dual models for both natural language and code search.
    """
    
    def __init__(self, base_path, supported_extensions=None, use_qdrant_cloud=False):
        """
        Initialize the embedding manager.
        
        Args:
            base_path: Path to the project directory
            supported_extensions: List of file extensions to index
            use_qdrant_cloud: Whether to use Qdrant Cloud or local instance
        """
        self.base_path = Path(base_path)
        self.supported_extensions = supported_extensions or []
        
        # Load environment variables from .env
        load_dotenv()
        
        # Set up Qdrant client
        self.qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = os.environ.get("QDRANT_API_KEY")
        
        if QDRANT_AVAILABLE:
            if use_qdrant_cloud and self.qdrant_api_key:
                print(f"Using Qdrant Cloud at {self.qdrant_url}", file=sys.stderr)
                self.client = QdrantClient(
                    url=self.qdrant_url,
                    api_key=self.qdrant_api_key
                )
                self.storage_type = "qdrant"
            else:
                # Use in-memory Qdrant instance for testing
                print("Using in-memory Qdrant instance", file=sys.stderr)
                self.client = QdrantClient(":memory:")
                self.storage_type = "qdrant"
        else:
            print("Qdrant not available, using pickle storage", file=sys.stderr)
            self.client = None
            self.storage_type = "pickle"
        
        # Initialize embedding storage
        self.storage = EmbeddingStorage(base_dir=str(self.base_path / ".code_indexer"))
        
        # Create .code_indexer directory if it doesn't exist
        self.index_dir = self.base_path / ".code_indexer"
        self.index_dir.mkdir(exist_ok=True)
        
        # Set up models
        self.text_model = DEFAULT_TEXT_MODEL
        self.code_model = DEFAULT_CODE_MODEL
        
        # Track latest embedding file
        self.latest_embedding_file = None
    
    def index_directory(self) -> int:
        """
        Index all files in the directory.
        Returns the number of files indexed.
        """
        start_time = time.time()
        
        print(f"Indexing directory {self.base_path}", file=sys.stderr)
        
        # Call the function tool correctly
        process_args = {
            "directory": str(self.base_path),
            "exclude_dirs": None,  # Use defaults
            "file_extensions": self.supported_extensions if self.supported_extensions else None,
            "max_files": 1000,  # Allow indexing more files
            "chunk_size": 500    # Granular chunking
        }
        
        # Call the process_directory function
        result = process_directory(**process_args)
        
        # Generate output path
        timestamp = int(time.time())
        output_file = str(self.index_dir / f"embeddings_{timestamp}.pkl")
        
        # Generate embeddings
        embedding_args = {
            "output_file": output_file,
            "text_model": self.text_model,
            "code_model": self.code_model,
            "batch_size": BATCH_SIZE,
            "storage_type": self.storage_type
        }
        
        # Call the generate_embeddings function
        embedding_result = generate_embeddings(**embedding_args)
        
        # Update latest embedding file
        self.latest_embedding_file = output_file
        
        elapsed_time = time.time() - start_time
        print(f"Indexing completed in {elapsed_time:.2f} seconds", file=sys.stderr)
        
        return result.get("files_processed", 0)
    
    def semantic_search(self, query: str, limit: int = 5, use_model: str = "both") -> List[Dict[str, Any]]:
        """
        Search for code semantically using embeddings.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            use_model: Which model to use - "text", "code", or "both"
            
        Returns:
            List of search results
        """
        if not self.latest_embedding_file and not self.storage_type == "qdrant":
            # Try to find the latest embedding file
            latest_file = self.storage.get_latest_embeddings(str(self.index_dir))
            if latest_file:
                self.latest_embedding_file = latest_file
            else:
                raise ValueError("No embeddings found. Please index the directory first.")
        
        # Determine embedding settings based on use_model
        use_text = use_model in ["text", "both"]
        use_code = use_model in ["code", "both"]
        
        # Set up the arguments for search
        search_args = {
            "query": query,
            "embeddings_source": self.latest_embedding_file if self.storage_type == "pickle" else COLLECTION_NAME,
            "top_k": limit,
            "use_text_embeddings": use_text,
            "use_code_embeddings": use_code,
            "storage_type": self.storage_type
        }
        
        # Call the search_documents function
        results = search_documents(**search_args)
        
        # Convert results to the expected format
        formatted_results = []
        for result in results:
            formatted_result = {
                "name": result.get("name", ""),
                "score": result.get("combined_score", 0.0),
                "file_path": result.get("metadata", {}).get("file_path", ""),
                "file_name": result.get("metadata", {}).get("file_name", ""),
                "snippet": result.get("snippet", ""),
                "code_type": result.get("metadata", {}).get("document_type", "file"),
                "line_from": result.get("metadata", {}).get("start_line", 0),
                "line_to": result.get("metadata", {}).get("end_line", 0),
                "metadata": result.get("metadata", {})
            }
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    def get_file_content(self, rel_path: str) -> str:
        """Get the content of a file."""
        full_path = self.base_path / rel_path
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        if self.storage_type == "qdrant" and self.client:
            try:
                collection_info = self.client.get_collection(COLLECTION_NAME)
                collection_stats = self.client.collection_info(COLLECTION_NAME)
                
                return {
                    "vector_count": collection_info.vectors_count,
                    "indexed_percent": collection_stats.indexed_percent,
                    "status": collection_info.status,
                    "storage_type": "qdrant"
                }
            except Exception as e:
                return {
                    "error": str(e),
                    "storage_type": "qdrant"
                }
        elif self.latest_embedding_file:
            try:
                # Load embedding data to get stats
                data = self.storage.load_embeddings(self.latest_embedding_file)
                return {
                    "vector_count": len(data.documents),
                    "text_model": data.text_model,
                    "code_model": data.code_model,
                    "storage_type": "pickle",
                    "file": self.latest_embedding_file
                }
            except Exception as e:
                return {
                    "error": str(e),
                    "storage_type": "pickle"
                }
        else:
            return {
                "status": "no_embeddings",
                "storage_type": self.storage_type
            }

class FileChangeHandler(FileSystemEventHandler):
    """Handle file changes for auto-indexing."""
    
    def __init__(self, embedding_manager):
        self.embedding_manager = embedding_manager
        self.last_modified_times = {}
        self.debounce_time = 2.0  # seconds
    
    def on_modified(self, event):
        """Process modified file event."""
        if event.is_directory:
            return
            
        # Debounce to avoid multiple indexing for rapid file changes
        path = event.src_path
        current_time = time.time()
        
        # Skip if this file was recently processed
        if path in self.last_modified_times:
            if current_time - self.last_modified_times[path] < self.debounce_time:
                return
        
        # Update the modification time
        self.last_modified_times[path] = current_time
        
        # Check if this is a file we should index
        rel_path = os.path.relpath(path, str(self.embedding_manager.base_path))
        _, ext = os.path.splitext(path)
        
        # Skip hidden files and directories
        if any(part.startswith('.') for part in Path(rel_path).parts):
            return
            
        # Skip if extension is not in supported extensions (if any are defined)
        if (self.embedding_manager.supported_extensions and 
            ext.lstrip('.') not in self.embedding_manager.supported_extensions):
            return
            
        print(f"File modified: {rel_path}, triggering re-indexing...", file=sys.stderr)
        # Re-index the entire directory
        # Could be optimized to just update this file's embeddings
        self.embedding_manager.index_directory()
    
    def on_deleted(self, event):
        """Process deleted file event."""
        if event.is_directory:
            return
            
        # Trigger re-indexing when a file is deleted
        rel_path = os.path.relpath(event.src_path, str(self.embedding_manager.base_path))
        _, ext = os.path.splitext(event.src_path)
        
        # Skip if extension is not in supported extensions (if any are defined)
        if (self.embedding_manager.supported_extensions and 
            ext.lstrip('.') not in self.embedding_manager.supported_extensions):
            return
            
        print(f"File deleted: {rel_path}, triggering re-indexing...", file=sys.stderr)
        # Re-index the entire directory
        self.embedding_manager.index_directory()
