#!/usr/bin/env python3
import os
import re
import glob
import hashlib
import time
from typing import Dict, List, Any, Optional, Tuple, Union
import pickle
from pathlib import Path
import numpy as np
from openai import OpenAI
from functools import lru_cache
from tqdm import tqdm
import ast
import logging

from agents import (
    Agent, 
    function_tool, 
    ModelSettings,
)

from . import text_processor
from .schema import ProcessingStats, EmbeddingStats, Document, SearchResult, SearchMetadata, EmbeddingData
from .embedding_models import get_embedding_models, DualEmbeddingModel
from .storage import EmbeddingStorage
from .guardrails import create_query_guardrail, create_directory_guardrail
from .search import search_embeddings, reciprocal_rank_fusion, group_by_field

# Import for vector database integration
try:
    from qdrant_client import QdrantClient, models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

# File extensions to process by default
DEFAULT_FILE_EXTENSIONS = [
    "py", "md", "rst", "txt", "ipynb", "js", "ts", "java", "c", "cpp", "h", "cs", "go", "rb"
]

# Directories to exclude by default
DEFAULT_EXCLUDE_DIRS = [
    "__pycache__", 
    ".git", 
    ".github", 
    ".vscode", 
    "venv", 
    "env", 
    ".env", 
    ".venv",
    "node_modules",
    "dist",
    "build"
]

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_document_id(file_path: str, content: str) -> str:
    """Generate a unique ID for a document based on its path and content."""
    content_hash = hashlib.md5(content.encode()).hexdigest()
    return f"{file_path}:{content_hash}"

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text."""
    # Simple extraction of words
    words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_]{2,}\b', text.lower())
    # Remove duplicates while preserving order
    unique_words = []
    seen = set()
    for word in words:
        if word not in seen and word not in STOP_WORDS:
            seen.add(word)
            unique_words.append(word)
    
    return unique_words[:max_keywords]

# Common stop words to exclude from keywords
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
    'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than',
    'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during',
    'to', 'from', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
    'each', 'few', 'more', 'most', 'some', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'with'
}

def textify_code(document: Dict[str, Any]) -> str:
    """
    Convert code to a textual representation optimized for semantic search.
    Similar to the Qdrant example's textify function.
    
    Args:
        document: Document dictionary with metadata
        
    Returns:
        Text representation of the code
    """
    # Extract necessary fields
    content = document.get("content", "")
    file_path = document.get("file_path", "")
    file_name = document.get("file_name", "")
    file_type = document.get("file_type", "")
    module_name = document.get("module_name", "")
    
    # Normalize code element names
    function_name = document.get("function_name", "")
    class_name = document.get("class_name", "")
    
    # Format name elements by converting camelCase/snake_case to human readable
    human_function_name = text_processor.humanize_identifier(function_name) if function_name else ""
    human_class_name = text_processor.humanize_identifier(class_name) if class_name else ""
    human_module_name = text_processor.humanize_identifier(module_name) if module_name else ""
    
    # Determine document type
    doc_type = "Class" if class_name else "Function" if function_name else "File"
    
    # Build the context
    context = f"in module {human_module_name} " if human_module_name else ""
    context += f"file {file_name}"
    
    if human_class_name:
        context = f"defined in class {human_class_name} {context}"
    
    # Extract docstring if available
    docstring = ""
    docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
    if docstring_match:
        docstring = f"that does {docstring_match.group(1).strip()} "
    
    # Build full text representation
    text_representation = f"{doc_type} {human_function_name or human_class_name or file_name} {docstring}defined as {content[:300]} {context}"
    
    # Remove any special characters and concatenate the tokens
    tokens = re.split(r"\W", text_representation)
    tokens = filter(lambda x: x, tokens)
    return " ".join(tokens)

@function_tool
def process_directory(
    directory: str,
    exclude_dirs: Optional[List[str]] = None,
    file_extensions: Optional[List[str]] = None,
    max_files: int = 100,
    chunk_size: int = 500,  # Add chunk size parameter for granular chunking
) -> Dict[str, Any]:
    """
    Process a directory to extract code and documentation for embedding.
    
    Args:
        directory: Path to the directory to process
        exclude_dirs: List of directories to exclude
        file_extensions: List of file extensions to include
        max_files: Maximum number of files to process
        chunk_size: Maximum size of code chunks for granular extraction
        
    Returns:
        Statistics about the processed library
    """
    if not os.path.isdir(directory):
        raise ValueError(f"Directory {directory} does not exist")
    
    # Set defaults
    if exclude_dirs is None:
        exclude_dirs = DEFAULT_EXCLUDE_DIRS
    
    if file_extensions is None:
        file_extensions = DEFAULT_FILE_EXTENSIONS
    
    # Prepare stats
    stats = {
        "total_files_found": 0,
        "files_processed": 0,
        "bytes_processed": 0,
        "file_types": {},
        "functions_found": 0,
        "classes_found": 0,
        "methods_found": 0,  # Track methods separately
        "code_chunks": 0,    # Track chunks
        "examples_found": 0,
        "api_references_found": 0
    }
    
    # Get all files
    all_files = []
    for ext in file_extensions:
        pattern = os.path.join(directory, f"**/*.{ext}")
        all_files.extend(glob.glob(pattern, recursive=True))
    
    # Filter out excluded directories
    filtered_files = []
    for file_path in all_files:
        if not any(exc_dir in file_path.split(os.sep) for exc_dir in exclude_dirs):
            filtered_files.append(file_path)
    
    stats["total_files_found"] = len(filtered_files)
    
    # Limit the number of files to process
    files_to_process = filtered_files[:max_files]
    
    documents = []
    
    # Process each file
    for file_path in files_to_process:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            file_size = len(content)
            stats["bytes_processed"] += file_size
            
            # Get file extension
            _, file_ext = os.path.splitext(file_path)
            file_ext = file_ext.lstrip('.')
            
            # Update file type count
            stats["file_types"][file_ext] = stats["file_types"].get(file_ext, 0) + 1
            
            # Extract file information
            file_name = os.path.basename(file_path)
            rel_path = os.path.relpath(file_path, directory)
            
            # Create metadata
            metadata = {
                "file_path": rel_path,
                "file_name": file_name,
                "file_type": file_ext,
            }
            
            # Check if this is an example file
            if "example" in rel_path.lower() or "demo" in rel_path.lower():
                metadata["is_example"] = True
                stats["examples_found"] += 1
            
            # Check if this is API reference
            if "api" in rel_path.lower() and "reference" in rel_path.lower():
                metadata["is_api_reference"] = True
                stats["api_references_found"] += 1
            
            # Add module information
            module_name = rel_path.replace("/", ".").replace("\\", ".").replace(f".{file_ext}", "")
            metadata["module_name"] = module_name
            
            # Enhanced code structure processing for different file types
            if file_ext == "py":
                # Extract code structure
                structure = text_processor.extract_code_structure(content)
                stats["functions_found"] += len(structure["functions"])
                stats["classes_found"] += len(structure["classes"])
                
                # Process file as a whole
                file_doc = Document(
                    document_id=generate_document_id(rel_path, content),
                    content=content,
                    file_path=rel_path,
                    file_name=file_name,
                    file_type=file_ext,
                    module_name=module_name,
                    functions=structure["functions"],
                    classes=structure["classes"],
                    is_example=metadata.get("is_example", False),
                    is_api_reference=metadata.get("is_api_reference", False),
                )
                documents.append(file_doc)
                
                # Process each function individually
                for func in structure["functions"]:
                    # Extract function code
                    func_start = func['start']
                    
                    # Find function end
                    next_def = content.find("def ", func_start + 1)
                    next_class = content.find("class ", func_start + 1)
                    if next_def != -1 and (next_class == -1 or next_def < next_class):
                        func_end = next_def
                    elif next_class != -1:
                        func_end = next_class
                    else:
                        func_end = len(content)
                    
                    func_code = content[func_start:func_end]
                    
                    func_id = f"{rel_path}:function:{func['name']}"
                    
                    func_doc = Document(
                        document_id=func_id,
                        content=func_code,
                        file_path=rel_path,
                        file_name=file_name,
                        file_type=file_ext,
                        module_name=module_name,
                        function_name=func['name'],
                        is_example=metadata.get("is_example", False),
                        is_api_reference=metadata.get("is_api_reference", False),
                    )
                    documents.append(func_doc)
                
                # Process each class and its methods
                for cls in structure["classes"]:
                    cls_start = cls['start']
                    
                    # Find class end
                    next_def = content.find("def ", cls_start + 1)
                    next_class = content.find("class ", cls_start + 1)
                    if next_class != -1 and (next_def == -1 or next_class < next_def):
                        cls_end = next_class
                    elif next_def != -1:
                        cls_end = next_def
                    else:
                        cls_end = len(content)
                    
                    cls_code = content[cls_start:cls_end]
                    
                    cls_id = f"{rel_path}:class:{cls['name']}"
                    
                    cls_doc = Document(
                        document_id=cls_id,
                        content=cls_code,
                        file_path=rel_path,
                        file_name=file_name,
                        file_type=file_ext,
                        module_name=module_name,
                        class_name=cls['name'],
                        is_example=metadata.get("is_example", False),
                        is_api_reference=metadata.get("is_api_reference", False),
                    )
                    documents.append(cls_doc)
                    
                    # Extract methods from class
                    method_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)'
                    for method_match in re.finditer(method_pattern, cls_code):
                        method_name = method_match.group(1)
                        # Skip dunder methods
                        if method_name.startswith('__') and method_name.endswith('__'):
                            continue
                        
                        method_start = method_match.start()
                        
                        # Find method end within class
                        next_method = cls_code.find("def ", method_start + 1)
                        if next_method != -1:
                            method_end = next_method
                        else:
                            method_end = len(cls_code)
                        
                        method_code = cls_code[method_start:method_end]
                        
                        method_id = f"{rel_path}:class:{cls['name']}:method:{method_name}"
                        
                        method_doc = Document(
                            document_id=method_id,
                            content=method_code,
                            file_path=rel_path,
                            file_name=file_name,
                            file_type=file_ext,
                            module_name=module_name,
                            class_name=cls['name'],
                            function_name=method_name,
                            is_example=metadata.get("is_example", False),
                            is_api_reference=metadata.get("is_api_reference", False),
                        )
                        documents.append(method_doc)
                        stats["methods_found"] += 1
                
            elif file_ext in ["js", "ts", "java", "c", "cpp", "go", "cs"]:
                # Perform language-specific chunking 
                # This is a simplified approach - a real implementation would use language-specific parsers
                
                # Match functions/methods
                func_pattern = r'(function|def|func)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)'
                for match in re.finditer(func_pattern, content):
                    func_name = match.group(2)
                    func_start = match.start()
                    
                    # Find function end (simplistic approach)
                    opening_braces = 0
                    func_end = func_start
                    
                    for i in range(func_start, len(content)):
                        if content[i] == '{':
                            opening_braces += 1
                        elif content[i] == '}':
                            opening_braces -= 1
                            if opening_braces == 0:
                                func_end = i + 1
                                break
                    
                    if func_end > func_start:
                        func_code = content[func_start:func_end]
                        
                        func_id = f"{rel_path}:function:{func_name}"
                        
                        func_doc = Document(
                            document_id=func_id,
                            content=func_code,
                            file_path=rel_path,
                            file_name=file_name,
                            file_type=file_ext,
                            module_name=module_name,
                            function_name=func_name,
                            is_example=metadata.get("is_example", False),
                            is_api_reference=metadata.get("is_api_reference", False),
                        )
                        documents.append(func_doc)
                        stats["functions_found"] += 1
                
                # Match classes
                class_pattern = r'(class|interface|struct)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
                for match in re.finditer(class_pattern, content):
                    class_name = match.group(2)
                    class_start = match.start()
                    
                    # Find class end (simplistic approach)
                    opening_braces = 0
                    class_end = class_start
                    
                    for i in range(class_start, len(content)):
                        if content[i] == '{':
                            opening_braces += 1
                        elif content[i] == '}':
                            opening_braces -= 1
                            if opening_braces == 0:
                                class_end = i + 1
                                break
                    
                    if class_end > class_start:
                        class_code = content[class_start:class_end]
                        
                        class_id = f"{rel_path}:class:{class_name}"
                        
                        class_doc = Document(
                            document_id=class_id,
                            content=class_code,
                            file_path=rel_path,
                            file_name=file_name,
                            file_type=file_ext,
                            module_name=module_name,
                            class_name=class_name,
                            is_example=metadata.get("is_example", False),
                            is_api_reference=metadata.get("is_api_reference", False),
                        )
                        documents.append(class_doc)
                        stats["classes_found"] += 1
            else:
                # For other files, use a chunking approach
                if len(content) > chunk_size:
                    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
                    
                    for i, chunk in enumerate(chunks):
                        chunk_id = f"{rel_path}:chunk:{i}"
                        
                        chunk_doc = Document(
                            document_id=chunk_id,
                            content=chunk,
                            file_path=rel_path,
                            file_name=file_name,
                            file_type=file_ext,
                            module_name=module_name,
                            is_example=metadata.get("is_example", False),
                            is_api_reference=metadata.get("is_api_reference", False),
                        )
                        documents.append(chunk_doc)
                        stats["code_chunks"] += 1
                else:
                    # For small files, just use the whole file
                    document = Document(
                        document_id=generate_document_id(rel_path, content),
                        content=content,
                        file_path=rel_path,
                        file_name=file_name,
                        file_type=file_ext,
                        module_name=module_name,
                        is_example=metadata.get("is_example", False),
                        is_api_reference=metadata.get("is_api_reference", False),
                    )
                    documents.append(document)
            
            stats["files_processed"] += 1
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Store the processed documents globally
    global processed_documents
    processed_documents = documents
    
    return ProcessingStats(**stats).dict()

@function_tool
def generate_embeddings(
    output_file: str,
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    code_model: str = "jinaai/jina-embeddings-v2-base-code",
    batch_size: int = 32,
    storage_type: str = "pickle"  # Options: "pickle", "qdrant"
) -> Dict[str, Any]:
    """
    Generate embeddings for the processed documents.
    
    Args:
        output_file: Path to save the embeddings (pickle file) or collection name (Qdrant)
        text_model: Name of the text embedding model
        code_model: Name of the code embedding model
        batch_size: Batch size for embedding generation
        storage_type: Type of storage ("pickle" or "qdrant")
        
    Returns:
        Statistics about the generated embeddings
    """
    global processed_documents
    
    if not processed_documents:
        raise ValueError("No documents have been processed yet. Run process_directory first.")
    
    # Get the embedding models
    embedding_model = get_embedding_models(text_model, code_model)
    
    # Prepare text representations for embedding
    text_documents = []
    code_documents = []
    
    for doc in processed_documents:
        # Convert to dictionary for processing
        doc_dict = doc.dict()
        
        text_representation = textify_code(doc_dict)
        text_documents.append(text_representation)
        
        # Use the original code for code embeddings
        code_documents.append(doc_dict["content"])
    
    # Generate embeddings in batches
    start_time = time.time()
    
    text_embeddings = []
    code_embeddings = []
    
    # Process in batches
    for i in range(0, len(text_documents), batch_size):
        batch_text = text_documents[i:i+batch_size]
        batch_code = code_documents[i:i+batch_size]
        
        batch_text_emb, batch_code_emb = embedding_model.embed_documents(batch_text, batch_code)
        
        text_embeddings.extend(batch_text_emb)
        code_embeddings.extend(batch_code_emb)
        
        print(f"Processed batch {i//batch_size + 1}/{(len(text_documents)-1)//batch_size + 1}")
    
    embedding_time = time.time() - start_time
    
    # Get vector dimensions
    text_vector_dim = embedding_model.get_text_vector_dimension()
    code_vector_dim = embedding_model.get_code_vector_dimension()
    
    # Save the embeddings
    if storage_type.lower() == "qdrant":
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant is not available. Install with 'pip install qdrant-client'")
        
        # Save to Qdrant
        collection_name = output_file
        
        # Connect to Qdrant (default local in-memory instance)
        client = QdrantClient(":memory:")
        
        # Check if collection exists and create if not
        collections = client.get_collections().collections
        collection_exists = any(c.name == collection_name for c in collections)
        
        if not collection_exists:
            # Create collection with two vector types
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "text": models.VectorParams(
                        size=text_vector_dim,
                        distance=models.Distance.COSINE,
                    ),
                    "code": models.VectorParams(
                        size=code_vector_dim,
                        distance=models.Distance.COSINE,
                    ),
                }
            )
        
        # Upload points in batches
        points = []
        for i, (doc, text_emb, code_emb) in enumerate(zip(processed_documents, text_embeddings, code_embeddings)):
            doc_dict = doc.dict()
            
            # Create payload
            payload = {
                "document_id": doc_dict["document_id"],
                "file_path": doc_dict["file_path"],
                "file_name": doc_dict["file_name"],
                "file_type": doc_dict["file_type"],
                "module_name": doc_dict.get("module_name", ""),
                "class_name": doc_dict.get("class_name", ""),
                "function_name": doc_dict.get("function_name", ""),
                "is_example": doc_dict.get("is_example", False),
                "is_api_reference": doc_dict.get("is_api_reference", False),
                "content": doc_dict["content"][:500],  # Truncate content to save space
                "code_type": "Function" if doc_dict.get("function_name") else "Class" if doc_dict.get("class_name") else "File",
            }
            
            points.append(
                models.PointStruct(
                    id=i,
                    vector={
                        "text": text_emb,
                        "code": code_emb,
                    },
                    payload=payload,
                )
            )
            
            # Upload in batches
            if len(points) >= batch_size or i == len(processed_documents) - 1:
                client.upload_points(
                    collection_name=collection_name,
                    points=points,
                )
                points = []
        
        # For consistency, also save a small pickle file with metadata
        storage = EmbeddingStorage()
        storage.save_metadata(
            {
                "storage_type": "qdrant",
                "collection_name": collection_name,
                "text_model": text_model,
                "code_model": code_model,
                "text_vector_dimension": text_vector_dim,
                "code_vector_dimension": code_vector_dim,
                "total_documents": len(processed_documents),
                "embedding_time_seconds": embedding_time
            },
            f"{collection_name}_metadata.pkl"
        )
        
    else:
        # Save to pickle file (traditional method)
        storage = EmbeddingStorage()
        
        # Convert Pydantic models to dictionaries
        doc_dicts = [doc.dict() for doc in processed_documents]
        
        # Save to either the specified file or the default location
        storage.save_embeddings(
            doc_dicts,
            text_embeddings,
            code_embeddings,
            {
                "text_model": text_model,
                "code_model": code_model,
                "text_vector_dimension": text_vector_dim,
                "code_vector_dimension": code_vector_dim,
                "total_documents": len(processed_documents),
                "embedding_time_seconds": embedding_time
            },
            output_file
        )
    
    # Create the result statistics
    result = EmbeddingStats(
        total_documents=len(processed_documents),
        text_embedding_model=text_model,
        code_embedding_model=code_model,
        text_vector_dimension=text_vector_dim,
        code_vector_dimension=code_vector_dim,
        embedding_time_seconds=embedding_time
    )
    
    return result.dict()

@function_tool
def search_documents(
    query: str,
    embeddings_source: str,
    top_k: int = 5,
    use_text_embeddings: bool = True,
    use_code_embeddings: bool = True,
    group_by: Optional[str] = None,  # Group results by field (e.g., "module_name")
    storage_type: str = "pickle"  # Options: "pickle", "qdrant"
) -> List[Dict[str, Any]]:
    """
    Search for documents similar to the query.
    
    Args:
        query: The query string
        embeddings_source: Path to the embeddings file or collection name in Qdrant
        top_k: Number of results to return
        use_text_embeddings: Whether to use text embeddings
        use_code_embeddings: Whether to use code embeddings
        group_by: Group results by a specific field (e.g., "module_name")
        storage_type: Type of storage ("pickle" or "qdrant")
        
    Returns:
        List of search results
    """
    if storage_type.lower() == "qdrant":
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant is not available. Install with 'pip install qdrant-client'")
        
        # Load metadata for model information
        metadata_file = f"{embeddings_source}_metadata.pkl"
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, "rb") as f:
                    metadata = pickle.load(f)
                text_model = metadata.get("text_model", "sentence-transformers/all-MiniLM-L6-v2")
                code_model = metadata.get("code_model", "jinaai/jina-embeddings-v2-base-code")
            except Exception:
                # Default models if metadata can't be loaded
                text_model = "sentence-transformers/all-MiniLM-L6-v2"
                code_model = "jinaai/jina-embeddings-v2-base-code"
        else:
            # Default models if metadata doesn't exist
            text_model = "sentence-transformers/all-MiniLM-L6-v2"
            code_model = "jinaai/jina-embeddings-v2-base-code"
        
        # Get embedding models
        embedding_model = get_embedding_models(text_model, code_model)
        
        # Generate query embeddings
        query_text_embedding, query_code_embedding = embedding_model.embed_query(query)
        
        # Connect to Qdrant
        client = QdrantClient(":memory:")
        
        if group_by:
            # Search with grouping
            results = client.query_points_groups(
                collection_name=embeddings_source,
                prefetch=[
                    models.Prefetch(
                        query=query_text_embedding if use_text_embeddings else None,
                        using="text",
                        limit=top_k * 2,
                    ) if use_text_embeddings else None,
                    models.Prefetch(
                        query=query_code_embedding if use_code_embeddings else None,
                        using="code",
                        limit=top_k * 2,
                    ) if use_code_embeddings else None,
                ],
                group_by=group_by,
                group_size=1,
                limit=top_k,
                query=models.FusionQuery(fusion=models.Fusion.RRF) if (use_text_embeddings and use_code_embeddings) else None
            )
            
            # Format results
            formatted_results = []
            for group in results.groups:
                for hit in group.hits:
                    formatted_results.append({
                        "group": group.id,
                        "file_path": hit.payload["file_path"],
                        "file_name": hit.payload["file_name"],
                        "code_type": hit.payload["code_type"],
                        "function_name": hit.payload.get("function_name", ""),
                        "class_name": hit.payload.get("class_name", ""),
                        "module_name": hit.payload.get("module_name", ""),
                        "similarity": hit.score,
                        "snippet": hit.payload["content"]
                    })
            
            return formatted_results
        else:
            # Standard search without grouping
            results = client.query_points(
                collection_name=embeddings_source,
                prefetch=[
                    models.Prefetch(
                        query=query_text_embedding if use_text_embeddings else None,
                        using="text",
                        limit=top_k * 2,
                    ) if use_text_embeddings else None,
                    models.Prefetch(
                        query=query_code_embedding if use_code_embeddings else None,
                        using="code",
                        limit=top_k * 2,
                    ) if use_code_embeddings else None,
                ],
                limit=top_k,
                query=models.FusionQuery(fusion=models.Fusion.RRF) if (use_text_embeddings and use_code_embeddings) else None
            )
            
            # Format results
            formatted_results = []
            for hit in results.points:
                formatted_results.append({
                    "file_path": hit.payload["file_path"],
                    "file_name": hit.payload["file_name"],
                    "code_type": hit.payload["code_type"],
                    "function_name": hit.payload.get("function_name", ""),
                    "class_name": hit.payload.get("class_name", ""),
                    "module_name": hit.payload.get("module_name", ""),
                    "similarity": hit.score,
                    "snippet": hit.payload["content"]
                })
            
            return formatted_results
    else:
        # Use the existing search implementation for pickle files
        from .search import search_embeddings_file
        
        # Add group_by parameter if supported
        if group_by:
            return search_embeddings_file(
                query=query,
                embeddings_file=embeddings_source,
                top_k=top_k,
                use_text_embeddings=use_text_embeddings,
                use_code_embeddings=use_code_embeddings,
                group_by=group_by
            )
        else:
            return search_embeddings_file(
                query=query,
                embeddings_file=embeddings_source,
                top_k=top_k,
                use_text_embeddings=use_text_embeddings,
                use_code_embeddings=use_code_embeddings
            )

# Initialize an empty list for storing processed documents
processed_documents = []

# Create the agent
embedder_agent = Agent(
    name="Document Embedder",
    instructions="""
    You are an expert document embedder agent that processes codebases to generate searchable embeddings.
    
    You support two types of embeddings:
    1. Text embeddings - for natural language queries
    2. Code embeddings - for code-to-code similarity search
    
    Your main functions are:
    - Process a directory to extract code and documentation with granular chunking
    - Generate dual embeddings for the processed documents
    - Search for similar documents using natural language or code snippets
    - Support multiple storage backends including pickle files and Qdrant vector database
    
    When processing a directory, you'll automatically identify functions, classes, methods, and code chunks.
    When generating embeddings, you'll convert code to optimized text representations for text embeddings and keep the original code for code embeddings.
    When searching, you'll combine text and code search results using Reciprocal Rank Fusion for optimal results.
    You can also group search results by module or other metadata fields.
    
    The system supports two storage types:
    - pickle: Traditional file storage (default)
    - qdrant: Vector database for scalable search
    """,
    tools=[
        process_directory,
        generate_embeddings,
        search_documents,
    ],
    model_settings=ModelSettings(
        temperature=0.1,
    ),
    input_guardrails=[
        create_query_guardrail(),
        create_directory_guardrail(),
    ]
) 