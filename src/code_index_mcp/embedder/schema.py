#!/usr/bin/env python3
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
import numpy as np

class ProcessingStats(BaseModel):
    """
    Statistics about processed documents.
    """
    total_files: int = Field(
        default=0,
        description="Total number of files found"
    )
    processed_files: int = Field(
        default=0,
        description="Number of files successfully processed"
    )
    files_by_extension: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of files by extension"
    )
    bytes_processed: int = Field(
        default=0,
        description="Total bytes processed"
    )
    function_count: int = Field(
        default=0,
        description="Number of functions found"
    )
    class_count: int = Field(
        default=0,
        description="Number of classes found"
    )
    method_count: int = Field(
        default=0,
        description="Number of class methods found"
    )
    elapsed_time: float = Field(
        default=0.0,
        description="Processing time in seconds"
    )

class EmbeddingStats(BaseModel):
    """
    Statistics about embeddings.
    """
    document_count: int = Field(
        default=0,
        description="Number of documents embedded"
    )
    text_model: str = Field(
        default="",
        description="Text embedding model used"
    )
    code_model: str = Field(
        default="",
        description="Code embedding model used"
    )
    elapsed_time: float = Field(
        default=0.0,
        description="Embedding time in seconds"
    )

class SearchMetadata(BaseModel):
    """
    Metadata about a document.
    """
    file_path: str = Field(
        description="Path to the file"
    )
    relative_path: Optional[str] = Field(
        default=None,
        description="Relative path to the file"
    )
    file_name: str = Field(
        description="Name of the file"
    )
    file_type: str = Field(
        description="Type of the file (extension)"
    )
    document_type: str = Field(
        description="Type of document (file, function, class, documentation)"
    )
    class_name: Optional[str] = Field(
        default=None,
        description="Name of the class (if document_type is class or method)"
    )
    function_name: Optional[str] = Field(
        default=None,
        description="Name of the function (if document_type is function)"
    )
    module_name: Optional[str] = Field(
        default=None,
        description="Name of the module"
    )
    start_line: Optional[int] = Field(
        default=None,
        description="Start line number of the document in the file"
    )
    end_line: Optional[int] = Field(
        default=None,
        description="End line number of the document in the file"
    )
    language: Optional[str] = Field(
        default=None,
        description="Programming language of the file"
    )
    keywords: Optional[List[str]] = Field(
        default=None,
        description="Keywords extracted from the document"
    )

class Document(BaseModel):
    """
    A document with content and metadata.
    """
    id: str = Field(
        description="Unique identifier for the document"
    )
    content: str = Field(
        description="Document content"
    )
    metadata: SearchMetadata = Field(
        description="Document metadata"
    )

class SearchResult(BaseModel):
    """
    A search result.
    """
    id: str = Field(
        description="Document ID"
    )
    content: str = Field(
        description="Document content"
    )
    metadata: SearchMetadata = Field(
        description="Document metadata"
    )
    text_score: float = Field(
        default=0.0,
        description="Text similarity score"
    )
    code_score: float = Field(
        default=0.0,
        description="Code similarity score"
    )
    combined_score: float = Field(
        default=0.0,
        description="Combined similarity score"
    )
    snippet: Optional[str] = Field(
        default=None,
        description="Document snippet"
    )
    
    class Config:
        arbitrary_types_allowed = True

class EmbeddingData(BaseModel):
    """
    Embedding data for documents.
    """
    documents: List[Document] = Field(
        description="List of documents"
    )
    text_embeddings: Optional[np.ndarray] = Field(
        default=None,
        description="Text embeddings for documents"
    )
    code_embeddings: Optional[np.ndarray] = Field(
        default=None,
        description="Code embeddings for documents"
    )
    text_model: str = Field(
        description="Text embedding model used"
    )
    code_model: str = Field(
        description="Code embedding model used"
    )
    stats: EmbeddingStats = Field(
        description="Embedding statistics"
    )
    
    class Config:
        arbitrary_types_allowed = True
        
    def dict(self, *args, **kwargs):
        """
        Custom dict method to handle numpy arrays.
        """
        result = super().dict(*args, **kwargs)
        if self.text_embeddings is not None:
            result["text_embeddings"] = self.text_embeddings.tolist()
        if self.code_embeddings is not None:
            result["code_embeddings"] = self.code_embeddings.tolist()
        return result 