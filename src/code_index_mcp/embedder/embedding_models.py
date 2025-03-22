#!/usr/bin/env python3
import os
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from tqdm import tqdm
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DualEmbeddingModel:
    """
    Manages both text and code embedding models.
    """
    
    def __init__(
        self, 
        text_model_name: str = "text-embedding-3-small", 
        code_model_name: str = "text-embedding-3-small"
    ):
        """
        Initialize the dual embedding model.
        
        Args:
            text_model_name: Name of the text embedding model
            code_model_name: Name of the code embedding model
        """
        self.client = OpenAI()
        self.text_model_name = text_model_name
        self.code_model_name = code_model_name
        
        # Cache dimensions
        self._text_dim = None
        self._code_dim = None
    
    def get_text_dimension(self) -> int:
        """
        Get the dimension of the text embedding model.
        
        Returns:
            Dimension of the text embedding model
        """
        if self._text_dim is None:
            # Create a test embedding to get dimensions
            embedding = self.client.embeddings.create(
                model=self.text_model_name,
                input="Test",
                encoding_format="float"
            ).data[0].embedding
            self._text_dim = len(embedding)
        
        return self._text_dim
    
    def get_code_dimension(self) -> int:
        """
        Get the dimension of the code embedding model.
        
        Returns:
            Dimension of the code embedding model
        """
        if self._code_dim is None:
            # Create a test embedding to get dimensions
            embedding = self.client.embeddings.create(
                model=self.code_model_name,
                input="Test",
                encoding_format="float"
            ).data[0].embedding
            self._code_dim = len(embedding)
        
        return self._code_dim
    
    def embed_documents(
        self, 
        text_documents: List[str], 
        code_documents: List[str],
        batch_size: int = 16,
        show_progress: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate embeddings for lists of text and code documents.
        
        Args:
            text_documents: List of text documents
            code_documents: List of code documents
            batch_size: Number of documents to embed at once
            show_progress: Whether to show progress bar
            
        Returns:
            Tuple of (text_embeddings, code_embeddings)
        """
        if len(text_documents) != len(code_documents):
            raise ValueError("Text and code document lists must have the same length")
        
        # Initialize arrays for embeddings
        n_docs = len(text_documents)
        text_dim = self.get_text_dimension()
        code_dim = self.get_code_dimension()
        
        text_embeddings = np.zeros((n_docs, text_dim), dtype=np.float32)
        code_embeddings = np.zeros((n_docs, code_dim), dtype=np.float32)
        
        # Process in batches
        iterator = range(0, n_docs, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating embeddings", total=len(iterator))
        
        for i in iterator:
            batch_end = min(i + batch_size, n_docs)
            batch_text = text_documents[i:batch_end]
            batch_code = code_documents[i:batch_end]
            
            batch_text_embeddings, batch_code_embeddings = self.embed_batch(batch_text, batch_code)
            
            text_embeddings[i:batch_end] = batch_text_embeddings
            code_embeddings[i:batch_end] = batch_code_embeddings
        
        return text_embeddings, code_embeddings
    
    def embed_batch(
        self, 
        text_batch: List[str], 
        code_batch: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate embeddings for a batch of documents.
        
        Args:
            text_batch: Batch of text documents
            code_batch: Batch of code documents
            
        Returns:
            Tuple of (text_embeddings, code_embeddings)
        """
        # Generate text embeddings
        try:
            text_response = self.client.embeddings.create(
                model=self.text_model_name,
                input=text_batch,
                encoding_format="float"
            )
            
            text_embeddings = np.array([data.embedding for data in text_response.data])
        except Exception as e:
            logger.error(f"Error generating text embeddings: {e}")
            text_embeddings = np.zeros((len(text_batch), self.get_text_dimension()))
        
        # Generate code embeddings
        try:
            code_response = self.client.embeddings.create(
                model=self.code_model_name,
                input=code_batch,
                encoding_format="float"
            )
            
            code_embeddings = np.array([data.embedding for data in code_response.data])
        except Exception as e:
            logger.error(f"Error generating code embeddings: {e}")
            code_embeddings = np.zeros((len(code_batch), self.get_code_dimension()))
        
        return text_embeddings, code_embeddings
    
    def embed_query(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate embeddings for a query.
        
        Args:
            query: Query string
            
        Returns:
            Tuple of (text_embedding, code_embedding)
        """
        # Generate text embedding
        try:
            text_response = self.client.embeddings.create(
                model=self.text_model_name,
                input=query,
                encoding_format="float"
            )
            
            text_embedding = np.array(text_response.data[0].embedding)
        except Exception as e:
            logger.error(f"Error generating text embedding for query: {e}")
            text_embedding = np.zeros(self.get_text_dimension())
        
        # Generate code embedding
        try:
            code_response = self.client.embeddings.create(
                model=self.code_model_name,
                input=query,
                encoding_format="float"
            )
            
            code_embedding = np.array(code_response.data[0].embedding)
        except Exception as e:
            logger.error(f"Error generating code embedding for query: {e}")
            code_embedding = np.zeros(self.get_code_dimension())
        
        return text_embedding, code_embedding

def get_embedding_models(
    text_model_name: str = "text-embedding-3-small",
    code_model_name: str = "text-embedding-3-small"
) -> DualEmbeddingModel:
    """
    Get the dual embedding model.
    
    Args:
        text_model_name: Name of the text embedding model
        code_model_name: Name of the code embedding model
        
    Returns:
        Dual embedding model
    """
    return DualEmbeddingModel(text_model_name, code_model_name)

def get_available_models() -> Dict[str, Dict[str, Any]]:
    """
    Get information about available embedding models.
    
    Returns:
        Dictionary mapping model types to available models
    """
    text_models = {
        "text-embedding-3-small": {
            "dimensions": 1536,
            "description": "Smallest text embedding model with good performance",
            "tokens_per_minute": 100000
        },
        "text-embedding-3-large": {
            "dimensions": 3072,
            "description": "Most powerful text embedding model",
            "tokens_per_minute": 100000
        },
        "text-embedding-ada-002": {
            "dimensions": 1536,
            "description": "Legacy text embedding model",
            "tokens_per_minute": 150000
        }
    }
    
    code_models = {
        "text-embedding-3-small": {
            "dimensions": 1536,
            "description": "General purpose model that works well for code",
            "tokens_per_minute": 100000
        },
        "text-embedding-3-large": {
            "dimensions": 3072,
            "description": "Most powerful model for code embedding",
            "tokens_per_minute": 100000
        },
        "text-embedding-ada-002": {
            "dimensions": 1536,
            "description": "Legacy model that works for code embedding",
            "tokens_per_minute": 150000
        }
    }
    
    return {
        "text_models": text_models,
        "code_models": code_models
    } 