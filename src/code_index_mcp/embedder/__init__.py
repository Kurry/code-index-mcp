#!/usr/bin/env python3
"""
Embedder package for generating dual embeddings and performing semantic search over codebases.
"""

from .embedder import embedder_agent, process_directory, generate_embeddings, search_documents
from .search import search_embeddings_file, search_embeddings, search_agent

__all__ = [
    "embedder_agent",
    "process_directory",
    "generate_embeddings",
    "search_documents",
    "search_embeddings_file",
    "search_embeddings",
    "search_agent",
]
