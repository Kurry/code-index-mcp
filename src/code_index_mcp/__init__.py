"""Code Index MCP package.

A Model Context Protocol server for code indexing, searching, and analysis.
"""

__version__ = "0.1.0"

from .core import EmbeddingManager, FileChangeHandler

__all__ = ["EmbeddingManager", "FileChangeHandler"]
