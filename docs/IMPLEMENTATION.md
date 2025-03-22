# Code Index MCP Implementation with Qdrant Vector Search

This document details the implementation of the Code Index MCP with Qdrant Vector Search functionality.

## Configuration Files

### Environment Configuration

Update the `.env.example` file to include Qdrant configuration options:

```
# OpenAI API Key for embeddings
OPEN_AI_KEY=your_openai_api_key_here

# Qdrant Cloud Configuration (optional)
QDRANT_URL=https://your-cluster-url.qdrant.tech
QDRANT_API_KEY=your_qdrant_api_key_here
```

### Run Script Update

Updated `run.py` file to support Qdrant integration.

### Startup Scripts

Created startup scripts to support the Qdrant Cloud option:

- `start_indexer.sh` for Unix/Linux/macOS
- `start_indexer.ps1` for Windows
- Both support the `--use-qdrant-cloud` flag for switching between local and cloud mode

## Key Components

### 1. Embedding Manager (`embedding_manager.py`)

A comprehensive embedding manager that:
- Uses **dual models** for both natural language and code similarity search
- Converts code to natural language using the `textify` function
- Stores embeddings in Qdrant (either local in-memory or Qdrant Cloud)
- Handles file change monitoring to update embeddings in real-time
- Provides semantic search capabilities with results grouping

### 2. Server Implementation (`server.py`)

The server now offers advanced search capabilities:
- Semantic search using natural language or code similarity
- Group search results by module
- File analysis and content retrieval 
- Project structure navigation

### 3. Configuration with Environment Variables

Added support for Qdrant API keys and configuration:
- Updated the `.env.example` file to include Qdrant configuration
- Added checks for the required API keys
- Implemented loading from either environment variables or `.env` file

### 4. Startup Scripts

Created convenient startup scripts:
- `start_indexer.sh` for Unix/Linux/macOS
- `start_indexer.ps1` for Windows
- Both support the `--use-qdrant-cloud` flag for switching between local and cloud mode

### 5. Comprehensive Documentation

The README.md provides:
- Clear explanation of how the system works with dual embeddings
- Installation and setup instructions
- Configuration options for API keys
- Detailed usage instructions
- Integration guide for Claude Desktop and Cursor
- Troubleshooting tips

## Key Features

1. **Semantic Code Search**: Find code using natural language queries
2. **Code-to-Code Similarity**: Find similar code patterns
3. **Grouped Results**: See related code across different modules
4. **Real-time Updates**: File changes are detected and embeddings are updated automatically
5. **Qdrant Integration**: Uses either local in-memory Qdrant or Qdrant Cloud for production
6. **Language-specific Analysis**: Provides specialized analysis for Python, JavaScript, and other languages

## How to Use This System

Once you have the server running, you can interact with it through Claude or Cursor to:

1. **Search with Natural Language**:
   - Ask questions like "How does error handling work in this codebase?"
   - The system will use the NLP model embeddings to find relevant code

2. **Search with Code Patterns**:
   - Provide a code snippet to find similar code patterns
   - The system will use the code model embeddings for more precise matching

3. **Browse Results by Module**:
   - Group results by module to get a broad overview of patterns across the codebase
   - Each module provides a representative code sample

## Local vs. Cloud Deployment

The system supports two deployment modes:

- **Local (In-Memory)**: Great for development and small to medium codebases
  - No additional setup required
  - Data is lost when the server restarts

- **Cloud (Qdrant Cloud)**: Ideal for production and larger codebases
  - Persistent storage across restarts
  - Better performance for large vector collections
  - Requires QDRANT_API_KEY and QDRANT_URL in your .env file

## Next Steps and Improvements

To further enhance this system, you could:

1. **Implement Better Chunking**: Use language-specific parsers to extract meaningful code structures
2. **Add Code Context**: Preserve relationships between functions, classes, and files
3. **Support More Languages**: Add specialized parsing for additional programming languages
4. **Implement Reranking**: Add a reranking step to improve search relevance
5. **Add User Feedback Loop**: Allow users to provide feedback on search results

The current implementation provides a strong foundation for semantic code search, and these enhancements would make it even more powerful. 