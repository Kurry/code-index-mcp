# Code Index MCP Development Summary

## Overview of Our Conversation

We've been working on enhancing the `code-index-mcp` project to create a more powerful code indexing and search tool that uses embeddings for semantic code search. Our work has proceeded through several phases:

### Phase 1: Initial Simplification

We first looked at simplifying the MCP configuration by adding a simpler `uvx` entry point in the `pyproject.toml` file. This allowed for a simpler MCP configuration in both Claude for Mac and Cursor:

```json
{
  "mcpServers": {
    "code-indexer": {
      "command": "uvx",
      "args": [
        "/path/to/code-index-mcp"
      ]
    }
  }
}
```

This involved:
- Adding the `uvx` entry point in `pyproject.toml`
- Updating the README to document this simpler configuration
- Adding Cursor integration instructions with the proper configuration paths

### Phase 2: Pre-indexing Feature

We then explored adding a feature to allow pre-indexing a directory at startup:
- Added a command-line argument (`--index`) to pre-index a directory
- Updated the main function to handle this argument
- Created installation scripts that support pre-indexing

### Phase 3: Complete Vector Database Redesign

The most significant work was a complete redesign using vector embeddings:
- Created an `EmbeddingManager` class to generate and store embeddings
- Added support for both OpenAI and Voyage AI embedding providers
- Implemented real-time file change monitoring
- Redesigned the server to require a directory path at startup
- Added semantic search capabilities

### Phase 4: Environment Variable Support

We've added support for loading API keys from a `.env` file:
- Added `python-dotenv` as a dependency
- Created a `.env.example` file with sample configuration
- Modified the startup scripts to check for a `.env` file
- Updated error messages to suggest creating a `.env` file
- Made the startup process more user-friendly

### Phase 5: Qdrant Integration for Vector Search

We've completely revamped the design to use Qdrant for vector search:
- Implemented dual model architecture (NLP + Code-specific models)
- Added code-to-text conversion for better natural language querying
- Integrated with Qdrant vector database (both local and cloud options)
- Added semantic search with grouping by module
- Improved search capabilities with fusion of results from multiple models

### Phase 6: Project Directory Configuration in .env

We've enhanced the tool to allow specifying the project directory in the .env file:
- Added `PROJECT_DIRECTORY` setting in .env.example
- Made the directory argument optional in server.py
- Updated the startup scripts to check for directory in .env if not provided as argument
- Improved the README to document both approaches for specifying the directory
- Made startup more flexible with multiple configuration options

### Key Files Created/Modified

1. **New Files:**
   - `src/code_index_mcp/embedding_manager.py`: Complete rewrite using Qdrant and dual models
   - `server.py`: Updated server with Qdrant integration and new search capabilities
   - `run.py`: Updated entry point
   - `README.md`: Comprehensive documentation for Qdrant-based approach
   - `start_indexer.sh` and `start_indexer.ps1`: Convenience scripts with Qdrant Cloud support
   - `.env.example`: Template for environment variables including Qdrant API keys and project directory

2. **Modified Files:**
   - `pyproject.toml`: Added new dependencies (fastembed, qdrant-client, inflection)
   - `.gitignore`: Added .env to ensure API keys aren't committed

## Technical Implementation Details

### Dual-Model Embedding Approach

The new implementation uses two specialized models:
1. **NLP Model** (`sentence-transformers/all-MiniLM-L6-v2`): For natural language queries
2. **Code Model** (`jinaai/jina-embeddings-v2-base-code`): For code-to-code similarity

This dual approach allows for more powerful searching:
- Users can ask natural language questions about the codebase
- Users can find code similar to a given snippet
- Results from both models can be fused for comprehensive searching

### Code-to-Text Conversion

We've implemented the `textify` function to convert code to natural language:
- Breaks down camelCase and snake_case into natural words
- Extracts context from module names, file paths, and docstrings
- Creates sentences that describe the code in natural language
- Makes code more searchable via text queries

### Qdrant Vector Database

We've replaced FAISS with Qdrant for several benefits:
- Multi-vector search capabilities (storing both NLP and code embeddings)
- Result grouping by module, file, or other properties
- Fusion search combining results from multiple models
- Cloud deployment option for production use

### Project Directory Configuration

We now support multiple ways to specify the project directory:
1. **In .env file (recommended)**: 
   - Set `PROJECT_DIRECTORY=/path/to/your/project` in .env 
   - This persists between runs for convenience

2. **Command line argument (override)**:
   - Pass directory as first argument: `./start_indexer.sh /path/to/override`
   - Useful for temporary scanning of different projects

3. **Claude/Cursor integration**:
   - Set in the args array of the MCP configuration
   - Overrides both .env and environment variables

### API Key Management

We support multiple methods for API key management:
1. **Environment Variables:**
   - `OPEN_AI_KEY` for embeddings
   - `QDRANT_API_KEY` and `QDRANT_URL` for Qdrant Cloud
   - `PROJECT_DIRECTORY` for the code directory to index

2. **.env File:**
   - Local `.env` file based on `.env.example`
   - Automatically loaded by `python-dotenv`
   - More secure and convenient for development

### Command-line Interface

The new server has multiple ways to run:
```bash
# Using directory from .env (simplest)
./start_indexer.sh

# Specifying directory explicitly
./start_indexer.sh /path/to/your/project

# Using Qdrant Cloud with directory from .env
./start_indexer.sh --use-qdrant-cloud
```

### Integration with Claude and Cursor

For Claude/Cursor integration, the configuration includes the project directory:
```json
{
  "mcpServers": {
    "code-indexer": {
      "command": "uvx",
      "args": [
        "/path/to/code-index-mcp",
        "/path/to/your/project",
        "--use-qdrant-cloud"  # Optional
      ]
    }
  }
}
```

## Current Status

We have a production-ready implementation of the embedding-based code indexer with Qdrant. The main features are:

- Dual model architecture for both text and code understanding
- Qdrant vector database integration (local or cloud)
- Semantic search with natural language
- Code similarity search
- Results grouping by module
- Real-time file change monitoring
- Flexible configuration via .env file
- Simple Claude/Cursor integration
- Multiple ways to specify the project directory

## Where to Pick Up Next Time

When we continue, we should focus on the following items:

1. **Implement advanced code chunking**
   - Currently, we treat each file as one chunk
   - We should implement language-specific chunking using AST parsers
   - This would involve splitting files into logical chunks (functions, classes, etc.)
   - Create a chunking strategy that preserves code hierarchy

2. **Optimize embedding generation**
   - Implement parallel processing for large codebases
   - Add progress tracking for long indexing operations
   - Create a resume capability for interrupted indexing

3. **Add advanced search features**
   - Implement hybrid search (vector + keyword)
   - Add filtering by language, file type, function type
   - Create a custom ranking function that considers code complexity

4. **Improve result presentation**
   - Add code highlighting in results
   - Create a context-aware result formatter
   - Generate summaries of code functionality

5. **Testing and benchmarking**
   - Test with large open-source repositories
   - Compare search quality between different models
   - Benchmark performance with different Qdrant configurations

6. **User experience improvements**
   - Create a simple web UI for direct interaction
   - Add search history and bookmarking
   - Implement user feedback collection

These enhancements would take the tool from its current solid foundation to a truly exceptional developer productivity tool.

## Key Decision Points

When we resume, we need to decide on:

1. Which chunking strategy to use for different languages
2. How to handle very large codebases efficiently
3. Whether to add a web UI component
4. How to implement hybrid search for better results
5. Whether to specialize in certain languages or maintain a general approach

These decisions will shape the direction of the project going forward.
