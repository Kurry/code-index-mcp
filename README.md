# Code Index MCP with Qdrant Vector Search

A Model Context Protocol server for code indexing, searching, and analysis.

## Key Features

- Semantic code search using dual embeddings (text and code models)
- Natural language understanding of your codebase
- Code-to-code similarity search
- Real-time file change monitoring and auto-indexing
- Web UI for easy management
- Integration with Claude Desktop and Cursor

## Quick Start

First, configure your project directory in the `.env` file:

```bash
# Copy the example .env file
cp .env.example .env

# Edit the .env file and set your project directory
echo "CODE_INDEX_DIRECTORY=/path/to/your/project" >> .env
```

Then run the components:

```bash
# Install
./scripts/install.sh

# Run the server with auto-indexing (Unix)
./scripts/start_indexer.sh --auto-index

# Run the server with auto-indexing (Windows)
.\scripts\start_indexer.ps1 --auto-index

# Run the web interface (uses CODE_INDEX_DIRECTORY from .env)
python run_streamlit.py
```

## Project Structure

The project has a clean, organized structure:

```
code-index-mcp/
├── docs/                     # Documentation files
├── scripts/                  # Helper scripts
├── src/                      # Source code
│   └── code_index_mcp/       # Main package
│       ├── cli/              # Command-line interface
│       ├── core/             # Core functionality
│       ├── embedder/         # Embedding functionality
│       ├── server/           # MCP server implementation
│       └── web/              # Web UI implementation
└── tests/                    # Unit tests
```

## Configuration

All configuration is managed through environment variables or a `.env` file:

- `CODE_INDEX_DIRECTORY`: The directory to index (required)
- `OPEN_AI_KEY`: Your OpenAI API key for embeddings
- `QDRANT_URL`: Qdrant Cloud URL (optional)
- `QDRANT_API_KEY`: Qdrant Cloud API key (optional)

## Using with LLMs

This MCP server allows Large Language Models to:

1. Index your entire codebase with semantic understanding
2. Search for relevant code using natural language
3. Analyze code structure and patterns
4. Answer questions about your code with context

For complete documentation, please see the [docs/README.md](docs/README.md) file.
