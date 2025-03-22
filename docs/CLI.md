# Code Index MCP Command Line Interface

Code Index MCP provides a comprehensive command-line interface (CLI) for managing your code indexing and search.

## Configuration

All configuration is managed through environment variables or a `.env` file in the project root:

```
# Project directory to index (required)
CODE_INDEX_DIRECTORY=/path/to/your/project

# For embeddings (required)
OPEN_AI_KEY=sk-your-openai-api-key

# For Qdrant Cloud (optional)
QDRANT_URL=https://your-cluster-url.qdrant.tech
QDRANT_API_KEY=your_qdrant_api_key

# Auto-indexing (optional)
CODE_INDEX_AUTO_INDEX=true
```

## Available Commands

The CLI supports two main commands:

1. `server` - Start the MCP server
2. `index` - Index a directory without starting a server

## Common Options

### Server Command

```bash
python -m code_index_mcp.cli.commands server [OPTIONS]
```

Options:
- `--directory`, `-d` : Project directory to index
- `--port`, `-p` : Port to run the server on (default: 8000)
- `--host` : Host to run the server on (default: 127.0.0.1)
- `--reload` : Enable auto-reload of the server when code changes
- `--use-qdrant-cloud` : Use Qdrant Cloud instead of local instance
- `--extensions` : File extensions to index (space-separated list)
- `--auto-index` : Enable auto-indexing of files when they change

Examples:

```bash
# Start server with default settings using directory from .env
python -m code_index_mcp.cli.commands server

# Start server with specific directory and auto-indexing
python -m code_index_mcp.cli.commands server --directory /path/to/your/project --auto-index

# Start server with specific file extensions
python -m code_index_mcp.cli.commands server --extensions py js ts

# Start server with Qdrant Cloud
python -m code_index_mcp.cli.commands server --use-qdrant-cloud
```

### Index Command

```bash
python -m code_index_mcp.cli.commands index DIRECTORY [OPTIONS]
```

Options:
- `--extensions` : File extensions to index (space-separated list)
- `--use-qdrant-cloud` : Use Qdrant Cloud instead of local instance

Examples:

```bash
# Index a directory
python -m code_index_mcp.cli.commands index /path/to/your/project

# Index only Python and JavaScript files
python -m code_index_mcp.cli.commands index /path/to/your/project --extensions py js

# Index using Qdrant Cloud
python -m code_index_mcp.cli.commands index /path/to/your/project --use-qdrant-cloud
```

## Using with the Scripts

The project includes helper scripts that make it easier to run the commands:

### Unix/Linux/macOS

```bash
# Start server
./scripts/start_indexer.sh [DIRECTORY] [OPTIONS]

# Options:
#   --use-qdrant-cloud : Use Qdrant Cloud
#   --auto-index : Enable auto-indexing
```

### Windows

```powershell
# Start server
.\scripts\start_indexer.ps1 [DIRECTORY] [OPTIONS]

# Options:
#   --use-qdrant-cloud : Use Qdrant Cloud
#   --auto-index : Enable auto-indexing
```

## Environment Variables

The CLI respects the following environment variables:

- `CODE_INDEX_DIRECTORY` : Project directory to index
- `CODE_INDEX_PORT` : Port to run the server on
- `CODE_INDEX_HOST` : Host to run the server on
- `CODE_INDEX_RELOAD` : Enable auto-reload (1/0 or true/false)
- `CODE_INDEX_USE_QDRANT_CLOUD` : Use Qdrant Cloud (1/0 or true/false)
- `CODE_INDEX_AUTO_INDEX` : Enable auto-indexing (1/0 or true/false)
- `CODE_INDEX_EXTENSIONS` : Comma-separated list of file extensions to index

These can be set in your shell or in a `.env` file in the project root.

## Examples

### Basic Usage

```bash
# Index a project and start the server
python -m code_index_mcp.cli.commands server --directory /path/to/project

# Just index a project without starting a server
python -m code_index_mcp.cli.commands index /path/to/project
```

### Advanced Usage

```bash
# Start a server with custom port, auto-indexing, and specific extensions
python -m code_index_mcp.cli.commands server \
  --directory /path/to/project \
  --port 9000 \
  --auto-index \
  --extensions py js ts jsx tsx

# Index a project and use Qdrant Cloud
python -m code_index_mcp.cli.commands index \
  /path/to/project \
  --use-qdrant-cloud
```
