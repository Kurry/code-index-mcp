# Code Index MCP Quick Start Guide

This guide will help you get started with Code Index MCP quickly.

## Prerequisites

- Python 3.10 or later
- Git
- uv (recommended for dependency management)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/code-index-mcp.git
   cd code-index-mcp
   ```

2. **Install uv** (if not already installed):
   ```bash
   # Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Install dependencies**:
   ```bash
   uv pip install -e .
   ```

   Alternatively, you can use the installation scripts:
   ```bash
   # Unix
   ./scripts/install.sh
   # Windows
   .\scripts\install.ps1
   ```

## Configuration

1. **Create a .env file** for your API keys:
   ```bash
   cp .env.example .env
   ```

2. **Edit the .env file** with your API keys and project directory:
   ```
   # Project directory to index
   PROJECT_DIRECTORY=/Users/username/my-project

   # For embeddings
   OPEN_AI_KEY=sk-your-openai-api-key

   # For Qdrant Cloud (optional)
   QDRANT_URL=https://your-cluster-url.qdrant.tech
   QDRANT_API_KEY=your_qdrant_api_key
   ```

## Running the Server

### Method 1: Using the provided scripts

```bash
# Unix - Use directory from .env
./scripts/start_indexer.sh

# Unix - Use a specific directory with auto-indexing
./scripts/start_indexer.sh /path/to/your/project --auto-index

# Windows - Use directory from .env
.\scripts\start_indexer.ps1

# Windows - Use a specific directory with auto-indexing
.\scripts\start_indexer.ps1 C:\path\to\your\project --auto-index
```

### Method 2: Using the Python module directly

```bash
# Start the server with a specific directory
python -m code_index_mcp.cli.commands server --directory /path/to/your/project

# Index a directory without starting a server
python -m code_index_mcp.cli.commands index /path/to/your/project
```

## Using the Web Interface

The web interface now uses the CODE_INDEX_DIRECTORY from your .env file or environment variables:

```bash
# First, make sure your .env file has the CODE_INDEX_DIRECTORY set
echo "CODE_INDEX_DIRECTORY=/path/to/your/project" >> .env

# Method 1: Using the run_streamlit.py script (recommended)
python run_streamlit.py

# Method 2: Running the module directly
python -m code_index_mcp.web

# Method 3: Running the Streamlit app directly
streamlit run src/code_index_mcp/web/streamlit_app.py
```

Navigate to `http://localhost:8501` in your browser to access the web interface.

## Integrating with Claude Desktop

1. Find or create the Claude Desktop configuration file:
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
   - macOS/Linux: `~/Library/Application Support/Claude/claude_desktop_config.json`

2. Add the following configuration:
   ```json
   {
     "mcpServers": {
       "code-indexer": {
         "command": "python",
         "args": [
            "-m", "code_index_mcp.cli.commands", "server",
            "--directory", "/path/to/your/project",
            "--auto-index"
          ]
       }
     }
   }
   ```

3. Restart Claude Desktop to use Code Indexer for analyzing code projects.

## Next Steps

- See [docs/README.md](README.md) for complete documentation
- Check out [docs/integration](integration/) for more integration options
- Learn about the [supported file types and features](README.md#supported-file-types)

## Troubleshooting

- If you encounter any issues, check the [Troubleshooting](README.md#troubleshooting) section in the main documentation
- Make sure API keys are correctly set in your .env file
- Verify Python version (3.10 or later required)
