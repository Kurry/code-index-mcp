# Code Index MCP with Qdrant Vector Search

Code Index MCP is a Model Context Protocol server that enables large language models (LLMs) to index, search, and analyze code in project directories using vector embeddings and Qdrant.

## Features

- Semantic code search using dual embedding models
- Natural language search across your codebase
- Code-to-code similarity search
- Real-time file change monitoring and index updates
- Integration with both local Qdrant and Qdrant Cloud
- Support for multiple programming languages
- Grouping search results by module or file
- Streamlit web interface for easy management

## How It Works

This tool uses two specialized embedding models:

1. **NLP Model**: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) for natural language understanding
2. **Code Model**: [jinaai/jina-embeddings-v2-base-code](https://huggingface.co/jinaai/jina-embeddings-v2-base-code) for code similarity

It processes your code in two ways:
- Converts code to natural language representation for semantic search
- Uses raw code for code-to-code similarity search

The embeddings are stored in Qdrant, a vector database optimized for similarity search.

## Project Structure

The project has a clean, organized structure:

```
code-index-mcp/
├── docs/                     # Documentation files
├── scripts/                  # Helper scripts for installation and starting the server
├── src/                      # Source code
│   └── code_index_mcp/       # Main package
│       ├── cli/              # Command-line interface
│       ├── core/             # Core functionality (embedding manager, etc.)
│       ├── embedder/         # Embedding functionality
│       ├── server/           # MCP server implementation
│       └── web/              # Web UI implementation with Streamlit
└── tests/                    # Unit tests
```

## Installation

This project uses uv for environment management and dependency installation.

1. Ensure you have Python 3.10 or later installed
2. Install uv (recommended):

   ```bash
   # Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
3. Getting the code:

   ```bash
   # Clone the repository
   git clone https://github.com/your-username/code-index-mcp.git
   cd code-index-mcp
   ```

4. Install the package with dependencies:
   ```bash
   uv pip install -e .
   ```

## Required Dependencies

The system uses the following main dependencies:
- `fastembed`: For generating vector embeddings
- `qdrant-client`: For vector similarity search
- `inflection`: For text normalization
- `watchdog`: For file monitoring
- `python-dotenv`: For environment variables
- `streamlit`: For the web UI (optional)

## Usage

### Configuration

The server uses a `.env` file in the project root for all configuration. This centralized approach makes it easy to configure once and run from anywhere.

```bash
# Copy the example file
cp .env.example .env

# Edit the file with your preferred text editor
nano .env
```

Example `.env` file content:
```
# Project directory to index (required)
CODE_INDEX_DIRECTORY=/Users/username/my-project

# For embeddings (required)
OPEN_AI_KEY=sk-your-openai-api-key

# For Qdrant Cloud (optional)
QDRANT_URL=https://your-cluster-url.qdrant.tech
QDRANT_API_KEY=your_qdrant_api_key

# Auto-indexing (optional)
CODE_INDEX_AUTO_INDEX=true
```

### Running the Server

The server can be run in multiple ways:

**Using the CLI from the scripts directory:**
```bash
# Using the directory specified in .env
./scripts/start_indexer.sh

# Using Qdrant Cloud
./scripts/start_indexer.sh --use-qdrant-cloud

# With auto-indexing enabled (automatically update index when files change)
./scripts/start_indexer.sh --auto-index
```

For Windows:
```powershell
# Using directory from .env
.\scripts\start_indexer.ps1

# Using Qdrant Cloud
.\scripts\start_indexer.ps1 --use-qdrant-cloud
```

**Using the Python module directly:**
```bash
# Start the server using CODE_INDEX_DIRECTORY from .env
python -m code_index_mcp.cli.commands server

# Index the directory from .env without starting a server
python -m code_index_mcp.cli.commands index
```

The server will:
1. Index all supported files in the specified directory (from .env)
2. Generate embeddings for each file
3. Store the embeddings in Qdrant (local or cloud)
4. Monitor for file changes and update the index in real-time
5. Provide semantic search capabilities for the indexed code

### Using the Web Interface

Code Index MCP includes a Streamlit web interface for easy management:

```bash
# Start the web interface (uses CODE_INDEX_DIRECTORY from .env)
python run_streamlit.py
```

Alternatively, you can run it directly as a module:

```bash
# Run as a module
python -m code_index_mcp.web
```

The web interface allows you to:
- View the indexing status of your codebase
- Manually trigger re-indexing
- Configure which directories to index
- Monitor changes in real-time

### Integrating with Claude Desktop

You can easily integrate Code Index MCP with Claude Desktop:

1. Ensure you have UV installed and the package is installed (see installation section above)
2. Configure your `.env` file with the `CODE_INDEX_DIRECTORY` setting
3. Find or create the Claude Desktop configuration file:

   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
   - macOS/Linux: `~/Library/Application Support/Claude/claude_desktop_config.json`
4. Add the following configuration:

   **For Windows**:

   ```json
   {
     "mcpServers": {
       "code-indexer": {
         "command": "python",
         "args": [
            "-m", "code_index_mcp.cli.commands", "server",
            "--auto-index"
          ]
       }
     }
   }
   ```

   **For macOS/Linux**:

   ```json
   {
     "mcpServers": {
       "code-indexer": {
         "command": "python",
         "args": [
            "-m", "code_index_mcp.cli.commands", "server",
            "--auto-index"
          ]
       }
     }
   }
   ```

   To use Qdrant Cloud, add an additional argument:
   ```json
   {
     "mcpServers": {
       "code-indexer": {
         "command": "python",
         "args": [
            "-m", "code_index_mcp.cli.commands", "server",
            "--auto-index",
            "--use-qdrant-cloud"
          ]
       }
     }
   }
   ```

5. Restart Claude Desktop to use Code Indexer for analyzing code projects

### Integrating with Cursor

You can also integrate Code Index MCP with Cursor:

1. Ensure you have UV installed (see installation section above)
2. Configure your `.env` file with the `CODE_INDEX_DIRECTORY` setting
3. Create a configuration file in one of these locations:

   **Project-specific configuration** (recommended):
   - Create a `.cursor/mcp.json` file in your project directory
   
   **Global configuration**:
   - Create a `~/.cursor/mcp.json` file in your home directory
   
4. Add the same configuration as for Claude Desktop

### Search Features

Once the server is running, you can use the following search capabilities:

1. **Semantic Search**:
   - Search for code using natural language queries
   - Example: "How do I handle authentication in this codebase?"

2. **Code-to-Code Similarity**:
   - Find code similar to a given code snippet
   - Example: "Find code similar to: `function calculateAverage(numbers) { ... }`"

3. **Grouped Search**:
   - Group search results by module to see related code across the codebase
   - Example: "Show me error handling across different modules"

4. **File Summaries**:
   - Get a summary of a file's structure, including functions, classes, and complexity metrics
   - Example: "Summarize the structure of app.py"

## Technical Details

### Vector Storage

All embeddings are stored in Qdrant, which provides:
- Efficient vector similarity search
- Filtering and grouping capabilities
- Cloud or local deployment options

### Chunking Strategy

The current implementation processes code at multiple levels:
- Whole files for context-aware search
- Individual functions and classes for precise search
- Methods within classes for detailed analysis
- Automatic chunking for large files

### Supported File Types

The following file types are currently supported for indexing and analysis:

- Python (.py)
- JavaScript/TypeScript (.js, .ts, .jsx, .tsx)
- Java (.java)
- C/C++ (.c, .cpp, .h, .hpp)
- C# (.cs)
- Go (.go)
- Ruby (.rb)
- PHP (.php)
- Swift (.swift)
- Kotlin (.kt)
- Rust (.rs)
- Scala (.scala)
- Shell (.sh, .bash)
- HTML/CSS (.html, .css, .scss)
- Markdown (.md)
- JSON (.json)
- XML (.xml)
- YAML (.yml, .yaml)

## Security Considerations

- File path validation prevents directory traversal attacks
- API keys are loaded from environment variables or .env file
- Qdrant Cloud connections use HTTPS
- The `.code_indexer` folder is included in `.gitignore`

## Troubleshooting

**Connection Issues with Qdrant Cloud**:
- Ensure your API key is correct
- Check that the URL is in the format `https://your-cluster-url.qdrant.tech`
- Verify that your network allows outbound HTTPS connections

**Slow Indexing**:
- Large codebases may take time to index
- Consider using a more powerful machine for indexing
- Try using fewer extensions to reduce the number of files indexed

**Missing Embeddings**:
- Ensure that the necessary API keys are set
- Check the logs for any errors during indexing
- Verify that the file extensions are supported

**Import Errors with Relative Imports**:
- If you see errors like `ImportError: attempted relative import with no known parent package`, make sure you're running the code as a module
- Use `python -m code_index_mcp.web` instead of directly running the script
- Alternatively, use the provided scripts in the `scripts` directory
- If all else fails, install the package in development mode: `uv pip install -e .`

## Contributing

Contributions via issues or pull requests to add new features or fix bugs are welcome.
