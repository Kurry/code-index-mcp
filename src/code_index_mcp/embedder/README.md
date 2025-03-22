# Dual-Embedding Document Processor

A powerful document embedding system using the OpenAI Agents SDK that combines text and code embeddings for enhanced semantic search capabilities.

## Features

- **Dual Embedding System**: Uses separate models for text and code embeddings (OpenAI, Sentence Transformers, etc.)
- **Text Processing**: Converts code to natural language for better text understanding with humanized identifiers
- **Code Structure Analysis**: Extracts functions, classes, methods, and other code elements with granular chunking
- **Reciprocal Rank Fusion**: Combines results from both embedding spaces for optimal ranking
- **Multiple Storage Backends**: Supports both file-based storage and Qdrant vector database
- **Input Validation Guardrails**: Prevents misuse and protects sensitive directories
- **Result Grouping**: Groups search results by file path, module name, or other metadata
- **Modular Architecture**: Clean separation of concerns with utility modules

## Installation

```bash
# Clone the repository
git clone [repository-url]
cd [repository-name]

# Install dependencies
pip install -r requirements.txt

# Optional: Install Qdrant for vector database support
pip install qdrant-client
```

## Usage

### Process a Directory

Process a codebase to extract code and documentation:

```python
from agents import Runner
from embedder.embedder import embedder_agent

# Process a directory
result = Runner.run(
    embedder_agent,
    "Process the './src' directory, excluding node_modules, with a max of 100 files"
)
```

### Generate Embeddings

Generate dual embeddings for the processed documents:

```python
# Generate embeddings
result = Runner.run(
    embedder_agent,
    "Generate embeddings using text-embedding-3-small for text and code, and save to 'embeddings.pkl'"
)

# Or use Qdrant storage
result = Runner.run(
    embedder_agent,
    "Generate embeddings and store in Qdrant collection 'code_search'"
)
```

### Search Embeddings

Search through generated embeddings:

```python
# Search with both text and code embeddings
result = Runner.run(
    embedder_agent,
    "Search for 'how to implement authentication' in embeddings.pkl with top-k=5"
)

# Group results by module
result = Runner.run(
    embedder_agent,
    "Search for 'database connection' in embeddings.pkl with top-k=10 grouped by module_name"
)
```

## How It Works

1. **Document Processing**: 
   - Scans directories for documents and extracts code structures
   - Identifies functions, classes, methods with precise start/end positions
   - Extracts keywords and builds metadata for enhanced search

2. **Dual Embedding Generation**: 
   - **Text Embeddings**: Code is converted to natural language text using the `textify_code` function
   - **Code Embeddings**: Original code is embedded using a code-specific model
   - Batched processing for efficiency with progress tracking

3. **Semantic Search**: 
   - Queries are embedded with both models
   - Similarities are calculated in both embedding spaces
   - Results are combined using Reciprocal Rank Fusion
   - Optional grouping by metadata fields for diverse results

4. **Storage Options**:
   - **File-based**: Traditional pickle file storage for simple use cases
   - **Qdrant**: Vector database integration for scalable, production-ready search

## Architecture

- `embedder.py`: Core agent with document processing and embedding functionality
- `embedding_models.py`: DualEmbeddingModel class managing text and code models
- `search.py`: Search functionality with Reciprocal Rank Fusion and result grouping
- `storage.py`: Storage backends including pickle files and Qdrant integration
- `text_processor.py`: Text normalization and code-to-text conversion utilities
- `schema.py`: Pydantic models for structured data representation
- `guardrails.py`: Input validation guardrails for query and directory validation

## Available Models

The system supports various embedding models:

- **Text Models**:
  - `text-embedding-3-small`: 1536 dimensions, good performance
  - `text-embedding-3-large`: 3072 dimensions, most powerful
  - `text-embedding-ada-002`: 1536 dimensions, legacy model

- **Code Models**:
  - `text-embedding-3-small`: Works well for code
  - `text-embedding-3-large`: Most powerful for code embedding
  - `text-embedding-ada-002`: Legacy model for code

## Using the Agent Directly with OpenAI Agents SDK

You can use the embedder agent directly in your Python code with the OpenAI Agents SDK:

```python
import asyncio
from agents import Runner
from embedder.embedder import embedder_agent

async def main():
    # Process a directory and generate embeddings
    result = await Runner.run(
        embedder_agent, 
        "Please process the 'src' directory and generate embeddings in 'output.pkl'"
    )
    print(result.final_output)
    
    # Search through embeddings
    result = await Runner.run(
        embedder_agent,
        "Search for 'implementing authentication' in the embeddings at 'output.pkl'"
    )
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
```

## Extending the Agent

Here are several ways to extend the embedder agent using the OpenAI Agents SDK capabilities:

### 1. Add Specialized Agent Handoffs

Create specialized agents for different document types and enable handoffs:

```python
from agents import Agent, handoff

# Create specialized agents
python_code_agent = Agent(
    name="Python Code Specialist",
    instructions="You analyze Python code structure and dependencies in detail.",
    tools=[process_directory, search_documents]
)

markdown_agent = Agent(
    name="Markdown Specialist",
    instructions="You analyze documentation files with focus on structure and examples.",
    tools=[process_directory, search_documents]
)

# Update the main agent with handoffs
enhanced_embedder_agent = embedder_agent.clone(
    handoffs=[
        python_code_agent,
        handoff(markdown_agent, tool_name_override="analyze_markdown")
    ]
)
```

### 2. Add Output Types for Structured Responses

Define structured output types for more predictable responses:

```python
from pydantic import BaseModel
from typing import List

class EmbeddingResult(BaseModel):
    total_files: int
    total_embeddings: int
    file_types: List[str]
    output_path: str
    stats: dict

# Clone the agent with the output type
structured_agent = embedder_agent.clone(
    output_type=EmbeddingResult
)
```

### 3. Add Streaming Capability

Use streaming for real-time progress updates during processing:

```python
import asyncio
from agents import Runner

async def process_with_streaming():
    result = Runner.run_streamed(
        embedder_agent,
        "Process the large 'project' directory with 500 files"
    )
    
    # Stream progress updates
    async for event in result.stream_events():
        if event.type == "run_item_stream_event":
            if event.item.type == "message_output_item":
                print(f"Progress update: {event.item.content}")
            elif event.item.type == "tool_call_item":
                print(f"Running: {event.item.name}")

asyncio.run(process_with_streaming())
```

### 4. Integrate with External Vector Databases

The system already supports Qdrant. To integrate with other databases:

```python
from agents import function_tool

@function_tool
def store_in_pinecone(embeddings_file: str, index_name: str) -> dict:
    """Store embeddings in Pinecone vector database.
    
    Args:
        embeddings_file: Path to the embeddings pickle file
        index_name: Name of the Pinecone index
        
    Returns:
        Storage statistics
    """
    # Implementation to load embeddings and store in Pinecone
    # ...
    
    return {"status": "success", "vectors_stored": 1000}

# Add the tool to the agent
extended_agent = embedder_agent.clone(
    tools=embedder_agent.tools + [store_in_pinecone]
)
```

### 5. Add Dynamic Instructions

Use context-aware instructions for the agent:

```python
def dynamic_instructions(context, agent):
    available_embedding_files = [f for f in os.listdir() if f.endswith('.pkl')]
    
    return f"""
    You are an expert document embedder agent that processes codebases to generate searchable embeddings.
    
    Currently available embedding files: {', '.join(available_embedding_files)}
    
    Your main functions are:
    - Process a directory to extract code and documentation
    - Generate dual embeddings for the processed documents
    - Search for similar documents using natural language or code snippets
    
    Current system time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """

# Update the agent with dynamic instructions
dynamic_agent = embedder_agent.clone(
    instructions=dynamic_instructions
)
``` 