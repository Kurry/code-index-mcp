"""
Code Index MCP Server

This MCP server allows LLMs to index, search, and analyze code from a project directory.
It provides semantic search with dual embeddings using the OpenAI Agents SDK's document processor.
"""
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Dict, List, Optional, Tuple, Any
import os
import pathlib
import json
import fnmatch
import sys
import argparse
from mcp.server.fastmcp import FastMCP, Context, Image
from mcp import types
from watchdog.observers import Observer
from dotenv import load_dotenv

# Import the embedding manager and file change handler
from ..core.embedding_manager import EmbeddingManager
from ..core.watcher import FileChangeHandler

# Create the MCP server
mcp = FastMCP("CodeIndexer", dependencies=["pathlib"])

# Global variables
_embedding_manager = None

# List of supported file extensions
supported_extensions = [
    '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.h', '.hpp',
    '.cs', '.go', '.rb', '.php', '.swift', '.kt', '.rs', '.scala', '.sh',
    '.bash', '.html', '.css', '.scss', '.md', '.json', '.xml', '.yml', '.yaml'
]

@dataclass
class CodeIndexerContext:
    """Context for the Code Indexer MCP server."""
    base_path: str
    file_count: int = 0
    observer: Optional[Observer] = None

@asynccontextmanager
async def indexer_lifespan(server: FastMCP) -> AsyncIterator[CodeIndexerContext]:
    """Manage the lifecycle of the Code Indexer MCP server."""
    global _embedding_manager
    
    # Get arguments from command line
    args = server.app_args if hasattr(server, 'app_args') else None
    
    # Determine project directory
    project_dir = ""
    if args and hasattr(args, 'directory') and args.directory:
        project_dir = args.directory
    else:
        # Try to find the project directory
        # First, check for common project files in the current directory
        current_dir = os.getcwd()
        project_markers = [
            'package.json', 'setup.py', 'pyproject.toml', 'Cargo.toml',
            'pom.xml', 'build.gradle', 'Makefile'
        ]
        
        if any(os.path.exists(os.path.join(current_dir, marker)) for marker in project_markers):
            project_dir = current_dir
        else:
            # Look for git root
            git_dir = os.path.join(current_dir, '.git')
            if os.path.exists(git_dir) and os.path.isdir(git_dir):
                project_dir = current_dir
            else:
                # Try parent directory
                parent_dir = os.path.dirname(current_dir)
                if any(os.path.exists(os.path.join(parent_dir, marker)) for marker in project_markers):
                    project_dir = parent_dir
                else:
                    # Default to current directory with a warning
                    project_dir = current_dir
                    print("Warning: Could not detect project directory. Using current directory.", file=sys.stderr)
    
    # Get supported extensions
    supported_exts = []
    if args and hasattr(args, 'extensions') and args.extensions:
        supported_exts = [e if e.startswith('.') else f'.{e}' for e in args.extensions]
    
    if not supported_exts:
        supported_exts = supported_extensions
    
    # Check if directory exists
    if not os.path.exists(project_dir) or not os.path.isdir(project_dir):
        print(f"Error: Project directory {project_dir} does not exist", file=sys.stderr)
        project_dir = ""
    
    # Initialize the context
    context = CodeIndexerContext(
        base_path=project_dir,
        file_count=0
    )
    
    # Initialize the embedding manager if we have a valid project directory
    if project_dir:
        print(f"Initializing embedding manager with project directory: {project_dir}", file=sys.stderr)
        
        # Determine if we should use Qdrant Cloud
        use_qdrant_cloud = False
        if args and hasattr(args, 'use_qdrant_cloud'):
            use_qdrant_cloud = args.use_qdrant_cloud
        
        # Initialize the embedding manager
        _embedding_manager = EmbeddingManager(
            base_path=project_dir, 
            supported_extensions=supported_exts,
            use_qdrant_cloud=use_qdrant_cloud
        )
        
        # Check if auto-indexing is enabled
        auto_index = False
        if args and hasattr(args, 'auto_index'):
            auto_index = args.auto_index
            
        if auto_index:
            # Create file change handler
            handler = FileChangeHandler(_embedding_manager)
            observer = Observer()
            observer.schedule(handler, project_dir, recursive=True)
            observer.start()
            
            # Store observer in context
            context.observer = observer
            
            print("Auto-indexing enabled. Watching for file changes...", file=sys.stderr)
        
        # Index the directory
        print("Indexing project directory...", file=sys.stderr)
        file_count = _embedding_manager.index_directory()
        context.file_count = file_count
        print(f"Indexed {file_count} files", file=sys.stderr)
    
    try:
        # Yield the context to the server
        yield context
    finally:
        # Stop the file observer if it exists
        if context.observer and context.observer.is_alive():
            context.observer.stop()
            context.observer.join()

# Initialize the server with our lifespan manager
mcp = FastMCP("CodeIndexer", lifespan=indexer_lifespan)

# ----- RESOURCES -----

@mcp.resource("config://code-indexer")
def get_config() -> str:
    """Get the current configuration of the Code Indexer."""
    ctx = mcp.get_context()
    
    # Get the base path from context
    base_path = ctx.request_context.lifespan_context.base_path
    
    # Check if base_path is set
    if not base_path:
        return json.dumps({
            "status": "not_configured",
            "message": "Project path not set. You must provide a directory at startup.",
            "supported_extensions": supported_extensions
        }, indent=2)
    
    # Get collection info if available
    collection_stats = {}
    if _embedding_manager:
        collection_stats = _embedding_manager.get_collection_stats()
    
    config = {
        "base_path": base_path,
        "supported_extensions": supported_extensions,
        "file_count": ctx.request_context.lifespan_context.file_count,
        "embedding_provider": {
            "text_model": _embedding_manager.text_model if _embedding_manager else "text-embedding-3-small",
            "code_model": _embedding_manager.code_model if _embedding_manager else "text-embedding-3-small",
        },
        "collection_stats": collection_stats
    }
    
    return json.dumps(config, indent=2)

@mcp.resource("files://{file_path}")
def get_file_content(file_path: str) -> str:
    """Get the content of a specific file."""
    ctx = mcp.get_context()
    
    # Get the base path from context
    base_path = ctx.request_context.lifespan_context.base_path
    
    # Check if base_path is set
    if not base_path:
        return "Error: Project path not set. You must provide a directory at startup."
    
    # Handle absolute paths (especially Windows paths starting with drive letters)
    if os.path.isabs(file_path) or (len(file_path) > 1 and file_path[1] == ':'):
        # Absolute paths are not allowed via this endpoint
        return f"Error: Absolute file paths like '{file_path}' are not allowed. Please use paths relative to the project root."
    
    # Normalize the file path
    norm_path = os.path.normpath(file_path)
    
    # Check for path traversal attempts
    if "..\\" in norm_path or "../" in norm_path or norm_path.startswith(".."): 
        return f"Error: Invalid file path: {file_path} (directory traversal not allowed)"
    
    # Construct the full path and verify it's within the project bounds
    full_path = os.path.join(base_path, norm_path)
    real_full_path = os.path.realpath(full_path)
    real_base_path = os.path.realpath(base_path)
    
    if not real_full_path.startswith(real_base_path):
        return f"Error: Access denied. File path must be within project directory."
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return content
    except UnicodeDecodeError:
        return f"Error: File {file_path} appears to be a binary file or uses unsupported encoding."
    except Exception as e:
        return f"Error reading file: {e}"

@mcp.resource("structure://project")
def get_project_structure() -> str:
    """Get the structure of the project as a JSON tree."""
    ctx = mcp.get_context()
    
    # Get the base path from context
    base_path = ctx.request_context.lifespan_context.base_path
    
    # Check if base_path is set
    if not base_path:
        return json.dumps({
            "status": "not_configured",
            "message": "Project path not set. You must provide a directory at startup."
        }, indent=2)
    
    # Build a simple directory tree
    tree = {}
    _build_directory_tree(base_path, tree)
    
    return json.dumps(tree, indent=2)

# ----- TOOLS -----

@mcp.tool()
def semantic_search(query: str, ctx: Context, limit: int = 5, use_model: str = "both") -> Dict[str, Any]:
    """
    Search for code semantically using embeddings.
    
    Args:
        query: The search query
        limit: Maximum number of results to return
        use_model: Which model to use - "text", "code", or "both"
        
    Returns:
        Dictionary with search results
    """
    global _embedding_manager
    
    if not _embedding_manager:
        return {"error": "Embedding manager not initialized. Server was not started with a directory."}
    
    if use_model not in ["text", "code", "both"]:
        return {"error": f"Invalid model type: {use_model}. Must be 'text', 'code', or 'both'."}
    
    try:
        results = _embedding_manager.semantic_search(query, limit=limit, use_model=use_model)
        
        return {
            "query": query,
            "model": use_model,
            "results": results
        }
    except Exception as e:
        return {"error": f"Error during semantic search: {e}"}

@mcp.tool()
def find_files(pattern: str, ctx: Context) -> List[str]:
    """
    Find files in the project that match the given pattern.
    Supports glob patterns like *.py or **/*.js.
    """
    base_path = ctx.request_context.lifespan_context.base_path
    
    # Check if base_path is set
    if not base_path:
        return ["Error: Project path not set. You must provide a directory at startup."]
    
    matching_files = []
    
    # Find all files matching the pattern
    for root, _, files in os.walk(base_path):
        # Skip hidden directories and common build/dependency directories
        if any(d.startswith('.') for d in pathlib.Path(root).parts) or \
           any(d in ['node_modules', 'venv', '__pycache__', 'build', 'dist'] 
               for d in pathlib.Path(root).parts):
            continue
        
        for file in files:
            # Skip hidden files
            if file.startswith('.'):
                continue
                
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, base_path)
            
            if fnmatch.fnmatch(rel_path, pattern):
                matching_files.append(rel_path.replace('\\', '/'))
    
    return matching_files

@mcp.tool()
def get_file_summary(file_path: str, ctx: Context) -> Dict[str, Any]:
    """
    Get a summary of a specific file, including:
    - Line count
    - Function/class definitions (for supported languages)
    - Import statements
    - Basic complexity metrics
    """
    base_path = ctx.request_context.lifespan_context.base_path
    
    # Check if base_path is set
    if not base_path:
        return {"error": "Project path not set. You must provide a directory at startup."}
    
    # Normalize the file path
    norm_path = os.path.normpath(file_path)
    
    # Check for path traversal attempts
    if "..\\" in norm_path or "../" in norm_path or norm_path.startswith(".."):
        return {"error": f"Invalid file path: {file_path} (directory traversal not allowed)"}
    
    # Construct the full path and verify it's within the project bounds
    full_path = os.path.join(base_path, norm_path)
    real_full_path = os.path.realpath(full_path)
    real_base_path = os.path.realpath(base_path)
    
    if not real_full_path.startswith(real_base_path):
        return {"error": "Access denied. File path must be within project directory."}
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Get basic file info
        lines = content.splitlines()
        line_count = len(lines)
        _, ext = os.path.splitext(full_path)
        
        summary = {
            "file_path": file_path,
            "line_count": line_count,
            "blank_lines": sum(1 for line in lines if not line.strip()),
            "extension": ext,
        }
        
        # Language-specific analysis
        if ext.lower() in ['.py']:
            # Python analysis
            import_lines = [line for line in lines if line.strip().startswith('import ') or line.strip().startswith('from ') and ' import ' in line]
            functions = [line for line in lines if line.strip().startswith('def ')]
            classes = [line for line in lines if line.strip().startswith('class ')]
            
            # Extract function and class names
            function_names = [line.strip().split('def ')[1].split('(')[0].strip() for line in functions]
            class_names = [line.strip().split('class ')[1].split('(')[0].strip(':').strip() for line in classes]
            
            summary.update({
                "language": "Python",
                "imports": import_lines,
                "functions": function_names,
                "classes": class_names,
                "complexity": {
                    "function_count": len(function_names),
                    "class_count": len(class_names),
                    "import_count": len(import_lines),
                }
            })
            
        elif ext.lower() in ['.js', '.jsx', '.ts', '.tsx']:
            # JavaScript/TypeScript analysis
            import_lines = [line for line in lines if 'import ' in line]
            function_lines = [line for line in lines if 'function ' in line or '=>' in line]
            class_lines = [line for line in lines if line.strip().startswith('class ')]
            
            summary.update({
                "language": "JavaScript/TypeScript",
                "imports": import_lines,
                "functions": function_lines,
                "classes": class_lines,
                "complexity": {
                    "function_count": len(function_lines),
                    "class_count": len(class_lines),
                    "import_count": len(import_lines),
                }
            })
            
        # Use embedder's code structure extraction if available
        if ext.lower() in ['.py', '.js', '.jsx', '.ts', '.tsx']:
            try:
                from ..embedder.text_processor import extract_code_structure, extract_keywords_from_code
                
                # Extract code structure
                structure = extract_code_structure(content)
                summary["code_structure"] = {
                    "functions": [func["name"] for func in structure.get("functions", [])],
                    "classes": [cls["name"] for cls in structure.get("classes", [])]
                }
                
                # Extract keywords
                keywords = extract_keywords_from_code(content, max_keywords=15)
                summary["keywords"] = keywords
                
            except ImportError:
                # Continue without structure extraction if embedder is not available
                pass
        
        return summary
        
    except UnicodeDecodeError:
        return {"error": f"File {file_path} appears to be a binary file or uses unsupported encoding."}
    except Exception as e:
        return {"error": f"Error analyzing file: {e}"}

@mcp.tool()
def refresh_embeddings(ctx: Context) -> str:
    """
    Refresh the code embeddings.
    This will re-index all files in the project directory.
    """
    global _embedding_manager
    
    base_path = ctx.request_context.lifespan_context.base_path
    
    # Check if base_path is set
    if not base_path:
        return "Error: Project path not set. You must provide a directory at startup."
    
    if not _embedding_manager:
        return "Error: Embedding manager not initialized."
    
    try:
        count = _embedding_manager.index_directory()
        
        # Update the file count in the context
        ctx.request_context.lifespan_context.file_count = count
        
        return f"Successfully refreshed embeddings for {count} files."
    except Exception as e:
        return f"Error refreshing embeddings: {e}"

@mcp.tool()
def search_by_module(query: str, ctx: Context, limit: int = 5) -> Dict[str, Any]:
    """
    Search for code semantically and group results by module.
    
    Args:
        query: The search query
        limit: Maximum number of results to return per module
        
    Returns:
        Dictionary with search results grouped by module
    """
    global _embedding_manager
    
    if not _embedding_manager:
        return {"error": "Embedding manager not initialized. Server was not started with a directory."}
    
    try:
        # Perform semantic search with module grouping
        # This uses the embedder's group_by feature
        results = _embedding_manager.semantic_search(
            query=query, 
            limit=limit*3,  # Get more results so we have enough after grouping
            use_model="both"
        )
        
        # Group results by module if possible
        modules = {}
        for result in results:
            module = result.get("metadata", {}).get("module_name", "unknown")
            if module not in modules:
                modules[module] = []
            
            if len(modules[module]) < limit:
                modules[module].append(result)
        
        # Format the response
        formatted_modules = {}
        for module_name, module_results in modules.items():
            formatted_modules[module_name] = [
                {
                    "name": r.get("name", ""),
                    "file_path": r.get("file_path", ""),
                    "score": r.get("score", 0.0),
                    "snippet": r.get("snippet", ""),
                }
                for r in module_results
            ]
        
        return {
            "query": query,
            "modules": formatted_modules,
            "total_modules": len(formatted_modules),
            "total_results": sum(len(results) for results in formatted_modules.values())
        }
    except Exception as e:
        return {"error": f"Error during module-based search: {e}"}

@mcp.prompt()
def analyze_code(file_path: str = "", query: str = "") -> list[types.PromptMessage]:
    """Analyze code semantically and in context"""
    if file_path and not query:
        return [
            {"role": "system", "content": "You are a code analyzer that provides clear explanations about code. Focus on explaining components, patterns, and design decisions."},
            {"role": "user", "content": f"Analyze the code in the file {file_path} and explain what it does, how it works, and any important patterns or design decisions."}
        ]
    elif query:
        return [
            {"role": "system", "content": "You are a code analyzer that provides clear explanations about code. Focus on explaining components, patterns, and design decisions."},
            {"role": "user", "content": f"Analyze the codebase to answer this question: {query}"}
        ]
    else:
        return [
            {"role": "system", "content": "You are a code analyzer that provides clear explanations about code."},
            {"role": "user", "content": "To analyze code, provide either a file_path or a query."}
        ]

@mcp.prompt()
def code_search(query: str = "") -> types.TextContent:
    """Find and explain relevant code for a given query"""
    if not query:
        return types.TextContent("Please provide a query to search for code.")
    
    return types.TextContent(f"""
I'll search for code related to "{query}".

First, I'll perform a semantic search to find the most relevant code snippets, then explain how they work and how they relate to your query.
""")

def _build_directory_tree(path, tree, rel_path=""):
    """Build a directory tree recursively."""
    if os.path.isdir(path):
        # Skip hidden directories and common excluded directories
        dir_name = os.path.basename(path)
        if dir_name.startswith('.') or dir_name in ['node_modules', 'venv', '__pycache__', 'build', 'dist']:
            return
        
        # Process files in this directory
        files = []
        dirs = {}
        
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            item_rel_path = os.path.join(rel_path, item)
            
            if os.path.isfile(item_path) and not item.startswith('.'):
                files.append(item)
            elif os.path.isdir(item_path):
                # Process subdirectory
                subdir = {}
                _build_directory_tree(item_path, subdir, item_rel_path)
                
                if subdir:  # Only add non-empty directories
                    dirs[item] = subdir
        
        # Add files and directories to the tree
        tree["files"] = files
        tree["dirs"] = dirs
    
    return tree

def main():
    """Run the Code Indexer MCP server."""
    import os
    
    # Get configuration from environment variables
    directory = os.environ.get('CODE_INDEX_DIRECTORY', os.getcwd())
    port = int(os.environ.get('CODE_INDEX_PORT', '8000'))
    host = os.environ.get('CODE_INDEX_HOST', '127.0.0.1')
    reload = os.environ.get('CODE_INDEX_RELOAD', '').lower() in ('true', '1', 'yes')
    use_qdrant_cloud = os.environ.get('CODE_INDEX_USE_QDRANT_CLOUD', '').lower() in ('true', '1', 'yes')
    auto_index = os.environ.get('CODE_INDEX_AUTO_INDEX', '').lower() in ('true', '1', 'yes')
    
    # Get extensions from environment variable (comma-separated list)
    extensions_str = os.environ.get('CODE_INDEX_EXTENSIONS', '')
    extensions = [ext.strip() for ext in extensions_str.split(',')] if extensions_str else []
    
    # Create a new FastMCP instance with the lifespan
    server = FastMCP("CodeIndexer", lifespan=indexer_lifespan)
    
    # Store the configuration in the server instance so the lifespan can access it
    server.app_args = type('Args', (), {
        'directory': directory,
        'port': port,
        'host': host,
        'reload': reload,
        'use_qdrant_cloud': use_qdrant_cloud,
        'extensions': extensions,
        'auto_index': auto_index
    })
    
    # Run the server - FastMCP doesn't accept port or reload arguments directly
    server.run()

if __name__ == "__main__":
    main()

