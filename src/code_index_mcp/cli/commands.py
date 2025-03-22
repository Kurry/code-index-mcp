"""
CLI commands for Code Index MCP
"""
import os
import argparse
import sys
from pathlib import Path

from ..server.server import main as server_main

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Code Index MCP CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Run the Code Index MCP server")
    server_parser.add_argument("--directory", "-d", type=str, help="Project directory to index")
    server_parser.add_argument("--port", "-p", type=int, default=8000, help="Port to run the server on")
    server_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the server on")
    server_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    server_parser.add_argument("--use-qdrant-cloud", action="store_true", help="Use Qdrant Cloud instead of local instance")
    server_parser.add_argument("--extensions", type=str, nargs="+", help="File extensions to index")
    server_parser.add_argument("--auto-index", action="store_true", help="Enable auto-indexing")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index a directory")
    index_parser.add_argument("directory", type=str, help="Project directory to index")
    index_parser.add_argument("--extensions", type=str, nargs="+", help="File extensions to index")
    index_parser.add_argument("--use-qdrant-cloud", action="store_true", help="Use Qdrant Cloud instead of local instance")
    
    return parser.parse_args()

def main():
    """Main entry point for the CLI."""
    args = parse_arguments()
    
    if args.command == "server":
        # Configure environment variables
        if args.directory:
            os.environ["CODE_INDEX_DIRECTORY"] = args.directory
        os.environ["CODE_INDEX_PORT"] = str(args.port)
        os.environ["CODE_INDEX_HOST"] = args.host
        os.environ["CODE_INDEX_RELOAD"] = "1" if args.reload else "0"
        os.environ["CODE_INDEX_USE_QDRANT_CLOUD"] = "1" if args.use_qdrant_cloud else "0"
        os.environ["CODE_INDEX_AUTO_INDEX"] = "1" if args.auto_index else "0"
        
        if args.extensions:
            os.environ["CODE_INDEX_EXTENSIONS"] = ",".join(args.extensions)
        
        # Run the server
        server_main()
    elif args.command == "index":
        from ..core.embedding_manager import EmbeddingManager
        
        # Configure embedding manager
        extensions = args.extensions if args.extensions else None
        manager = EmbeddingManager(
            base_path=args.directory,
            supported_extensions=extensions,
            use_qdrant_cloud=args.use_qdrant_cloud
        )
        
        # Index the directory
        print(f"Indexing directory {args.directory}")
        file_count = manager.index_directory()
        print(f"Indexed {file_count} files")
    else:
        print("Please specify a command. Use --help for more information.")
        sys.exit(1)

if __name__ == "__main__":
    main()
