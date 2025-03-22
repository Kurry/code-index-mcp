#!/usr/bin/env python3
"""
Generate Embeddings Script

This script uses the embedding agent to generate embeddings for a directory of code.
It allows specifying input and output directories, max files, and dry run mode.
"""
import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv

# Add the src directory to the Python path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, src_path)

# Import embedder components - we use relative imports to ensure it works
try:
    from code_index_mcp.embedder import process_directory, generate_embeddings
    from code_index_mcp.embedder.schema import ProcessingStats
except ImportError as e:
    print(f"Error importing embedding modules: {e}")
    print("Make sure you've installed the package with 'uv pip install -e .'")
    sys.exit(1)

# Default extensions to process if none are specified
DEFAULT_EXTENSIONS = [
    "py", "md", "rst", "txt", "ipynb", "js", "ts", "java", "c", "cpp", "h", 
    "cs", "go", "rb", "php", "swift", "kt", "rs", "scala", "sh", "bash", 
    "html", "css", "scss", "json", "xml", "yml", "yaml"
]

# Default directories to exclude
DEFAULT_EXCLUDE_DIRS = [
    "__pycache__", ".git", ".github", ".vscode", "venv", "env", ".env",
    ".venv", "node_modules", "dist", "build"
]

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings for a directory of code"
    )
    
    # Required arguments
    parser.add_argument(
        "--input", "-i", 
        required=True,
        help="Directory containing code to process"
    )
    
    parser.add_argument(
        "--output", "-o", 
        required=True,
        help="Directory to save embedding files"
    )
    
    # Optional arguments
    parser.add_argument(
        "--max-files", "-m", 
        type=int, 
        default=1000,
        help="Maximum number of files to process (default: 1000)"
    )
    
    parser.add_argument(
        "--extensions", "-e", 
        nargs="+", 
        default=DEFAULT_EXTENSIONS,
        help="File extensions to process (default: most common code file types)"
    )
    
    parser.add_argument(
        "--exclude-dirs", "-x", 
        nargs="+", 
        default=DEFAULT_EXCLUDE_DIRS,
        help="Directories to exclude from processing"
    )
    
    parser.add_argument(
        "--chunk-size", "-c", 
        type=int, 
        default=500,
        help="Size of code chunks for granular embedding (default: 500)"
    )
    
    parser.add_argument(
        "--dry-run", "-d", 
        action="store_true",
        help="Run in dry-run mode (don't actually generate embeddings)"
    )
    
    parser.add_argument(
        "--text-model", "-t", 
        default="text-embedding-3-small",
        help="Text embedding model to use (default: text-embedding-3-small)"
    )
    
    parser.add_argument(
        "--code-model", "-cm", 
        default="text-embedding-3-small",
        help="Code embedding model to use (default: text-embedding-3-small)"
    )
    
    parser.add_argument(
        "--use-qdrant-cloud", "-q", 
        action="store_true",
        help="Use Qdrant Cloud instead of local storage"
    )
    
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Enable verbose output"
    )

    return parser.parse_args()

def main():
    """Main entry point for the script."""
    # Parse arguments
    args = parse_args()
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Check for OpenAI API key
    openai_key = os.environ.get("OPEN_AI_KEY")
    if not openai_key:
        print("ERROR: OPEN_AI_KEY environment variable not set.")
        print("Please set it in your .env file or environment.")
        sys.exit(1)
    
    # Check if input directory exists
    input_dir = Path(args.input)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"ERROR: Input directory '{args.input}' does not exist or is not a directory.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp-based embedding filename
    timestamp = int(time.time())
    embedding_file = output_dir / f"embeddings_{timestamp}.pkl"
    
    # Print configuration
    print("Embedding Generation Configuration:")
    print(f"  Input directory:  {input_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Output file:      {embedding_file}")
    print(f"  Max files:        {args.max_files}")
    print(f"  File extensions:  {', '.join(args.extensions)}")
    print(f"  Chunk size:       {args.chunk_size}")
    print(f"  Text model:       {args.text_model}")
    print(f"  Code model:       {args.code_model}")
    print(f"  Dry run:          {args.dry_run}")
    print(f"  Using Qdrant:     {args.use_qdrant_cloud}")
    print()
    
    # Process directory
    print("Processing directory...")
    start_time = time.time()
    
    try:
        # Process the directory and get statistics
        process_args = {
            "directory": str(input_dir),
            "exclude_dirs": args.exclude_dirs,
            "file_extensions": args.extensions,
            "max_files": args.max_files,
            "chunk_size": args.chunk_size
        }
        
        if args.verbose:
            print(f"Process arguments: {process_args}")
        
        stats = process_directory(**process_args)
        
        elapsed_time = time.time() - start_time
        print(f"Directory processing completed in {elapsed_time:.2f} seconds")
        
        # Print statistics
        files_found = stats.get("total_files_found", 0) 
        files_processed = stats.get("files_processed", 0)
        
        print(f"Found {files_found} files, processed {files_processed} files")
        
        if "file_types" in stats:
            print("File types processed:")
            for ext, count in stats["file_types"].items():
                print(f"  .{ext}: {count} files")
        
        print(f"Functions found: {stats.get('functions_found', 0)}")
        print(f"Classes found: {stats.get('classes_found', 0)}")
        print(f"Methods found: {stats.get('methods_found', 0)}")
        print(f"Code chunks: {stats.get('code_chunks', 0)}")
        
        # If dry run, stop here
        if args.dry_run:
            print("\nDRY RUN: Skipping embedding generation")
            return
        
        # Generate embeddings
        print("\nGenerating embeddings...")
        emb_start_time = time.time()
        
        embedding_args = {
            "output_file": str(embedding_file),
            "text_model": args.text_model,
            "code_model": args.code_model,
            "batch_size": 32,  # Default batch size
            "storage_type": "qdrant" if args.use_qdrant_cloud else "pickle"
        }
        
        if args.verbose:
            print(f"Embedding arguments: {embedding_args}")
        
        embedding_result = generate_embeddings(**embedding_args)
        
        emb_elapsed_time = time.time() - emb_start_time
        print(f"Embedding generation completed in {emb_elapsed_time:.2f} seconds")
        
        # Print embedding statistics
        print(f"\nEmbedding Statistics:")
        print(f"Total documents: {embedding_result.get('total_documents', 0)}")
        print(f"Text embedding model: {embedding_result.get('text_embedding_model', '')}")
        print(f"Code embedding model: {embedding_result.get('code_embedding_model', '')}")
        print(f"Text vector dimension: {embedding_result.get('text_vector_dimension', 0)}")
        print(f"Code vector dimension: {embedding_result.get('code_vector_dimension', 0)}")
        
        # Print success message
        total_time = time.time() - start_time
        if args.use_qdrant_cloud:
            print(f"\nSuccess! Embeddings stored in Qdrant Cloud.")
        else:
            print(f"\nSuccess! Embeddings stored in {embedding_file}")
        print(f"Total processing time: {total_time:.2f} seconds")
        
    except Exception as e:
        print(f"ERROR: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
