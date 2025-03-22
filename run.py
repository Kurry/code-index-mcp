#!/usr/bin/env python
"""
Development convenience script to run the Code Index MCP server with embeddings.
"""
import sys
import os
import traceback

# Add src directory to path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, src_path)

try:
    from code_index_mcp.cli.commands import main
    
    if __name__ == "__main__":
        print("Starting Code Index MCP server with Qdrant embeddings...", file=sys.stderr)
        print(f"Added path: {src_path}", file=sys.stderr)
        
        # Pass through all arguments to the main function
        main()
except ImportError as e:
    print(f"Import Error: {e}", file=sys.stderr)
    print(f"Current sys.path: {sys.path}", file=sys.stderr)
    print("Traceback:", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
except Exception as e:
    print(f"Error starting server: {e}", file=sys.stderr)
    print("Traceback:", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
