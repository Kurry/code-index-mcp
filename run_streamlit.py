#!/usr/bin/env python3
"""
Run script for the Code Index MCP Streamlit web interface.
"""
import os
import sys
import argparse
from pathlib import Path

# Add the src directory to the Python path
src_path = str(Path(__file__).parent / "src")
sys.path.insert(0, src_path)

def main():
    """Run the Streamlit web interface for Code Index MCP."""
    parser = argparse.ArgumentParser(description='Code Index MCP Streamlit Web Interface')
    parser.add_argument('--port', '-p', type=int, default=8501, help='Port to run the Streamlit server on')
    
    args = parser.parse_args()
    
    # Command to run the Streamlit app
    streamlit_cmd = [
        "streamlit", "run", 
        str(Path(__file__).parent / "src" / "code_index_mcp" / "web" / "streamlit_app.py"),
        "--server.port", str(args.port),
        "--theme.base", "dark",
        "--browser.gatherUsageStats", "false",
        "--server.headless", "true"
    ]
    
    print(f"Starting Code Index MCP Streamlit web interface on port {args.port}...")
    print("Project directory will be loaded from .env file or environment variables.")
    
    # Run the Streamlit command
    os.system(" ".join(streamlit_cmd))

if __name__ == "__main__":
    main()
