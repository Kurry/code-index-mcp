#!/usr/bin/env python3
"""
Main module for running the Code Index MCP Streamlit web interface.
"""
import os
import sys
import argparse
import streamlit.web.cli as stcli
from pathlib import Path

def main():
    """Run the Streamlit web interface for Code Index MCP."""
    parser = argparse.ArgumentParser(description='Code Index MCP Streamlit Web Interface')
    parser.add_argument('--port', '-p', type=int, default=8501, help='Port to run the Streamlit server on')
    
    args = parser.parse_args()
    
    # Get the path to the streamlit_app.py file
    app_path = Path(__file__).parent / "streamlit_app.py"
    
    # Print startup information
    print(f"Starting Code Index MCP Streamlit web interface on port {args.port}...")
    print("Project directory will be loaded from .env file or environment variables.")
    
    # Set up Streamlit arguments
    streamlit_args = [
        "--", 
        str(app_path),
        "--server.port", str(args.port),
        "--theme.base", "dark",
        "--browser.gatherUsageStats", "false",
        "--server.headless", "true"
    ]
    
    # Run Streamlit directly
    sys.argv = ["streamlit", "run"] + streamlit_args
    stcli.main()

if __name__ == "__main__":
    main()
