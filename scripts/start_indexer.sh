#!/bin/bash
# Script to start the Code Index MCP server with embeddings

set -e # Exit on error

# Initialize variables
USE_QDRANT_CLOUD=false
PROJECT_DIR=""

# Check for directory argument
if [ $# -gt 0 ] && [[ ! "$1" =~ ^- ]]; then
    # First arg is not a flag, treat as directory
    PROJECT_DIR="$1"
    shift
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --use-qdrant-cloud)
            USE_QDRANT_CLOUD=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [/path/to/project] [--use-qdrant-cloud]"
            exit 1
            ;;
    esac
done

# Get the full path of the code-index-mcp repository
REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check for a .env file
ENV_FILE="$REPO_PATH/.env"
if [ -f "$ENV_FILE" ]; then
    echo "💼 Found .env file, will use environment variables from there"
else
    echo "ℹ️ No .env file found, checking for environment variables directly"
    
    # Check for API keys and Qdrant configuration
    if [ "$USE_QDRANT_CLOUD" = true ]; then
        if [ -z "$QDRANT_API_KEY" ]; then
            echo "⚠️ Warning: QDRANT_API_KEY environment variable not set"
            echo "You should create a .env file based on .env.example or set the environment variable"
        fi
        
        if [ -z "$QDRANT_URL" ]; then
            echo "⚠️ Warning: QDRANT_URL environment variable not set"
            echo "You should create a .env file based on .env.example or set the environment variable"
        fi
    fi
    
    # Check for OpenAI API key
    if [ -z "$OPEN_AI_KEY" ]; then
        echo "⚠️ Warning: OPEN_AI_KEY environment variable not set"
        echo "You should create a .env file based on .env.example or set the environment variable"
    fi
    
    # Check for project directory if not specified as argument
    if [ -z "$PROJECT_DIR" ] && [ -z "$PROJECT_DIRECTORY" ]; then
        echo "⚠️ Warning: No project directory specified and PROJECT_DIRECTORY environment variable not set"
        echo "You should create a .env file based on .env.example or specify the directory as an argument"
    fi
fi

echo "Starting Code Index MCP server..."

# If directory was passed as argument, use it
if [ -n "$PROJECT_DIR" ]; then
    echo "Project directory from argument: $PROJECT_DIR"
else
    echo "Using project directory from .env or environment variables"
fi

echo "Using Qdrant Cloud: $USE_QDRANT_CLOUD"

# Build the command
CMD="uv run run.py"

# Add project directory if specified as argument
if [ -n "$PROJECT_DIR" ]; then
    CMD="$CMD \"$PROJECT_DIR\""
fi

# Add Qdrant Cloud flag if needed
if [ "$USE_QDRANT_CLOUD" = true ]; then
    CMD="$CMD --use-qdrant-cloud"
fi

# Run the server
cd "$REPO_PATH"
echo "Executing: $CMD"
eval "$CMD"
