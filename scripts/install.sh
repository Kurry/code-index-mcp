#!/bin/bash
# Script to install code-index-mcp and configure it for Claude for Mac and Cursor

set -e # Exit on error

# Process command-line arguments
PRE_INDEX_DIR=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --index|-i)
            shift
            PRE_INDEX_DIR="$1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "📦 Installing code-index-mcp..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "🔍 uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to the current PATH
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Get the full path of the code-index-mcp repository
REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "📂 Repository path: $REPO_PATH"

# Install the package in development mode
echo "🔧 Installing package in development mode..."
uv pip install -e "$REPO_PATH"

# Pre-index a directory if specified
if [ -n "$PRE_INDEX_DIR" ]; then
    if [ -d "$PRE_INDEX_DIR" ]; then
        echo "💾 Pre-indexing directory: $PRE_INDEX_DIR"
        uvx "$REPO_PATH" --index "$PRE_INDEX_DIR"
        echo "✅ Directory pre-indexed successfully!"
    else
        echo "⚠️  Error: $PRE_INDEX_DIR is not a valid directory."
        exit 1
    fi
fi

# Configure Claude for Mac
CLAUDE_CONFIG_PATH="$HOME/Library/Application Support/Claude/claude_desktop_config.json"
CURSOR_CONFIG_PATH="$HOME/.cursor/mcp.json"

# Create Claude config if it doesn't exist
if [ ! -f "$CLAUDE_CONFIG_PATH" ]; then
    echo "📝 Creating Claude config file..."
    mkdir -p "$(dirname "$CLAUDE_CONFIG_PATH")"
    echo '{"mcpServers":{}}' > "$CLAUDE_CONFIG_PATH"
fi

# Update Claude config
echo "🔄 Updating Claude for Mac configuration..."
# Use temp file to avoid issues with in-place editing
TMP_FILE=$(mktemp)
if [ -f "$CLAUDE_CONFIG_PATH" ]; then
    jq --arg path "$REPO_PATH" '.mcpServers."code-indexer" = {"command": "uvx", "args": [$path]}' "$CLAUDE_CONFIG_PATH" > "$TMP_FILE"
    mv "$TMP_FILE" "$CLAUDE_CONFIG_PATH"
    echo "✅ Claude for Mac configuration updated!"
    echo "   Note: Please restart Claude for Mac for the changes to take effect."
else 
    echo "⚠️  Claude config file not found. Skipping Claude configuration."
fi

# Update Cursor config (if it exists)
if [ -f "$CURSOR_CONFIG_PATH" ]; then
    echo "🔄 Updating Cursor configuration..."
    jq --arg path "$REPO_PATH" '.mcpServers."code-indexer" = {"command": "uvx", "args": [$path]}' "$CURSOR_CONFIG_PATH" > "$TMP_FILE"
    mv "$TMP_FILE" "$CURSOR_CONFIG_PATH"
    echo "✅ Cursor configuration updated!"
    echo "   Note: Please restart Cursor for the changes to take effect."
else
    echo "📝 Creating Cursor config file..."
    mkdir -p "$(dirname "$CURSOR_CONFIG_PATH")"
    echo "{\"mcpServers\":{\"code-indexer\":{\"command\":\"uvx\",\"args\":[\"$REPO_PATH\"]}}}" > "$CURSOR_CONFIG_PATH"
    echo "✅ Cursor configuration created!"
    echo "   Note: Please restart Cursor for the changes to take effect."
fi

echo ""
echo "🎉 Installation complete!"
echo ""
echo "The code-index-mcp tool is now installed and configured with the simplified uvx format:"
echo '{
  "mcpServers": {
    "code-indexer": {
      "command": "uvx",
      "args": [
        "'"$REPO_PATH"'"
      ]
    }
  }
}'
echo ""
echo "To use the Code Indexer in Claude, follow these steps:"
echo "1. Restart Claude for Mac"
echo "2. Ask Claude to help you analyze a project by saying:"
echo "   \"I need to analyze a project, help me set up the project path\""
echo ""
