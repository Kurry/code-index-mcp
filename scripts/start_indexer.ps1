# PowerShell script to start the Code Index MCP server with embeddings

# Exit on error
$ErrorActionPreference = "Stop"

# Initialize variables
$useQdrantCloud = $false
$projectDir = $null

# Check for directory argument
if ($args.Count -gt 0 -and -not $args[0].StartsWith("-")) {
    # First arg is not a flag, treat as directory
    $projectDir = $args[0]
    $args = $args[1..($args.Count-1)]
}

# Parse remaining arguments
for ($i = 0; $i -lt $args.Count; $i++) {
    if ($args[$i] -eq "--use-qdrant-cloud") {
        $useQdrantCloud = $true
    } else {
        Write-Host "Unknown option: $($args[$i])" -ForegroundColor Red
        Write-Host "Usage: .\start_indexer.ps1 [C:\path\to\project] [--use-qdrant-cloud]"
        exit 1
    }
}

# Get the full path of the code-index-mcp repository
$repoPath = $PSScriptRoot

# Check for a .env file
$envFile = Join-Path -Path $repoPath -ChildPath ".env"
if (Test-Path -Path $envFile -PathType Leaf) {
    Write-Host "💼 Found .env file, will use environment variables from there" -ForegroundColor Cyan
} else {
    Write-Host "ℹ️ No .env file found, checking for environment variables directly" -ForegroundColor Yellow
    
    # Check for API keys and Qdrant configuration
    if ($useQdrantCloud) {
        if (-not $env:QDRANT_API_KEY) {
            Write-Host "⚠️ Warning: QDRANT_API_KEY environment variable not set" -ForegroundColor Yellow
            Write-Host "You should create a .env file based on .env.example or set the environment variable" -ForegroundColor Yellow
        }
        
        if (-not $env:QDRANT_URL) {
            Write-Host "⚠️ Warning: QDRANT_URL environment variable not set" -ForegroundColor Yellow
            Write-Host "You should create a .env file based on .env.example or set the environment variable" -ForegroundColor Yellow
        }
    }
    
    # Check for OpenAI API key
    if (-not $env:OPEN_AI_KEY) {
        Write-Host "⚠️ Warning: OPEN_AI_KEY environment variable not set" -ForegroundColor Yellow
        Write-Host "You should create a .env file based on .env.example or set the environment variable" -ForegroundColor Yellow
    }
    
    # Check for project directory if not specified as argument
    if (-not $projectDir -and -not $env:PROJECT_DIRECTORY) {
        Write-Host "⚠️ Warning: No project directory specified and PROJECT_DIRECTORY environment variable not set" -ForegroundColor Yellow
        Write-Host "You should create a .env file based on .env.example or specify the directory as an argument" -ForegroundColor Yellow
    }
}

Write-Host "Starting Code Index MCP server..." -ForegroundColor Cyan

# If directory was passed as argument, use it
if ($projectDir) {
    Write-Host "Project directory from argument: $projectDir" -ForegroundColor Cyan
} else {
    Write-Host "Using project directory from .env or environment variables" -ForegroundColor Cyan
}

Write-Host "Using Qdrant Cloud: $useQdrantCloud" -ForegroundColor Cyan

# Build the command
$cmd = "uv run run.py"

# Add project directory if specified as argument
if ($projectDir) {
    $cmd += " `"$projectDir`""
}

# Add Qdrant Cloud flag if needed
if ($useQdrantCloud) {
    $cmd += " --use-qdrant-cloud"
}

# Run the server
Set-Location -Path $repoPath
Write-Host "Executing: $cmd" -ForegroundColor Cyan
Invoke-Expression $cmd
