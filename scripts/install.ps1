# PowerShell script to install code-index-mcp and configure it for Claude for Windows and Cursor

# Exit on error
$ErrorActionPreference = "Stop"

# Process command-line arguments
$preIndexDir = $null
for ($i = 0; $i -lt $args.Count; $i++) {
    if ($args[$i] -eq "--index" -or $args[$i] -eq "-i") {
        if ($i + 1 -lt $args.Count) {
            $preIndexDir = $args[$i + 1]
            $i++
        }
    }
}

Write-Host "📦 Installing code-index-mcp..." -ForegroundColor Cyan

# Check if uv is installed
$uvInstalled = $null
try {
    $uvInstalled = Get-Command uv -ErrorAction SilentlyContinue
} catch {
    # Do nothing
}

if ($null -eq $uvInstalled) {
    Write-Host "🔍 uv not found. Installing uv..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri https://astral.sh/uv/install.ps1 -OutFile .\install-uv.ps1
    .\install-uv.ps1
    Remove-Item .\install-uv.ps1 -Force
    
    # Add uv to the current PATH (for this session)
    $env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"
}

# Get the full path of the code-index-mcp repository
$RepoPath = $PSScriptRoot
Write-Host "📂 Repository path: $RepoPath" -ForegroundColor Cyan

# Check if code-index-mcp is already installed
$alreadyInstalled = $false
if (Get-Command uvx -ErrorAction SilentlyContinue) {
    Write-Host "🔍 Checking if code-index-mcp is already installed..." -ForegroundColor Cyan
    $installationStatus = uvx --help 2>&1
    if ($installationStatus -match "code_index_mcp.server") {
        Write-Host "✅ code-index-mcp is already installed." -ForegroundColor Green
        $alreadyInstalled = $true
    } else {
        Write-Host "ℹ️  uvx command exists but code-index-mcp is not installed yet." -ForegroundColor Yellow
        $alreadyInstalled = $false
    }
}

# Install the package in development mode if not already installed
if (-not $alreadyInstalled) {
    Write-Host "🔧 Installing package in development mode..." -ForegroundColor Cyan
    uv pip install -e $RepoPath
} else {
    Write-Host "ℹ️  Skipping installation as package is already installed." -ForegroundColor Cyan
}

# Pre-index a directory if specified
if ($preIndexDir) {
    if (Test-Path -Path $preIndexDir -PathType Container) {
        Write-Host "💾 Pre-indexing directory: $preIndexDir" -ForegroundColor Cyan
        uvx $RepoPath --index $preIndexDir
        Write-Host "✅ Directory pre-indexed successfully!" -ForegroundColor Green
    } else {
        Write-Host "⚠️ Error: $preIndexDir is not a valid directory." -ForegroundColor Red
        exit 1
    }
}

# Configure Claude for Windows
$ClaudeConfigPath = "$env:APPDATA\Claude\claude_desktop_config.json"
$CursorConfigPath = "$env:APPDATA\Cursor\mcp_config.json"

# Create Claude config directory if it doesn't exist
if (-not (Test-Path (Split-Path $ClaudeConfigPath -Parent))) {
    New-Item -ItemType Directory -Path (Split-Path $ClaudeConfigPath -Parent) -Force | Out-Null
}

# Create or update Claude config
Write-Host "🔄 Updating Claude for Windows configuration..." -ForegroundColor Cyan
$ClaudeConfig = $null

if (Test-Path $ClaudeConfigPath) {
    $ClaudeConfig = Get-Content $ClaudeConfigPath -Raw | ConvertFrom-Json
} else {
    $ClaudeConfig = @{
        mcpServers = @{}
    }
}

# Update or add the code-indexer configuration
$ClaudeConfig.mcpServers.'code-indexer' = @{
    command = "uvx"
    args = @($RepoPath)
}

# Save the updated configuration
$ClaudeConfig | ConvertTo-Json -Depth 10 | Set-Content -Path $ClaudeConfigPath -Encoding UTF8
Write-Host "✅ Claude for Windows configuration updated!" -ForegroundColor Green
Write-Host "   Note: Please restart Claude for Windows for the changes to take effect." -ForegroundColor Yellow

# Update Cursor config (if it exists)
if (Test-Path $CursorConfigPath) {
    Write-Host "🔄 Updating Cursor configuration..." -ForegroundColor Cyan
    $CursorConfig = Get-Content $CursorConfigPath -Raw | ConvertFrom-Json
    
    # Update or add the code-indexer configuration
    $CursorConfig.mcpServers.'code-indexer' = @{
        command = "uvx"
        args = @($RepoPath)
    }
    
    # Save the updated configuration
    $CursorConfig | ConvertTo-Json -Depth 10 | Set-Content -Path $CursorConfigPath -Encoding UTF8
    Write-Host "✅ Cursor configuration updated!" -ForegroundColor Green
    Write-Host "   Note: Please restart Cursor for the changes to take effect." -ForegroundColor Yellow
} else {
    Write-Host "ℹ️  Cursor config not found. If you use Cursor, you'll need to manually configure it." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎉 Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "The code-index-mcp tool is now installed and configured with the simplified uvx format:" -ForegroundColor Cyan
Write-Host @"
{
  "mcpServers": {
    "code-indexer": {
      "command": "uvx",
      "args": [
        "$($RepoPath -replace '\\', '\\')"
      ]
    }
  }
}
"@ -ForegroundColor White
Write-Host ""
Write-Host "To use the Code Indexer in Claude, follow these steps:" -ForegroundColor Cyan
Write-Host "1. Restart Claude for Windows" -ForegroundColor White
Write-Host "2. Ask Claude to help you analyze a project by saying:" -ForegroundColor White
Write-Host "   \"I need to analyze a project, help me set up the project path\"" -ForegroundColor Yellow
Write-Host ""
