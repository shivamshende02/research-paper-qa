$ErrorActionPreference = "Stop"

Write-Host "`n=== Environment Setup ===" -ForegroundColor Cyan

# 1. Check Python
try {
    $pythonExe = (& py -3 -c "import sys; print(sys.executable)").Trim()
    $pythonVersion = (& py -3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
}
catch {
    Write-Error "Python 3.10+ is required. Install Python and retry."
    exit 1
}

$version = [version]$pythonVersion

if ($version -lt [version]"3.10" -or $version -ge [version]"3.14") {
    Write-Error "Unsupported Python version: $pythonVersion. Required: 3.10–3.13."
    exit 1
}

Write-Host "Python: $pythonVersion" -ForegroundColor Green

# 2. Check uv
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv not found. Install using: winget install --id=astral-sh.uv -e"
    exit 1
}

# 3. Verify project files
if (-not (Test-Path "pyproject.toml")) {
    Write-Error "pyproject.toml not found."
    exit 1
}

if (-not (Test-Path "uv.lock")) {
    Write-Error "uv.lock not found. Run 'uv lock' first."
    exit 1
}

# 4. Create virtual environment
Write-Host "`nCreating virtual environment..."

uv venv --python $pythonExe .venv

if ($LASTEXITCODE -ne 0) {
    throw "Virtual environment creation failed."
}

# 5. Install locked dependencies
Write-Host "`nInstalling locked dependencies..."

uv sync --locked --python $pythonExe

if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed."
}

# 6. Verify critical packages
Write-Host "`nVerifying packages..."

$verification = @'
import asyncio
import httpx
import bs4
import lxml
import sentence_transformers
from importlib.metadata import version

packages = [
    "httpx",
    "beautifulsoup4",
    "lxml",
    "sentence-transformers",
]

for package in packages:
    print(f"{package}: {version(package)}")

print("asyncio: available (Python standard library)")
print("\nAll critical packages verified successfully.")
'@

& ".\.venv\Scripts\python.exe" -c $verification

if ($LASTEXITCODE -ne 0) {
    throw "Package verification failed."
}

Write-Host "`nEnvironment setup completed successfully!" -ForegroundColor Green