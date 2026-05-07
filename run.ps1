Param()

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $scriptDir "venv\Scripts\python.exe"
$appPath = Join-Path $scriptDir "app\app.py"

if (-not (Test-Path $venvPython)) {
    Write-Host "Python virtual environment not found at $venvPython"
    Write-Host "Create it with:"
    Write-Host "  python -m venv venv"
    exit 1
}

if (-not (Test-Path $appPath)) {
    Write-Host "Application file not found at $appPath"
    exit 1
}

Write-Host "Starting AgroIntell Flask app..."
& $venvPython $appPath

