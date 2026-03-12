param(
    [switch]$Fix
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot ".venv\\Scripts\\python.exe"

if (-not (Test-Path $pythonExe)) {
    throw "Virtual environment python not found: $pythonExe"
}

Push-Location $projectRoot
try {
    if ($Fix) {
        & $pythonExe -m ruff format src apps tests scripts
        & $pythonExe -m ruff check --fix src apps tests scripts
    } else {
        & $pythonExe -m ruff format --check src apps tests scripts
        & $pythonExe -m ruff check src apps tests scripts
    }

    & $pythonExe -m pyright
    & $pythonExe -m pytest -q
} finally {
    Pop-Location
}
