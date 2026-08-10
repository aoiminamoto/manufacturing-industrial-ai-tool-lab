$ErrorActionPreference = "Stop"

$AppDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $AppDir ".ocr-venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    throw "OCR environment not found. Create .ocr-venv and install requirements-ocr.txt first."
}

$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = "True"
$env:OMP_NUM_THREADS = if ($env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS } else { "4" }

Set-Location $AppDir
Write-Host "Starting local-only OCR at http://127.0.0.1:8506"
& $VenvPython .\ocr_server.py
