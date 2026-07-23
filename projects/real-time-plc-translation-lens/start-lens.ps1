$ErrorActionPreference = "Stop"

$AppDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $AppDir ".venv\Scripts\python.exe"
$Python = if (Test-Path $VenvPython) { $VenvPython } else { "python" }
$Port = if ($env:PORT) { $env:PORT } else { "8505" }

Set-Location $AppDir
Write-Host "Starting Real-Time PLC Translation Lens at http://localhost:$Port"
& $Python -m uvicorn app:app --host 127.0.0.1 --port $Port

