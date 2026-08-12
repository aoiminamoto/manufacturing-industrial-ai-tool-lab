$ErrorActionPreference = "Stop"

$AppDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LensPython = Join-Path $AppDir ".venv\Scripts\python.exe"
$OcrPython = Join-Path $AppDir ".ocr-venv\Scripts\python.exe"
$RuntimeDir = Join-Path $AppDir ".runtime"

if (-not (Test-Path $LensPython)) {
    throw "Lens environment not found. Create .venv and install requirements.txt first."
}
if (-not (Test-Path $OcrPython)) {
    throw "OCR environment not found. Create .ocr-venv and install requirements-ocr.txt first."
}

New-Item -ItemType Directory -Path $RuntimeDir -Force | Out-Null

function Test-Health([string]$Url) {
    try {
        $response = Invoke-RestMethod -Uri $Url -TimeoutSec 5
        return $response.status -eq "ok"
    } catch {
        return $false
    }
}

if (-not (Test-Health "http://127.0.0.1:8506/health")) {
    $env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = "True"
    $env:OMP_NUM_THREADS = if ($env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS } else { "4" }
    Start-Process -FilePath $OcrPython -ArgumentList @("ocr_server.py") `
        -WorkingDirectory $AppDir -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $RuntimeDir "ocr-runtime.log") `
        -RedirectStandardError (Join-Path $RuntimeDir "ocr-error.log") | Out-Null
}

$ocrReady = $false
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    if (Test-Health "http://127.0.0.1:8506/health") { $ocrReady = $true; break }
    Start-Sleep -Seconds 1
}
if (-not $ocrReady) { throw "Local OCR did not become healthy. See .runtime\ocr-error.log." }

if (-not (Test-Health "http://127.0.0.1:8505/health")) {
    Start-Process -FilePath $LensPython `
        -ArgumentList @("-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8505") `
        -WorkingDirectory $AppDir -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $RuntimeDir "lens-runtime.log") `
        -RedirectStandardError (Join-Path $RuntimeDir "lens-error.log") | Out-Null
}

$lensReady = $false
for ($attempt = 0; $attempt -lt 20; $attempt++) {
    if (Test-Health "http://127.0.0.1:8505/health") { $lensReady = $true; break }
    Start-Sleep -Seconds 1
}
if (-not $lensReady) { throw "Lens did not become healthy. See .runtime\lens-error.log." }

Write-Host "PLC Lens stack is healthy."
Write-Host "Lens: http://127.0.0.1:8505"
Write-Host "OCR:  http://127.0.0.1:8506"
Write-Host "Logs: $RuntimeDir"
