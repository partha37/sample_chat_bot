<#
.SYNOPSIS
    Starts the FastAPI backend and the Node frontend together.

.DESCRIPTION
    Creates .env from the template if it is missing, installs dependencies on
    first run, then launches both servers with logs written to ./.logs.

.EXAMPLE
    .\scripts\dev.ps1
#>
[CmdletBinding()]
param(
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 3000,
    [switch]$SkipInstall=$true,
    [switch]$BackendOnly,
    [switch]$FrontendOnly
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$logs = Join-Path $root '.logs'
$python = Join-Path $root '.venv\Scripts\python.exe'
$envFile = Join-Path $root '.env'

New-Item -ItemType Directory -Force -Path $logs | Out-Null

if (-not (Test-Path $envFile)) {
    Copy-Item (Join-Path $root '.env.example') $envFile
    Write-Host "[setup] Created .env - add your GEMINI_API_KEY to it" -ForegroundColor Yellow
}

if (-not $SkipInstall) {
    if (-not (Test-Path $python)) {
        Write-Host "[setup] Creating virtualenv..." -ForegroundColor Cyan
        python -m venv (Join-Path $root '.venv')
    }
    Write-Host "[setup] Installing Python dependencies..." -ForegroundColor Cyan
    & $python -m pip install -q -r (Join-Path $root 'requirements.txt')
    if (-not $FrontendOnly) {
        Write-Host "[setup] Installing Node dependencies..." -ForegroundColor Cyan
        Push-Location (Join-Path $root 'frontend')
        npm install --silent
        Pop-Location
    }
}

$processes = @()

try {
    if (-not $FrontendOnly) {
        Write-Host "[backend] Starting on http://127.0.0.1:$BackendPort" -ForegroundColor Cyan
        $backend = Start-Process -FilePath $python `
            -ArgumentList '-m', 'uvicorn', 'backend.main:app', '--host', '127.0.0.1', '--port', $BackendPort, '--reload' `
            -WorkingDirectory $root `
            -RedirectStandardOutput (Join-Path $logs 'backend.out.log') `
            -RedirectStandardError (Join-Path $logs 'backend.err.log') `
            -PassThru -WindowStyle Hidden
        $processes += $backend
    }

    if (-not $BackendOnly) {
        Write-Host "[frontend] Starting on http://localhost:$FrontendPort" -ForegroundColor Cyan
        $frontend = Start-Process -FilePath 'node' `
            -ArgumentList 'server.js' `
            -WorkingDirectory (Join-Path $root 'frontend') `
            -RedirectStandardOutput (Join-Path $logs 'frontend.out.log') `
            -RedirectStandardError (Join-Path $logs 'frontend.err.log') `
            -PassThru -WindowStyle Hidden
        $processes += $frontend
    }

    Start-Sleep -Seconds 3
    Write-Host ''
    Write-Host '  Backend  ->  http://127.0.0.1:' + $BackendPort + '/docs' -ForegroundColor Green
    Write-Host '  Frontend ->  http://localhost:' + $FrontendPort -ForegroundColor Green
    Write-Host '  Logs     ->  .\.logs\' -ForegroundColor DarkGray
    Write-Host ''
    Write-Host '  Press Ctrl+C to stop everything.' -ForegroundColor DarkGray

    while ($true) {
        Start-Sleep -Seconds 2
        $alive = $processes | Where-Object { -not $_.HasExited }
        if (-not $alive) { break }
    }
}
finally {
    Write-Host ''
    Write-Host '[stop] Shutting down...' -ForegroundColor Yellow
    foreach ($process in $processes) {
        if (-not $process.HasExited) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        }
    }
}
