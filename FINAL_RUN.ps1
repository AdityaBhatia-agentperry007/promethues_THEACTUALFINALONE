$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Frontend = Join-Path $Root "frontend"

Write-Host "PROMETHEUS FINAL RUN"
Write-Host "ROOT=$Root"
Write-Host "BACKEND=http://127.0.0.1:8000"
Write-Host "FRONTEND=http://localhost:3000"

Start-Process -FilePath python -ArgumentList "-B","-m","uvicorn","backend.app:app","--host","127.0.0.1","--port","8000" -WorkingDirectory $Root -WindowStyle Hidden
Start-Sleep -Seconds 2

Start-Process -FilePath "C:\Program Files\nodejs\npm.cmd" -ArgumentList "run","dev","--","-p","3000" -WorkingDirectory $Frontend -WindowStyle Hidden
Start-Sleep -Seconds 2

Write-Host "SERVERS_STARTED"
Write-Host "Open http://localhost:3000"
