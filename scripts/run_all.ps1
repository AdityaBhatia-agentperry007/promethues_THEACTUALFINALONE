$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$libraryPath = Join-Path (Get-Location) "backend\data\result_library.npz"
$scenarioPath = Join-Path (Get-Location) "backend\data\scenarios.json"
if (-not (Test-Path -LiteralPath $libraryPath) -or -not (Test-Path -LiteralPath $scenarioPath)) {
    python -m backend.data.precompute
}
$python = (Get-Command python).Source
$npm = (Get-Command npm.cmd).Source
$backend = Start-Process -FilePath $python -ArgumentList "-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", "8000" -PassThru -WindowStyle Hidden
$frontend = Start-Process -FilePath $npm -ArgumentList "run", "dev", "--", "-p", "3000" -WorkingDirectory "frontend" -PassThru -WindowStyle Hidden
Write-Host "Backend running at http://127.0.0.1:8000 (PID $($backend.Id))"
Write-Host "Frontend running at http://localhost:3000 (PID $($frontend.Id))"
Write-Host "Stop them with: Stop-Process -Id $($backend.Id),$($frontend.Id)"
