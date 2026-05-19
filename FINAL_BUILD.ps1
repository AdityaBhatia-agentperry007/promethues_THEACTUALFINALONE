$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$env:PYTHONDONTWRITEBYTECODE = "1"
$env:NEXT_TELEMETRY_DISABLED = "1"

Write-Host "PROMETHEUS FINAL BUILD"
Write-Host "ROOT=$Root"

python -B -m backend.data.precompute

Push-Location "$Root\frontend"
try {
    if (-not (Test-Path -LiteralPath "node_modules")) {
        Write-Host "node_modules missing. Run npm install in frontend or copy node_modules into frontend for offline build."
        npm install
    }
    npm run build
}
finally {
    Pop-Location
}

Write-Host "BUILD_OK"
