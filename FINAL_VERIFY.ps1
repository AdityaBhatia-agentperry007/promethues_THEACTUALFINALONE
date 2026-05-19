$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$env:PYTHONDONTWRITEBYTECODE = "1"
$env:NEXT_TELEMETRY_DISABLED = "1"

powershell -ExecutionPolicy Bypass -File "$Root\scripts\verify_all.ps1"
