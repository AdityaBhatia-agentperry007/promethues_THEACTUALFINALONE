$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
python scripts/verify.py

