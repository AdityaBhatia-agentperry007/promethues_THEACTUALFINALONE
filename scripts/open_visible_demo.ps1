$ErrorActionPreference = "Stop"
$kaggleUrl = "https://www.kaggle.com/code/agentperry007/prometheus-well-mhd64-emulator"
$localUrl = "http://localhost:3000"
Start-Process -FilePath "chrome.exe" -ArgumentList $kaggleUrl -WindowStyle Normal
Start-Sleep -Seconds 2
Start-Process -FilePath "chrome.exe" -ArgumentList $localUrl -WindowStyle Normal
Write-Host "Opened Kaggle code/results page and local PROMETHEUS terminal UI."

