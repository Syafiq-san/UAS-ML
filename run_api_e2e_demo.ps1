$ErrorActionPreference = "Stop"

$workspace = "C:\Users\fatih\Documents\UAS\ML"
$python = "C:\Users\fatih\Documents\UAS\ML\.venv\Scripts\python.exe"
$logPath = Join-Path $workspace "reports\e2e_test.log"
$serverLogPath = Join-Path $workspace "reports\uvicorn_server.log"
$serverErrPath = Join-Path $workspace "reports\uvicorn_server.err.log"
$port = 8000
$baseUrl = "http://127.0.0.1:$port"

New-Item -ItemType Directory -Path (Join-Path $workspace "reports") -Force | Out-Null

@(
    "=== API end-to-end test ===",
    "Timestamp: $(Get-Date -Format o)"
) | Set-Content -Path $logPath

$serverProcess = Start-Process -FilePath $python -ArgumentList @("-m", "uvicorn", "src.api:app", "--host", "127.0.0.1", "--port", "$port") -WorkingDirectory $workspace -PassThru -RedirectStandardOutput $serverLogPath -RedirectStandardError $serverErrPath

try {
    for ($i = 0; $i -lt 30; $i++) {
        try {
            $healthResponse = Invoke-WebRequest -Uri "$baseUrl/health" -UseBasicParsing
            if ($healthResponse.StatusCode -eq 200) { break }
        } catch {
            Start-Sleep -Seconds 1
        }
    }

    "Health check:" | Add-Content -Path $logPath
    curl.exe -s -i "$baseUrl/health" | Add-Content -Path $logPath

    "Successful prediction request:" | Add-Content -Path $logPath
    $validBody = '{"features": {"Car_ID": 1, "Brand": "Toyota", "Model": "Corolla", "Year": 2018, "Kilometers_Driven": 50000, "Fuel_Type": "Petrol", "Transmission": "Manual", "Owner_Type": "First", "Mileage": 15, "Engine": 1498, "Power": 108, "Seats": 5}}'
    curl.exe -s -i -X POST "$baseUrl/predict" -H "Content-Type: application/json" -d $validBody | Add-Content -Path $logPath

    "Rejected 422 request:" | Add-Content -Path $logPath
    $invalidBody = '{"features": {"Car_ID": 1, "Brand": "Toyota", "Model": "Corolla", "Year": 2018, "Kilometers_Driven": 50000, "Fuel_Type": "Electric", "Transmission": "Manual", "Owner_Type": "First", "Mileage": 15, "Engine": 1498, "Power": 108, "Seats": 5}}'
    curl.exe -s -i -X POST "$baseUrl/predict" -H "Content-Type: application/json" -d $invalidBody | Add-Content -Path $logPath

    "Pytest run:" | Add-Content -Path $logPath
    & $python -m pytest tests/ -v 2>&1 | Tee-Object -FilePath $logPath -Append
}
finally {
    if ($serverProcess -and -not $serverProcess.HasExited) {
        Stop-Process -Id $serverProcess.Id -Force
    }
}

"Completed." | Add-Content -Path $logPath
Write-Output "Log written to $logPath"
