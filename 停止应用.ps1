$ErrorActionPreference = 'Stop'
$appRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pidRoot = Join-Path $appRoot 'runtime-data\pids'

foreach ($name in @('frontend', 'backend')) {
    $pidPath = Join-Path $pidRoot "$name.pid"
    if (-not (Test-Path -LiteralPath $pidPath)) { continue }
    $processId = [int](Get-Content -LiteralPath $pidPath -Raw).Trim()
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$processId" -ErrorAction SilentlyContinue
    if ($process -and $process.CommandLine -like "*$appRoot*") {
        Stop-Process -Id $processId -Force
        Write-Host "已停止 $name。"
    } elseif ($process) {
        Write-Warning "$name.pid 指向的进程不属于本项目，未停止。"
    }
    Remove-Item -LiteralPath $pidPath -Force
}

Write-Host '项目脚本启动的前后端已停止；Ollama 保持运行。' -ForegroundColor Green
