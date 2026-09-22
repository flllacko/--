param([switch]$NoBrowser)

$ErrorActionPreference = 'Stop'
$appRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendRoot = Join-Path $appRoot 'backend'
$frontendRoot = Join-Path $appRoot 'frontend'
$runtimeRoot = Join-Path $appRoot 'runtime-data'
$logRoot = Join-Path $runtimeRoot 'logs'
$pidRoot = Join-Path $runtimeRoot 'pids'
$pythonPath = Join-Path $backendRoot '.venv\Scripts\python.exe'
$vitePath = Join-Path $frontendRoot 'node_modules\vite\bin\vite.js'

New-Item -ItemType Directory -Force -Path $logRoot, $pidRoot | Out-Null

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw '后端虚拟环境不存在。请先按 README 执行第一次安装。'
}
if (-not (Test-Path -LiteralPath $vitePath)) {
    throw '前端依赖不存在。请先在 frontend 目录执行 npm install。'
}

$nodeCommand = Get-Command node -ErrorAction Stop
$ollamaCommand = Get-Command ollama -ErrorAction Stop

function Test-Http([string]$url) {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 2
        return $response.StatusCode -ge 200 -and $response.StatusCode -lt 500
    } catch { return $false }
}

function Wait-Http([string]$url, [int]$seconds) {
    for ($attempt = 0; $attempt -lt ($seconds * 2); $attempt++) {
        if (Test-Http $url) { return $true }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

if (-not (Test-Http 'http://127.0.0.1:11434/api/tags')) {
    Start-Process -WindowStyle Hidden -FilePath $ollamaCommand.Source -ArgumentList 'serve' | Out-Null
    if (-not (Wait-Http 'http://127.0.0.1:11434/api/tags' 15)) {
        throw 'Ollama 启动失败，请打开 Ollama 后重试。'
    }
}

$models = (& $ollamaCommand.Source list | Out-String)
foreach ($requiredModel in @('qwen3.5:9b', 'qwen3-embedding:0.6b')) {
    if ($models -notmatch [regex]::Escape($requiredModel)) {
        throw "缺少模型 $requiredModel，请先执行 ollama pull $requiredModel。"
    }
}

if (-not (Test-Http 'http://127.0.0.1:8000/api/health')) {
    if (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue) {
        throw '8000 端口已被其他程序占用，无法启动后端。'
    }
    $backend = Start-Process -PassThru -WindowStyle Hidden -FilePath $pythonPath `
        -WorkingDirectory $backendRoot `
        -ArgumentList '-m','uvicorn','app.main:app','--host','127.0.0.1','--port','8000' `
        -RedirectStandardOutput (Join-Path $logRoot 'backend-out.log') `
        -RedirectStandardError (Join-Path $logRoot 'backend-error.log')
    Set-Content -LiteralPath (Join-Path $pidRoot 'backend.pid') -Value $backend.Id
}

if (-not (Wait-Http 'http://127.0.0.1:8000/api/health' 20)) {
    throw "后端未通过健康检查，请查看 $logRoot\backend-error.log。"
}

if (-not (Test-Http 'http://127.0.0.1:5173')) {
    if (Get-NetTCPConnection -LocalPort 5173 -State Listen -ErrorAction SilentlyContinue) {
        throw '5173 端口已被其他程序占用，无法启动前端。'
    }
    $frontend = Start-Process -PassThru -WindowStyle Hidden -FilePath $nodeCommand.Source `
        -WorkingDirectory $frontendRoot `
        -ArgumentList $vitePath,'--host','127.0.0.1','--port','5173','--strictPort' `
        -RedirectStandardOutput (Join-Path $logRoot 'frontend-out.log') `
        -RedirectStandardError (Join-Path $logRoot 'frontend-error.log')
    Set-Content -LiteralPath (Join-Path $pidRoot 'frontend.pid') -Value $frontend.Id
}

if (-not (Wait-Http 'http://127.0.0.1:5173' 20)) {
    throw "前端未正常打开，请查看 $logRoot\frontend-error.log。"
}

Write-Host '学生服务问答助手已启动：http://127.0.0.1:5173' -ForegroundColor Green
if (-not $NoBrowser) { Start-Process 'http://127.0.0.1:5173' }
