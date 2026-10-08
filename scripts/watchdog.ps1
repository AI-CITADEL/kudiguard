# Keep N-ATLAS, KudiGuard and the tunnel running; restart anything that stops or hangs.
#   .\scripts\watchdog.ps1 [-Named]     start/supervise (checks every 30 s; events in var\logs\watchdog.log)
#   .\scripts\watchdog.ps1 -Stop        stop the watchdog and everything it started (and only that)
# Never stop cloudflared by name: other projects on this PC may run their own tunnels.
param([switch]$Named, [switch]$Stop, [int]$Port = 8010, [int]$LlmPort = 8081, [int]$IntervalS = 30, [int]$MaxFailures = 3)
$root = Resolve-Path "$PSScriptRoot\.."
$logs = Join-Path $root "var\logs"
$run = Join-Path $root "var\run"
New-Item -ItemType Directory -Force $logs, $run | Out-Null

function Write-Event([string]$text) { "$(Get-Date -Format s) $text" | Out-File -Append -Encoding utf8 (Join-Path $logs "watchdog.log") }

# PID files hold "pid start-time-ticks": a PID reused by an unrelated process (e.g. after a reboot) never matches.
function Save-Part([string]$name, [System.Diagnostics.Process]$process) {
    "$($process.Id) $($process.StartTime.Ticks)" | Out-File -Encoding ascii (Join-Path $run "$name.pid")
}

function Get-Part([string]$name) {
    $pidFile = Join-Path $run "$name.pid"
    if (-not (Test-Path $pidFile)) { return $null }
    $id, $ticks = (Get-Content $pidFile -Raw).Trim() -split " "
    $process = Get-Process -Id $id -ErrorAction SilentlyContinue
    if ($process -and $process.StartTime.Ticks -eq [long]$ticks) { return $process }
    Remove-Item $pidFile -Force  # stale: the process ended, or the PID now belongs to something else
    return $null
}

function Stop-Part([string]$name) {
    $process = Get-Part $name
    if ($process) {
        taskkill /T /F /PID $process.Id 2>&1 | Out-Null  # the whole tree: powershell wrapper + server/tunnel
        Write-Event "stopped $name (pid $($process.Id))"
    }
    Remove-Item (Join-Path $run "$name.pid") -Force -ErrorAction SilentlyContinue
}

function Start-Part([string]$name, [string]$script, [string[]]$arguments) {
    $log = Join-Path $logs "$name-$(Get-Date -Format yyyyMMdd-HHmmss).log"  # one log per start
    $all = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$(Join-Path $PSScriptRoot $script)`"") + $arguments
    $process = Start-Process powershell -ArgumentList $all -WindowStyle Hidden `
        -RedirectStandardOutput "$log" -RedirectStandardError "$log.err" -PassThru
    Save-Part $name $process
    Write-Event "started $name (pid $($process.Id), log $log)"
}

function Test-Url([string]$url) {
    try { (Invoke-WebRequest -Uri $url -TimeoutSec 5 -UseBasicParsing).StatusCode -eq 200 } catch { $false }
}

$names = @("watchdog", "tunnel", "server", "llm")  # the watchdog first, so it cannot respawn the others
if ($Stop) { foreach ($name in $names) { Stop-Part $name }; exit 0 }
foreach ($name in $names) { Stop-Part $name }  # a previous run (and its children): never run twice
Save-Part "watchdog" (Get-Process -Id $PID)

$checks = @{
    llm    = @{ url = "http://127.0.0.1:$LlmPort/health"; script = "run_llm.ps1"; args = @("-Port", "$LlmPort") }
    server = @{ url = "http://127.0.0.1:$Port/healthz"; script = "run_server.ps1"; args = @("-Port", "$Port") }
}
$failures = @{ llm = 0; server = 0 }
while ($true) {
    foreach ($name in @("llm", "server")) {
        $check = $checks[$name]
        if (Test-Url $check.url) { $failures[$name] = 0; continue }
        if (-not (Get-Part $name)) {
            if ($name -eq "llm" -and (Test-NetConnection 127.0.0.1 -Port $LlmPort -InformationLevel Quiet -WarningAction SilentlyContinue)) {
                continue  # an N-ATLAS server started outside the watchdog is loading: leave it alone
            }
            Start-Part $name $check.script $check.args
            $failures[$name] = 0
            continue
        }
        $failures[$name] += 1  # alive but not healthy: loading, or hung
        if ($failures[$name] -ge $MaxFailures) {
            Write-Event "$name unhealthy for $($failures[$name]) checks: restarting"
            Stop-Part $name
            Start-Part $name $check.script $check.args
            $failures[$name] = 0
        }
    }
    if (-not (Get-Part "tunnel")) {
        Start-Part "tunnel" "run_tunnel.ps1" (@("-Port", "$Port") + $(if ($Named) { @("-Named") } else { @() }))
    }
    Start-Sleep -Seconds $IntervalS
}
