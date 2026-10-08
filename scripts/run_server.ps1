# Start KudiGuard (API, web page, WhatsApp webhook) on this PC. Settings come from .env (see .env.example).
#   Requires: N-ATLAS running (scripts\run_llm.ps1) and KUDIGUARD_HASH_SECRET in .env.
param([int]$Port = 8010)
$root = Resolve-Path "$PSScriptRoot\.."
$envFile = Join-Path $root ".env"
if (-not (Select-String -Path $envFile -Pattern '^KUDIGUARD_HASH_SECRET=.{32,}' -Quiet)) {
    Write-Error "Add KUDIGUARD_HASH_SECRET (>= 32 chars) to .env. Generate: .venv\Scripts\python -c `"import secrets;print(secrets.token_urlsafe(48))`""
    exit 1
}
$env:KUDIGUARD_TRUST_PROXY_HEADERS = "true"  # behind the Cloudflare tunnel only: client IP from CF-Connecting-IP
$env:HF_HUB_OFFLINE = "1"                    # never download models at runtime
& "$root\.venv\Scripts\kudiguard.exe" serve --host 127.0.0.1 --port $Port
