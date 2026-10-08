# Expose the local server over HTTPS with Cloudflare Tunnel (no open ports: the PC dials out).
#   Quick tunnel (default): free random *.trycloudflare.com URL, changes on every restart (testing only).
#   Named tunnel (-Named): stable hostname from deploy\cloudflared\kudiguard.yml (docs\DEPLOY.md).
# Each mode uses its own config file, so personal ~/.cloudflared/config.yml settings are never picked up.
param([switch]$Named, [int]$Port = 8010)
$root = Resolve-Path "$PSScriptRoot\.."
$cloudflared = Join-Path $root "tools\cloudflared\cloudflared.exe"
if ($Named) {
    & $cloudflared tunnel --config (Join-Path $root "deploy\cloudflared\kudiguard.yml") run
} else {
    & $cloudflared tunnel --config (Join-Path $root "deploy\cloudflared\quick.yml") --url "http://127.0.0.1:$Port"
}
