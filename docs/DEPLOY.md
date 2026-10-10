# Deploy

## On this PC (competition)
1. `.env`: `KUDIGUARD_HASH_SECRET`, `WHATSAPP_PHONE_NUMBER_ID`, `WHATSAPP_ACCESS_TOKEN` (permanent System User
   token), `WHATSAPP_APP_SECRET`, `WHATSAPP_VERIFY_TOKEN` (any random string), optional `KUDIGUARD_API_KEYS`.
2. `.\scripts\watchdog.ps1 [-Named]` starts N-ATLAS (8081), KudiGuard (8010) and the tunnel and restarts what stops or
   hangs; `-Stop` stops only what it started. Never `Stop-Process -Name cloudflared` (other tunnels on this PC).
   (events in `var\logs\watchdog.log`, one log per start; the tunnel URL is in the latest `tunnel-*.log.err`).
3. Meta → WhatsApp → Configuration → Webhook: URL `https://<tunnel-host>/webhooks/whatsapp`, verify token from
   `.env`; subscribe to **messages** (statuses arrive on the same field).
4. Check: `https://<tunnel-host>/readyz` → `{"status":"ok", "whatsapp":true, ...}`; send "Hello" from WhatsApp.

**Meta limits**: the free test number reaches only up to 5 verified recipient numbers; real users need a registered
business number and the app in Live mode. Replies are free within 24 h of the user's message (KudiGuard only replies).

**Stable URL (named tunnel)**: quick tunnels change URL on restart, so Meta's webhook would break. One-time:
`.\scripts\setup_named_tunnel.ps1` (browser login for the domain; default `kudiguard.theaicitadel.com`), then
`.\scripts\watchdog.ps1 -Named`. It keeps its own certificate and config, so other tunnels on this PC are untouched.

**Voice-note donation** (opt-in; docs/DPIA_VOICE.md): encrypted audio lives in `var/voice/` (or
`KUDIGUARD_VOICE_DIR`, which must belong to the same database), is purged after 12 months by the server and by
`kudiguard purge`, and is erased by STOP DONATING / DELETE. Back up `var/` without `.env` (the key comes from
`KUDIGUARD_HASH_SECRET`). Requests by email: `kudiguard voice access|erase --user <number | web:<session>>`.

**Incident** (personal-data breach): contain, assess, notify the NDPC within 72 hours when people's rights are at
risk, tell affected users when the risk is high, record it (docs/DPIA_VOICE.md §8).

## In the cloud (later)
* **Container**: `docker build -t kudiguard .` then run with the same env vars and a volume for `/app/var`.
* **N-ATLAS**: any GPU host running llama.cpp (or vLLM) with the same GGUF; set `KUDIGUARD_LLM_URL`.
* **Models**: mount the converted speech models at `/models` and the embedding model at `/models/fastembed`
  (speech runs on CPU; without the embedding model, search falls back to keywords).
* **HTTPS**: a public VM needs no tunnel; put Caddy/nginx in front. Set `KUDIGUARD_TRUST_PROXY_HEADERS=true` only
  behind Cloudflare (other proxies pass a client-sent `CF-Connecting-IP` through).
* **Several instances**: move storage to a shared database and route each user to one instance.
