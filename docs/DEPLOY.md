# Deploy

## On this PC (competition)
1. `.env`: `KUDIGUARD_HASH_SECRET`, `WHATSAPP_PHONE_NUMBER_ID`, `WHATSAPP_ACCESS_TOKEN` (permanent System User
   token), `WHATSAPP_APP_SECRET`, `WHATSAPP_VERIFY_TOKEN` (any random string). Leave the partner API off in the
   pilot (`KUDIGUARD_PARTNER_API=false`, the default; see "Partner API" below).
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

**Where data is stored** (shown to every user): `KUDIGUARD_STORAGE_LOCATION=outside-nigeria` (the default; this PC)
or `nigeria` once the server really is in Nigeria. Moving abroad again needs a new `CONSENT_VERSION` (an operator
step: bump it in `pipeline.py` and `voice.html`; the code does not detect the move).

**Keep the clock synced.** Consent is taken only from a message the user sent at least 5 s after the welcome was
sent (accepted by WhatsApp). That compares WhatsApp's timestamp with this PC's clock.
* A clock more than 5 s **slow** (behind Meta) would treat a quick early message as consent.
* A **fast** clock only makes users wait longer before their reply counts, by up to the skew. STOP and DELETE always
  work.
* Windows: turn on Settings > Time > "Set time automatically". Then shorten the sync interval in an elevated prompt:

      w32tm /config /manualpeerlist:"time.windows.com,0x9" /syncfromflags:manual /update
      reg add HKLM\SYSTEM\CurrentControlSet\Services\W32Time\TimeProviders\NtpClient /v SpecialPollInterval /t REG_DWORD /d 900 /f
      w32tm /resync

  Alternatively, add a scheduled task that runs `w32tm /resync` every 15 minutes.
* **Before going live,** check the drift: `w32tm /stripchart /computer:time.windows.com /samples:5 /dataonly` must
  stay well under 5 s.

**Keep the server on during the pilot.** Retention purges run every 6 hours while it runs. If the PC sleeps or is off
for days, purges run late, so DELETE markers and expired data can outlive their limits until the next run.
`KUDIGUARD_RETENTION_DAYS` (31 to 90) and `KUDIGUARD_PATTERN_RETENTION_DAYS` (30 to 730) may be lowered; the
welcome, the web page and /privacy show the configured values. Longer periods are refused at startup: they would
need a new consent text and `CONSENT_VERSION`. Messages are kept at least 31 days because the licence cap counts
active users from them over a rolling 30 days: the purge runs one pass (~7 h) early, so "up to N days" holds while
the server runs, but it never cuts inside the last 30 days. `kudiguard purge --days` refuses less than 30 while the
cap is on. A DELETE by a user active in the last 30 days leaves a licence-cap marker (the time of their last
message), so their slot stays counted until 30 days after it; DELETEs made before this version left none, so on an
existing pilot database those users stop counting at once.

**Partner API (off in the pilot).** `/v1/check`, `/v1/check/audio` and `/v1/ask` are not mounted unless
`KUDIGUARD_PARTNER_API=true` (with `KUDIGUARD_API_KEYS`). The N-ATLaS licence caps *active end-users*, not API keys,
so when it is on every call must send `X-KudiGuard-End-User`: a stable pseudonymous id for the partner's customer
(8-137 characters: letters, digits, `. _ : -`). Send a SHA-256 (or SHA-512) hex digest of your own customer id,
or a UUID, optionally behind a short letter tag of up to 8 letters (`c-<sha256>`): these are always accepted. A
guardrail refuses (400) ids that look like a phone number or an email address (digits only, a run of 10+ digits once
separators are removed, a mail provider or a domain ending); it catches mistakes, not deliberate abuse, and other
formats (short hex tokens, date-based order numbers such as `order-2024-10-11-1234`) may be refused too. The id is
hashed as `api:<key name>:<id>`, logged as channel `api`, and counts toward `KUDIGUARD_ACTIVE_USER_CAP` with WhatsApp
and web users through the same admission gate; a new end user gets 503 `at_capacity` when the cap is full (or 503
`busy_new_users` from the new-user rate gate: `Retry-After: 3600` at the hourly limit, none at the daily one). The
new-user gate (`KUDIGUARD_NEW_USERS_PER_HOUR` / `_PER_DAY`, docs/ABUSE.md §5) is installed on this shared admission,
so it applies to WhatsApp, web and partner-API end users alike, and also when `KUDIGUARD_ACTIVE_USER_CAP` is unset
(it is an anti-flood guard, not a licence measure). Renaming a key
(`name:key`) re-counts its users as new for 30 days. A partner who needs more than our share of the cap needs its own N-ATLaS licence cover (its own deployment) or a
commercial licence. **The partner must tell its own customers and get their agreement** (they never see our
welcome); /privacy then describes what is kept for them. Their messages never feed the scam-pattern library.

**Partner knowledge packs in live mode** (`mode: live`) send the English text of each message to the partner's
server; they are refused at startup unless `KUDIGUARD_ALLOW_LIVE_PACKS=true`, which also adds that disclosure to
/privacy. Prefer `mode: sync` (nothing about the user leaves the server).

**Known friction:** the language menu arrives with the welcome. A user who taps a language within 5 s gets "please
read the message above" once, and their next tap counts.

**Restrict files to the service account** (Windows ignores POSIX modes). Run in an elevated prompt, with the
account that runs the watchdog:

    icacls var /inheritance:r /grant:r "%USERNAME%:(OI)(CI)F" "SYSTEM:(OI)(CI)F"
    icacls .env /inheritance:r /grant:r "%USERNAME%:F" "SYSTEM:F"

**Voice-note donation** (opt-in; docs/DPIA_VOICE.md). This is **off** until the DPIA is signed off.
* *Before enabling:* send one normal voice note recorded in WhatsApp to the live number, and check that the log
  line `voice message: recorded=True forwarded=False` appears. If `recorded` is False, nothing would ever be kept.
  Then send one real *forwarded* voice note and record in the DPIA whether Meta marks it.
* *Turning it on:* `KUDIGUARD_VOICE_DONATION=true`, then restart.
* *Storage:* encrypted audio lives in `var/voice/` (`KUDIGUARD_VOICE_DIR`). A `.vault` marker ties it to its
  database, so never point it at another database's folder.
* *Deletion:* audio is purged after 12 months by the server and by `kudiguard purge`. STOP DONATING, STOP and DELETE
  erase it at once.
* *Backups:* back up `var/` without `.env`, so the copy cannot be decrypted.
* *Key rotation:* `kudiguard voice wipe --all`, then change `KUDIGUARD_VOICE_KEY` (or the hash secret), then restart.
  If wipe exits with "ERROR: N voice files remain", the folder belongs to another database (check
  `KUDIGUARD_VOICE_DIR` and `KUDIGUARD_DB_PATH`) or the files are locked (stop the server). Fix that and run wipe
  again. Do not rotate until the folder holds no `.kgv` or `.part` files. Delete stray files by hand only once you
  are sure they are this deployment's.
* *Requests by email:* `kudiguard voice access|erase --user <number | web:<donation code>>`. Confirm the requester
  first (DPIA §6).
* *Flagged samples* (a child's voice, someone else's voice, or credentials read out):
  `kudiguard voice remove <id> ...`.
* *Tester sign-up form:* see docs/TESTER_SIGNUP.md.

**Launch checklist: crisis help numbers (hard requirement, docs/ABUSE.md §6).** Do not launch, and re-check every
quarter: call every number in the `crisis_reply` string of each language (`data/languages/*.yaml`): 112, SURPIN
0800 078 7746 (toll-free), 0903 440 0009, 0908 021 7555 and the Hausa line 0814 224 1007 (in the Hausa replies). Each must reach a person. Remove any number that does
not (in all four languages), and record the date and result next to the string. If none of the SURPIN lines works,
keep "112 / someone you trust" only.

**Incident** (personal-data breach): contain; assess; notify the NDPC within 72 hours when people's rights are at
risk; tell affected people immediately when the risk to them is high; record it (docs/DPIA_VOICE.md §8).

## In the cloud (later)
* **Container**: `docker build -t kudiguard .` then run with the same env vars and a volume for `/app/var`.
* **N-ATLAS**: any GPU host running llama.cpp (or vLLM) with the same GGUF; set `KUDIGUARD_LLM_URL`.
* **Models**: mount the converted speech models at `/models` and the embedding model at `/models/fastembed`
  (speech runs on CPU; without the embedding model, search falls back to keywords).
* **HTTPS**: a public VM needs no tunnel; put Caddy/nginx in front. Set `KUDIGUARD_TRUST_PROXY_HEADERS=true` only
  behind Cloudflare (other proxies pass a client-sent `CF-Connecting-IP` through).
* **Several instances**: move storage to a shared database and route each user to one instance.
