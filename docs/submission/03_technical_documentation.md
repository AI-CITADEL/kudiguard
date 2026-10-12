# Technical Documentation

**KudiGuard — NAIC 2026, Problem Statement 2 (Voice-First Access)**

Document 4 of the submission: architecture, setup and usage, sufficient for an independent party to run or integrate
the build. Sources: `README.md`, `docs/ARCHITECTURE.md`, `docs/DEPLOY.md`, `.env.example`, `docs/ABUSE.md`,
`docs/DPIA_VOICE.md`, `docs/CLAIMS.md`.

---

## 1. What it is

A money-safety assistant for Nigerians: send a voice note or a suspicious message in Hausa, Yoruba, Igbo, English or
Pidgin; get red / orange / green with what to do and where to report. Channels: **WhatsApp** (voice notes, text,
forwards, buttons), a **web voice page** at `/` (no install), and a **partner REST API** `/v1` (off in the pilot).
Built on **N-ATLAS** (LLM) and the **NCAIR1** speech models, all run locally.

## 2. Architecture

```
channels                      core (no I/O assumptions)              adapters
--------                      -------------------------              --------
WhatsApp webhook --+          pipeline.Conversation                  llm/openai_compat -> N-ATLAS (llama.cpp, GPU)
Web page (/) ------+-> api/app -> (consent, commands, buttons,  ->   speech/natlas_asr -> NCAIR1 (CTranslate2, CPU)
REST /v1 ----------+          limits, licence cap)                   knowledge/*       -> packs (files, partner APIs)
                              engine.Engine (stateless check)        storage/sqlite    -> hashed, redacted log
                              safety/* (rules, verdict, guards)

 PC (Windows, NVIDIA GPU >= 8 GB)
 +---------------------------------------------------------------------------+
 | scripts/watchdog.ps1  -- starts / restarts --+--------------+-------------+ |
 |   llama-server :8081 (N-ATLaS Q4_K_M GGUF)   | KudiGuard    | cloudflared | |
 |   <-- OpenAI-compatible HTTP, localhost -->  | :8010        | HTTPS tunnel| |
 |                                              | FastAPI      |  -> public  | |
 |   var/kudiguard.db (SQLite)                  | + ASR on CPU |    URL      | |
 +---------------------------------------------------------------------------+
```

**Engine flow** (`src/kudiguard/engine.py`): greeting shortcut -> N-ATLAS ANALYSE (JSON) -> translate fallback ->
deterministic verdict (`safety/decision.py`) -> knowledge search -> N-ATLAS COMPOSE -> output guard (retry once) ->
fixed fallback. Time-budgeted: model steps that no longer fit are skipped. Yoruba/Igbo replies are composed in English
and translated with fidelity and negation checks; Hausa/English are direct (`docs/ARCHITECTURE.md`).

| Component | Module | Role |
|---|---|---|
| HTTP app | `api/app.py`, `channels/whatsapp_routes.py` | web page, WhatsApp webhook, /v1, health |
| Conversation | `pipeline.py`, `commands.py` | consent, STOP/DELETE/language commands, buttons, ordering, rate limits, never-silent failures |
| Admission | `admission.py` | N-ATLaS licence cap (1,000 active users / rolling 30 days) across all channels |
| Engine | `engine.py`, `prompts.py`, `reply_translation.py` | the check: analyse, verdict, search, compose, translate |
| LLM client | `llm/base.py`, `llm/openai_compat.py`, `llm/sampling.py` | N-ATLAS calls with purposes, budgets, temperature 0 / seed 42 |
| Speech | `speech/audio.py`, `speech/natlas_asr.py`, `speech/service.py` | decode, trim, voiced-length check, NCAIR1 ASR, asr_terms |
| Safety | `safety/scam_rules.py`, `safety/decision.py`, `safety/output_guard.py`, `safety/guard_classifier.py`, `safety/grounding.py` | rules, verdict policy, reply guards |
| Knowledge | `knowledge/*`, `data/packs/*` | fact-checked packs, hybrid (BM25 + bge-small) search |
| Languages | `data/languages/{en,ha,yo,ig}.yaml` | lexicons, ASR model names, asr_terms, UI strings, abuse phrases |
| Storage | `storage/sqlite.py`, `privacy.py`, `text.py` | hashed ids, redacted text, retention purge |

## 3. Setup (Windows, NVIDIA GPU >= 8 GB)

From `README.md` and `docs/DEPLOY.md`. Tools and models are not in git:

* `tools\llama\`: llama.cpp release b11476 win-cuda-12.4 (`llama-server.exe` + CUDA DLLs).
* `models\N-ATLaS-GGUF-Q4_K_M.gguf` from Hugging Face `tosinamuda/N-ATLaS-GGUF`,
  SHA256 `1498091db609d8db8bf4637603fe8c705af17476a465b554c9c44e0ae2e69881`.
* `tools\ffmpeg\` (any LGPL build, or ffmpeg on PATH); `tools\cloudflared\cloudflared.exe` 2026.9.1.

```powershell
python -m venv .venv; .venv\Scripts\pip install -e ".[dev,semantic,asr]"
copy .env.example .env                        # fill in; KUDIGUARD_HASH_SECRET is required
.venv\Scripts\python scripts\prepare_asr.py   # one-time: NCAIR1 speech models -> CTranslate2 int8 (HF_TOKEN)
.venv\Scripts\python -c "from kudiguard.knowledge.embeddings import FastEmbedder as F; F(cache_dir='models/fastembed').embed_documents(['warm up'])"
.\scripts\watchdog.ps1                        # N-ATLAS (8081) + KudiGuard (8010) + HTTPS tunnel; restarts what stops or hangs
```

Then (WhatsApp): Meta -> WhatsApp -> Configuration -> Webhook URL `https://<tunnel-host>/webhooks/whatsapp`, verify
token from `.env`, subscribe to **messages**. Check `https://<tunnel-host>/readyz` -> `{"status":"ok", ...}`.
`scripts/setup_named_tunnel.ps1` + `watchdog.ps1 -Named` give a stable URL. N-ATLAS alone: `scripts/run_llm.ps1`.

**Hardware used**: RTX 4070 Laptop GPU 8 GB (N-ATLAS uses 7.5 GB; speech runs on CPU), `docs/SPIKES.md` S1.

**Cloud (later)**: `docker build -t kudiguard .`, same env vars, volume for `/app/var`; any GPU host running
llama.cpp or vLLM with the same GGUF via `KUDIGUARD_LLM_URL` (`docs/DEPLOY.md`).

### 3.1 `.env` keys (from `.env.example`; values omitted)

| Group | Keys |
|---|---|
| Privacy | `KUDIGUARD_HASH_SECRET` (required), `KUDIGUARD_STORAGE_LOCATION`, `KUDIGUARD_RETENTION_DAYS`, `KUDIGUARD_PATTERN_RETENTION_DAYS`, `KUDIGUARD_ALLOW_LIVE_PACKS`, `KUDIGUARD_VOICE_DONATION`, `KUDIGUARD_VOICE_NO_REASSURANCE`, `KUDIGUARD_VOICE_KEY`, `KUDIGUARD_VOICE_RETENTION_DAYS`, `KUDIGUARD_VOICE_MAX_PER_DAY`, `KUDIGUARD_VOICE_MAX_GB`, `KUDIGUARD_VOICE_DIR` |
| WhatsApp Cloud API | `WHATSAPP_PHONE_NUMBER_ID`, `WHATSAPP_ACCESS_TOKEN`, `WHATSAPP_APP_SECRET`, `WHATSAPP_VERIFY_TOKEN`, `WHATSAPP_API_VERSION`, `WHATSAPP_TEST_RECIPIENTS` |
| Models | `HF_TOKEN`, `HF_HOME`, `KUDIGUARD_LLM_URL`, `KUDIGUARD_LLM_API_KEY`, `KUDIGUARD_MODEL_CACHE`, `KUDIGUARD_EMBEDDINGS_CACHE`, `KUDIGUARD_SPEECH`, `KUDIGUARD_PRELOAD_LANGUAGES`, `KUDIGUARD_FFMPEG`, `KUDIGUARD_GUARD_CLASSIFIER_PATH` |
| Reply checks | `KUDIGUARD_GROUNDING_MODE` / `_THRESHOLD` / `_BUDGET_MS`, `KUDIGUARD_GUARD_CLASSIFIER_MODE` / `_PIDGIN_MODE` / `_THRESHOLD` / `_BUDGET_MS` |
| Advice quality | `KUDIGUARD_COMPOSE_STRATEGY`, `KUDIGUARD_KNOWLEDGE_MIN_RELEVANCE_DENSE` / `_BM25`, `KUDIGUARD_COMPOSE_EXAMPLES`, `KUDIGUARD_COMPOSE_EXAMPLE_MIN_RELEVANCE_DENSE` / `_BM25`, `KUDIGUARD_COMPOSE_EXAMPLE_COPY_MAX`, `KUDIGUARD_NOTICE_PREFIX`, `KUDIGUARD_NOTICE_MIN_RELEVANCE_DENSE` / `_BM25`, `KUDIGUARD_PIDGIN_REPLIES` |
| Data, channels, limits | `KUDIGUARD_DATA_DIR`, `KUDIGUARD_DB_PATH`, `KUDIGUARD_PACKS`, `KUDIGUARD_ALLOW_SAMPLE_PACKS`, `KUDIGUARD_WHATSAPP_ENABLED`, `KUDIGUARD_PUBLIC_URL`, `KUDIGUARD_DEFAULT_LANGUAGE`, `KUDIGUARD_PER_USER_PER_HOUR`, `KUDIGUARD_TURN_DEADLINE_S`, `KUDIGUARD_ACTIVE_USER_CAP`, `KUDIGUARD_WEB_SESSION_CAP`, `KUDIGUARD_API_RATE_PER_MINUTE`, `KUDIGUARD_WEB_RATE_PER_MINUTE`, `KUDIGUARD_TRUST_PROXY_HEADERS` |
| Partner API | `KUDIGUARD_PARTNER_API`, `KUDIGUARD_API_KEYS` |
| Malicious use (`off`/`shadow`/`enforce`) | `KUDIGUARD_ABUSE_CRISIS_MODE`, `KUDIGUARD_CRISIS_SEMANTIC_MODE` / `_THRESHOLD`, `KUDIGUARD_ABUSE_HARASSMENT_MODE`, `KUDIGUARD_ABUSE_SCAM_TOOL_MODE`, `KUDIGUARD_ABUSE_INJECTION_MODE`, `KUDIGUARD_ABUSE_INJECTION_SUSPECT_MODE`, `KUDIGUARD_REPLY_SCREEN_MODE`, `KUDIGUARD_REPLY_ECHO_WORDS`, `KUDIGUARD_ASR_HALLUCINATION_MODE`, `KUDIGUARD_PACK_SCREEN_MODE`, `KUDIGUARD_PACK_SCREEN_CORE_MODE`, `KUDIGUARD_LEARNING_GATE_MODE`, `KUDIGUARD_PATTERN_MAX_PER_USER_PER_DAY`, `KUDIGUARD_PATTERN_MAX_SIMILAR_PER_WEEK`, `KUDIGUARD_NEW_USER_MODE`, `KUDIGUARD_NEW_USERS_PER_HOUR` / `_PER_DAY`, `KUDIGUARD_FLOOD_MODE`, `KUDIGUARD_MAX_QUEUED_PER_USER`, `KUDIGUARD_RATE_NOTICE_INTERVAL_S`, `KUDIGUARD_VOICE_NOTES_PER_USER_PER_HOUR` |
| Public pages | `KUDIGUARD_OPERATOR_NAME`, `KUDIGUARD_PRIVACY_CONTACT` |

### 3.2 Settings added or decided on 12 Oct 2026 (live pilot)

| Setting | Default | Effect |
|---|---|---|
| `KUDIGUARD_PRELOAD_LANGUAGES` | `ha,en` (`config.py`) | speech models loaded at startup, so the first voice note does not wait for a model load |
| `KUDIGUARD_VOICE_NO_REASSURANCE` | `true` | a non-English voice note with no flags gets "I may not have heard every word ... please type your message" instead of the green "no red flags" line; presentation only |
| `KUDIGUARD_FOCUSED_RISK_CHECK` | `off` (experiment branch only, not merged) | extra N-ATLAS risk call on ha/yo/ig originals; measured on dev2 and rejected |
| `MIN_VOICED_SECONDS` (code constant) | 0.5 s | shorter voice notes are refused as "too short" |

## 4. Usage

### 4.1 WhatsApp
Message the number; accept the welcome (consent); choose a language; send a voice note, text or forward. Commands:
`STOP` (stop), `DELETE` (erase your data), language change; buttons "What should I do?", reporting, "Warn my
family". Meta's free test number reaches at most 5 verified recipients (`docs/DEPLOY.md`).

### 4.2 Web voice page
Open `https://<host>/`, pick a language, record or type. The page posts to `POST /web/check`; `GET /privacy` shows
the privacy notice with the configured retention values; `POST /web/donation/withdraw` withdraws a voice donation.

### 4.3 Partner REST API (`/v1`, off in the pilot)
Mounted only when `KUDIGUARD_PARTNER_API=true` with `KUDIGUARD_API_KEYS` (comma-separated `name:key` pairs).

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/check` | check a message, JSON `{"text": "...", "language": "ha"}` (`language`: ha, yo, ig, en; default en) |
| POST | `/v1/check/audio` | check a voice note (multipart upload) |
| POST | `/v1/ask` | money-safety question, JSON `{"question": "..."}` |
| GET | `/v1/languages` | supported languages |
| GET | `/v1/topics` | knowledge topics the answers are grounded on |

Headers on every call: **`X-API-Key`** and **`X-KudiGuard-End-User`**: a stable pseudonymous id for the partner's
customer, 8–137 characters (letters, digits, `. _ : -`), e.g. a SHA-256 hex digest or UUID, optionally `c-<sha256>`.
Phone- or email-shaped ids are refused (400). Each end user counts toward the shared licence cap; beyond it: 503
`at_capacity`, or 503 `busy_new_users` (`Retry-After: 3600`) from the new-user gate. Health: `GET /healthz`,
`GET /readyz`. OpenAPI schema from FastAPI.

## 5. Testing and quality

| Check | Command | Latest result |
|---|---|---|
| Unit + integration tests, coverage | `.venv\Scripts\python -m pytest --cov=kudiguard --cov-branch` | 5,533 passed, 1 skipped, 12 xfailed; **100% line + branch** (9,455 statements, 2,258 branches) (latest full run, master 17a4e13/b6cfd4e, 12 Oct 2026) |
| Lint / format | `ruff check`, `ruff format --check` | ruff clean (same run) |
| Types | `mypy` | clean (same run) |
| Live model tests | `python -m pytest -m live` | need N-ATLAS and the speech models; not counted above |
| Frozen benchmark | `PYTHONIOENCODING=utf-8 python scripts/eval_e2e.py --summary data/eval/results/e2e-live.jsonl` | T v1, document 3 |
| Mutation testing | `python scripts/mutation.py text legacy` | ten safety-critical modules 93.1% (round 3), `docs/CLAIMS.md` C9 |

Tests use fake models; coverage shows lines ran, not that behaviour is right (`docs/CLAIMS.md` C8).

## 6. Privacy and security summary

* **Identifiers**: phone numbers stored only as keyed hashes (`privacy.py`, `KUDIGUARD_HASH_SECRET`); logs never
  carry the WhatsApp message id (`docs/CLAIMS.md` P3).
* **Message text**: most numbers and emails masked before storage (`text.redact`); names and addresses are kept;
  known residual leak classes listed in `docs/CLAIMS.md` C13.
* **Voice**: voice notes deleted after transcription unless the user opts in to donation (off until the DPIA is
  signed off; `docs/DPIA_VOICE.md`); donated audio is encrypted (`KUDIGUARD_VOICE_KEY`), kept at most 12 months.
* **Retention**: messages ~90 days, scam examples under 2 years, purge every 6 hours while the server runs; users send
  `DELETE` or `STOP`.
* **Local models**: N-ATLAS, NCAIR1 and the embedding model run on our server; messages are not sent to outside AI
  companies (`docs/CLAIMS.md` P9). Pilot data is stored outside Nigeria and users are told so
  (`KUDIGUARD_STORAGE_LOCATION=outside-nigeria`).
* **Webhook security**: every WhatsApp webhook is verified with `X-Hub-Signature-256` (`WHATSAPP_APP_SECRET`).
* **Malicious use** (`docs/ABUSE.md`): crisis messages get help numbers first; abuse, scam-writing requests and prompt
  injections are screened; replies are screened for bank persona claims and prompt leaks; partner packs are screened
  at load; rate limits per user, per IP and for new users. Each defence is `off | shadow | enforce`.
* **Safety of advice**: the verdict comes from rules, not the model; KudiGuard never says a message is safe; the
  model never writes contacts; output guard on every reply (`docs/ARCHITECTURE.md`).
* **Licence**: N-ATLaS cap of 1,000 active end-users per rolling 30 days enforced in code (`admission.py`).
* **Limits**: Hausa/Yoruba/Igbo lexicons, guard patterns and texts are DRAFT pending native review; one server
  process (SQLite); one GPU serves ~10–12 checks a minute (`docs/CLAIMS.md` C11).
