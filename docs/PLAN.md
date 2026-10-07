# KudiGuard — Build Plan (NAIC 2026, Problem Statement 2: Voice-First Access)

> *Kuɗi* = "money" (Hausa). A WhatsApp voice-note assistant for **financial literacy and scam checks**,
> Hausa-first, also English; Yoruba and Igbo in beta. Built on **N-ATLAS LLM + official N-ATLAS (NCAIR1) ASR**.

Deadline **12 Oct 2026 23:59 WAT** · code freeze 12 Oct 10:00 · submit by 14:00 · service stays up through the
N-ATLAS integration check (15–17 Oct). Revision 2 — incorporates independent deep review (7 Oct).

## 1. User experience

1. First contact → fixed consent message (purpose, what is stored, `STOP`/`DELETE`, not financial advice,
   "Powered by N-ATLAS · Awarri") + language list (WhatsApp interactive list; also accepts typed/spoken choice).
2. User sends a **voice note** (≤90 s) or text: *"Someone called saying my BVN is blocked and asked for my OTP"*.
3. Reply in the user's language: fixed verdict headline (**likely scam / be careful / no red flags found — still
   verify with your bank**), short explanation, what to do, where to report; *"I heard: …"* echo for voice.
   Never says "safe". General money questions get short KB-grounded answers.
4. Reply buttons: 👍 Helpful / 👎 Not helpful → validation evidence.

### 1a. "Wow" experience (non-negotiables)

* **Instant feedback**: mark-as-read + "🎧 Listening…" acknowledgement within 1 s; full answer typically < 15 s.
* **Traffic-light verdict** 🔴 / 🟠 / 🟢 as the first line — scannable at a glance, readable by low-literacy users.
* **Voice back**: native-speaker recorded clip of the verdict headline sent with the text (TTS stretch).
* **One-tap next steps** (reply buttons): *What should I do?* · *Where do I report?* · *Warn my family*.
* **Warn my family**: generates a short forwardable warning in the user's language — useful, and drives organic
  growth of real interactions.
* **Forward-to-check**: a forwarded suspicious SMS/WhatsApp text is checked with no extra typing.
* **Actionable reporting**: correct channels (own bank's fraud line, NCC 622, EFCC, CBN complaints) in every scam reply.
* **Graceful failure**: never silence — friendly retry message in the user's language on any error.

### 1a-ii. Public API — anyone can integrate KudiGuard

Same core, exposed as a documented REST API so banks, fintechs, apps, USSD/IVR providers and NGOs can embed it:

* `POST /v1/check` — JSON `{text, language?}` or multipart `audio` → `{verdict, flags[], headline, explanation,
  actions[], report_channels[], transcript?, english?, language}`; stateless, no phone number needed.
* `POST /v1/ask` — general money-literacy question → grounded answer in the requested language.
* `GET /v1/languages`, `GET /v1/topics`, `GET /healthz`.
* API keys (header `X-API-Key`), per-key rate limits, OpenAPI docs at `/docs`, copy-paste examples (curl, Python, JS).
* WhatsApp and the web page are just clients of the same pipeline — proves the API in production.
* Scalability story for judges: one engine, many channels (WhatsApp today; bank apps, USSD/IVR via partners).

### 1b. Delivery process — sequential, gated

Each milestone: write tests → implement → `ruff` + `mypy` + `pytest` (coverage ≥90% on the module) →
**delegated independent review** → fix findings → only then start the next milestone.

| # | Milestone | Depends on |
|---|---|---|
| M1 | `domain` + `safety/scam_rules` (flag taxonomy, 4-language lexicons, scoring) | — |
| M2 | `knowledge/kb` + `data/kb/topics.yaml` (content for ~15 topics, report channels) | M1 |
| M3 | `i18n` (fixed strings, 4 languages) + `prompts` (analyse/compose, JSON schema) | M1, M2 |
| M4 | `llm/openai_compat` + live spike against N-ATLAS GGUF | M3 |
| M5 | `speech/audio` + `speech/natlas_asr` + live Hausa spike | — |
| M6 | `privacy` + `storage/sqlite` | M1 |
| M7 | `pipeline` (orchestration, commands, degradation, UX flows) | M1–M6 |
| M8 | `channels/whatsapp` (signature, parse, media, buttons, send) + fixtures | M7 |
| M9 | `api/app`: public REST API (`/v1/check`, `/v1/ask`, keys, rate limits, OpenAPI) + web voice page + CLI export | M7, M8 |
| M10 | Deploy (tunnel, watchdog), eval scripts, docs, validation | M9 |

## 2. NAIC requirements → evidence

| Requirement | Evidence |
|---|---|
| Working artefact | Live WhatsApp number + single-file web voice page fallback + repo |
| N-ATLAS integration depth | N-ATLAS LLM does structured red-flag extraction **and** reply generation; NCAIR1 ASR per language for all voice |
| ≥50 real interactions (target 80–100) | SQLite log (hashed users) → `kudiguard export` → `docs/VALIDATION.md`: unique non-team users, language split, verdict mix, helpful %, p50/p95 latency, 10 anonymised transcripts with glosses, testimonials |
| Technical rigour | ASR WER on 20 Hausa + 20 English recorded clips; scam precision/recall on 30 labelled scenarios (`scripts/eval_*.py`) |
| Docs, video, team, Track B docs | README, ARCHITECTURE, SUSTAINABILITY (cost/interaction, licence path, partners), DEMO script; video filmed D3 |

## 3. Pipeline (ports & adapters)

```
WhatsApp webhook ─► verify sig · ignore statuses · dedupe wamid · rate-limit (20/h) ─► 200 OK, process in background
Web voice page  ─┘
   │
   ├─ commands: STOP / DELETE / language
   ├─ audio → ffmpeg → 16 kHz mono (≤90 s) → ASR[lang]  (NCAIR1 Whisper-small, CPU)
   ├─ ANALYSE  (LLM call 1, JSON schema): {english, flags[], topic}         ← N-ATLAS
   ├─ RULES    lexicon over original + english ∪ LLM flags → score → Verdict  (deterministic, wins)
   ├─ KB       entries selected by topic/flags (YAML, English facts)
   ├─ COMPOSE  (LLM call 2): ≤250 tokens in user language, grounded on KB      ← N-ATLAS
   ├─ fixed i18n headline + disclaimer wrap the generated text
   └─ send reply + feedback buttons · log Interaction
```

* Core (`pipeline.py`) depends only on protocols: `ASR`, `LLM`, `Store`, `Messenger`. All adapters faked in tests.
* LLM failure / bad JSON → degrade to rules-only verdict + fixed text. ASR+LLM run behind a semaphore of 1 (one GPU).
* `speech/` and `llm/` import nothing from the rest of `kudiguard` → lifted unchanged into `natlas-gateway` later.

## 4. Serving (measured on D0 spike)

* LLM: `tosinamuda/N-ATLaS-GGUF` Q4_K_M (4.92 GB, ungated) on native Windows `llama-server` CUDA 12.4,
  `-c 4096 -ngl 99`; pass `temperature 0.1`, `repeat_penalty 1.12`, `max_tokens 250` explicitly.
  Quant kept private (no renamed redistribution). Record file SHA256.
* ASR: NCAIR1 models via `transformers` on **CPU** (avoid VRAM contention); one model per language, lazy-loaded.
* ffmpeg required (WhatsApp sends `audio/ogg; codecs=opus`).
* Stable public URL: cloudflared named tunnel or ngrok static domain; watchdog script restarts server + tunnel.

## 5. Folder structure

```
kudiguard/
├── pyproject.toml  .env.example  .gitignore  README.md
├── docs/  PLAN.md  ARCHITECTURE.md  VALIDATION.md  SUSTAINABILITY.md  DEMO.md
├── data/
│   ├── kb/topics.yaml            # topics: facts, actions, report-to (English source of truth)
│   └── eval/                     # scam scenarios (labelled), ASR clips manifest
├── scripts/  run_llm.ps1  watchdog.ps1  eval_asr.py  eval_scam.py
├── src/kudiguard/
│   ├── config.py  domain.py  i18n.py  privacy.py  prompts.py  pipeline.py  cli.py
│   ├── safety/scam_rules.py      # flag lexicons (4 langs) + scoring → Verdict
│   ├── knowledge/kb.py           # YAML loader, select by topic/flags
│   ├── speech/  base.py  audio.py  natlas_asr.py
│   ├── llm/     base.py  openai_compat.py
│   ├── storage/ base.py  sqlite.py
│   ├── channels/ whatsapp.py  web.py (+ static/voice.html)
│   └── api/app.py                # FastAPI factory + wiring
└── tests/
    ├── conftest.py               # FakeASR, FakeLLM, InMemoryStore, FakeMessenger
    ├── fixtures/                 # real webhook JSON payloads, real .ogg voice note
    ├── unit/                     # rules, kb, prompts (snapshots), audio, whatsapp parse/sig, privacy, pipeline
    └── integration/              # API end-to-end with fakes; `-m live` smoke tests for real ASR/LLM
```

## 6. Engineering rules

TDD with fakes for the core; recorded-fixture contract tests for WhatsApp; one live smoke test each for ASR and LLM
(`pytest -m live`, run daily). ruff + plain mypy. Phone numbers stored as HMAC-SHA256; raw audio not stored;
retention 90 days; `DELETE` wipes a user's rows. Secrets only in `.env`.

## 7. Schedule

| Day | Code | Human tasks (team) |
|---|---|---|
| D0 7 Oct | Scaffold; domain, rules, KB, prompts, pipeline (fakes); **spike: llama-server + GGUF running, VRAM measured, Hausa .ogg transcribed** | HF gating accepted + token; Meta app with **real SIM** + System User permanent token; CAC/IDs; recruit 30 testers |
| D1 8 Oct | Real adapters; WhatsApp (media, buttons, signature); SQLite + privacy; consent/STOP/DELETE; tunnel; KB ~15 topics | Native review of Hausa fixed strings; 10 team test runs |
| D2 9 Oct | Soft launch Hausa + English; bug fixes; ASR eval clips recorded | Testers onboarded; voice clips for fixed prompts |
| D3 10 Oct | Yoruba/Igbo beta; prompt/KB iteration; pass 50 interactions | Film demo on real phones |
| D4 11 Oct | Run WER + scam benchmarks; VALIDATION / ARCHITECTURE / SUSTAINABILITY; edit video | Testimonials; push to 80–100 |
| D5 12 Oct | Freeze 10:00, submit by 14:00; keep service up to 17 Oct | — |

## 8. Risks

| Risk | Mitigation |
|---|---|
| Meta number approval / 5-recipient test limit | Register real SIM D0; web voice page fallback |
| ASR weak on noise / code-switching | "I heard" echo; text accepted; WER reported honestly |
| Hallucinated financial advice | Deterministic verdict wins; KB-grounded; fixed disclaimer; never "safe" |
| 8 GB VRAM / single GPU | Q4 + 4K ctx on GPU, ASR on CPU, semaphore of 1 |
| Laptop downtime during integration check | Watchdog + stable tunnel; cloud GPU fallback documented |
| Licence (non-commercial, 1,000 active users / rolling 30 days) | Attribution in consent + docs; commercial-licence path in SUSTAINABILITY |
| Yoruba quality (human eval 2.69) | Labelled beta; templates dominate; short free text |

Cut: Twilio, vLLM, BM25, Docker for GPU path, strict mypy, TTS (pre-recorded prompt clips instead; MMS-TTS stretch).
