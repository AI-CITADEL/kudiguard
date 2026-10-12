# N-ATLAS Integration Evidence

**KudiGuard — NAIC 2026, Problem Statement 2 (Voice-First Access)**

Document 2 of the submission: how KudiGuard integrates the N-ATLAS language model and the NCAIR1 speech models.
Prepared 12 Oct 2026 from the repository at `***REMOVED***\nthe repository root` (branch `master` plus the same-day fixes
listed in document 3). Every number below names the file it comes from.

---

## 1. Summary

KudiGuard is a WhatsApp and web money-safety assistant: a user sends a voice note or a forwarded message in Hausa,
Yoruba, Igbo, English or Pidgin and gets a red / orange / green verdict, what to do, and where to report, in their
language. **Both AI stages are the official Nigerian models, run locally on our own server:**

| Stage | Model | How it runs | Where in the code |
|---|---|---|---|
| Speech to text (voice notes) | **NCAIR1** ASR models, one per language (Whisper-small fine-tunes) | converted once to **CTranslate2 int8**, served by faster-whisper on CPU | `scripts/prepare_asr.py`, `src/kudiguard/speech/natlas_asr.py` |
| Understanding, translation, reply writing | **N-ATLaS** LLM, `N-ATLaS-GGUF-Q4_K_M.gguf` | **llama.cpp** `llama-server` (OpenAI-compatible API) on the GPU, greedy decoding (temperature 0, seed 42) | `scripts/run_llm.ps1`, `src/kudiguard/llm/`, `src/kudiguard/prompts.py`, `src/kudiguard/engine.py` |

N-ATLAS is called on every checked message (up to seven call purposes, section 3). No message is sent to an outside
AI company: the default `KUDIGUARD_LLM_URL` is `127.0.0.1` (`docs/CLAIMS.md` P9).

## 2. Architecture at a glance

```
 WhatsApp voice note / text ─┐
 Web voice page  (/)  ───────┼─► api/app ─► pipeline.Conversation ─► engine.Engine
 Partner API /v1 (off) ──────┘   (consent, licence cap,               │
                                  rate limits, commands)              │
                                                                      ▼
   voice ─► speech/audio (ffmpeg decode, silence trim, >= 0.5 s voiced speech)
         ─► speech/natlas_asr ── NCAIR1/<lang>-ASR (CTranslate2 int8, CPU)
         ─► hallucination filter ─► asr_terms correction ("bifi ẹni" -> BVN)
   text ─► N-ATLAS ANALYSE (JSON: intent, English translation, flags)
         ─► (TRANSLATE fallback) ─► deterministic scam rules + verdict policy (safety/*)
         ─► knowledge search (packs) ─► N-ATLAS COMPOSE (reply)
         ─► [yo/ig: English draft ─► N-ATLAS REPLY_TRANSLATE ─► fidelity/negation check]
         ─► output guard (retry once) ─► fixed fallback text if anything fails
```

Source: `docs/ARCHITECTURE.md` (diagram and "Engine" paragraph). The model never decides the verdict alone and never
writes phone numbers or links; those come from rules and verified packs (`docs/ARCHITECTURE.md`, "Safety").

## 3. Where N-ATLAS is called

All model calls go through one client (`src/kudiguard/llm/openai_compat.py` behind the `LLM` interface in
`src/kudiguard/llm/base.py`). Each call carries a **purpose**, which labels metrics and sets its time budget
(`src/kudiguard/llm/base.py`, `class Purpose`):

| Purpose | What N-ATLAS does | Prompt builder | Called from |
|---|---|---|---|
| `ANALYSE` | Reads the user's message in its own language and returns JSON (schema-constrained): intent (scam check / question / greeting / other), an English translation, suspected red flags, topic | `prompts.py` (`ANALYSIS_SCHEMA`, `_chat(Purpose.ANALYSE, ...)`) | `Engine.analyse` (`engine.py`) |
| `TRANSLATE` | Fallback translation of the message to English when ANALYSE did not return one | `prompts.py` (`_chat(Purpose.TRANSLATE, ...)`) | `Engine.check` |
| `COMPOSE` | Writes the reply, grounded only on the rule verdict, the user's message and the top knowledge facts (plus one worked example) | `prompts.py` (`purpose=Purpose.COMPOSE`) | `Engine._compose` |
| `REPLY_TRANSLATE` | Translates an English-first reply into Yoruba or Igbo; the result is checked for kept steps, numbers, links and negations | `reply_translation.py` | `Engine._translate_reply` |
| `STEPS` | "What should I do?" button: numbered next steps | `prompts.py` (`_chat(Purpose.STEPS, ...)`) | `Engine.next_steps` |
| `WARN` | "Warn my family" button: a short shareable warning in the user's language | `prompts.py` (`_chat(Purpose.WARN, ...)`) | `Engine.warning` |
| `GENERAL` | Other short generations (labelled for metrics) | `llm/base.py` | — |

**Design choice backed by measurement.** The first spike (`docs/SPIKES.md` S1) rated N-ATLAS translation to English
"excellent" for Hausa, Yoruba and Pidgin, but its own flag extraction "unreliable". KudiGuard therefore uses N-ATLAS
for what it does well (understanding and translating Nigerian languages, writing the reply) and lets a deterministic
lexicon on the original text **and** the English translation decide the verdict (`safety/scam_rules.py`,
`safety/decision.py`; 13 red flags with weights 3/2/1, `docs/CLAIMS.md` P7). Model flags can only add.

**Reply strategy per language** (`KUDIGUARD_COMPOSE_STRATEGY`, default `english_first,ha=direct`;
`docs/ARCHITECTURE.md`): Hausa and English replies are composed directly by N-ATLAS; Yoruba and Igbo replies are
composed in English, checked, then translated by N-ATLAS and checked again. Measured on the 53-case dev set
(`docs/SPIKES.md` H): Yoruba key-point coverage 0.36 -> 0.58 with English-first; Hausa lost coverage English-first
(0.54 -> 0.34), so it stays direct.

## 4. How N-ATLAS is served

`scripts/run_llm.ps1` (full command):

```powershell
llama-server.exe -m models\N-ATLaS-GGUF-Q4_K_M.gguf -c 4096 -ngl 99 -np 1 --cache-ram 2048 --jinja `
                 --host 127.0.0.1 --port 8081
```

| Setting | Value | Why (source) |
|---|---|---|
| Model file | `N-ATLaS-GGUF-Q4_K_M.gguf` from Hugging Face `tosinamuda/N-ATLaS-GGUF`, SHA256 `1498091db609d8db8bf4637603fe8c705af17476a465b554c9c44e0ae2e69881` | fits an 8 GB laptop GPU (`README.md`, `docs/SPIKES.md` S1) |
| Runtime | llama.cpp release `b11476` win-cuda-12.4 | `README.md` |
| Context | `-c 4096` | VRAM 7.5 of 8 GB in use on an RTX 4070 Laptop (`run_llm.ps1` comment) |
| GPU layers | `-ngl 99` (all layers on GPU) | `run_llm.ps1` |
| Slots | `-np 1` | slot pinning would save at most ~35 ms per call (`docs/SPIKES.md` S3) |
| Prompt cache | `--cache-ram 2048` | without it ANALYSE re-processes 442 prompt tokens, ~+150 ms per call (`docs/SPIKES.md` S3) |
| Chat template | `--jinja` (the model's Llama-3 template) | `run_llm.ps1` |
| Decoding | **temperature 0, seed 42** (`DEFAULT_SEED = 42`, `src/kudiguard/llm/base.py`) | same message, same reply in all 3 runs for 90% of cases vs 9% at the old temperature 0.1 (`docs/CLAIMS.md` C1) |
| Watchdog | `scripts/watchdog.ps1` restarts N-ATLAS, KudiGuard and the tunnel if they stop or hang | `docs/DEPLOY.md` |

Every LLM call has a time budget; model steps that no longer fit the turn deadline are skipped and the rules plus
fixed text answer (`docs/ARCHITECTURE.md`, "Engine"). If N-ATLAS is down, the rules alone still flag 80% of scams
at 89% precision on a 40-message set (`docs/CLAIMS.md` C4; caveat: that set was used once for lexicon edits).

## 5. NCAIR1 speech models (ASR)

| Language | `asr_model` (from `data/languages/<code>.yaml`) | Whisper language token |
|---|---|---|
| Hausa | `NCAIR1/Hausa-ASR` | `ha` |
| Yoruba | `NCAIR1/Yoruba-ASR` | `yo` |
| Igbo | `NCAIR1/Igbo-ASR` | `yo` (Whisper has no Igbo token; the Igbo model detects `yo` on every FLEURS clip, so forcing it gives identical transcripts and is ~3 s faster, `ig.yaml` comment, `docs/SPIKES.md` H) |
| English (Nigerian-accented) | `NCAIR1/NigerianAccentedEnglish` | `en` |

* **Conversion:** `scripts/prepare_asr.py` downloads the gated NCAIR1 models (an `HF_TOKEN` for an account that
  accepted the model terms) and converts each once to CTranslate2 (`int8` by default, `--quantization float32`
  optional). It is never run on a user's request.
* **Runtime:** `faster-whisper` `WhisperModel(..., device="cpu", compute_type="int8")`, `beam_size=1`
  (`src/kudiguard/speech/natlas_asr.py`). Speech runs on CPU because N-ATLAS uses 7.5 of the 8.2 GB of VRAM
  (`docs/SPIKES.md` S1). On a Whisper-small proxy, CTranslate2 int8 took 1.6–1.9 s for a 10.9 s clip vs 5.8–8.2 s for
  transformers fp32 on 16 threads, with identical transcripts (`docs/SPIKES.md` S4).
* **Preloading:** the languages in `KUDIGUARD_PRELOAD_LANGUAGES` (default `ha,en`, `src/kudiguard/config.py`) are
  loaded at startup (`NatlasASR.preload`, called from `api/services.py`). Added on 12 Oct 2026 after the first live
  Yoruba voice note timed out while its model loaded on first use (document 3).
* **Around the model:** ffmpeg decoding of WhatsApp Opus audio, silence trim, a minimum of 0.5 s of voiced speech
  (`MIN_VOICED_SECONDS = 0.5`, `src/kudiguard/speech/audio.py`), a filter for Whisper hallucinations ("thanks for
  watching"), and per-language `asr_terms` that correct phonetically spelled bank terms in the transcript
  (`data/languages/yo.yaml`: "bifi ẹni" -> BVN; `ha.yaml`: "bibiyen" -> BVN, "akawun" -> account).

**Measured accuracy** (`docs/SPIKES.md` S6, `data/eval/asr/results/int8.jsonl`; FLEURS read sentences re-encoded as
WhatsApp audio, CTranslate2 int8 on CPU):

| Language | Clips | WER | WER with diacritics | CER | p50 latency |
|---|---|---|---|---|---|
| Hausa | 20 | 28.2% | 28.2% | 9.0% | 2.7 s |
| Yoruba | 10 | 32.7% | 64.0% | 8.4% | 2.6 s |
| Igbo | 10 (1 rejected) | 45.5% | 48.9% | 25.9% | 3.8 s |

Caveats (`docs/CLAIMS.md` C3): read speech, not real voice notes; may overlap NCAIR1 training data; English measured
on one synthetic clip only (0% WER, `docs/SPIKES.md` S5).

## 6. Measured latency

| Measure | Value | Source |
|---|---|---|
| ANALYSE call | p50 1,070 ms, p95 1,870 ms (108 calls) | `docs/SPIKES.md` S3 |
| COMPOSE call | p50 1,633 ms, p95 4,460 ms | `docs/SPIKES.md` S3 |
| Text check end to end, frozen set T v1 (366 cases) | p50 3.5 s, p95 8.3 s | `docs/CLAIMS.md` C10 |
| Same set re-run at current code (T v2) | p50 4.3 s, p95 10.9 s | `docs/CLAIMS.md` C10/C11 |
| ASR only, live pilot voice notes 12 Oct (13 notes) | p50 2,343 ms (range 1,823–3,802 ms) | `var/kudiguard.db`, `asr_ms` |
| Voice note end to end, live pilot 12 Oct (13 notes) | p50 7,475 ms, p95 17,686 ms | `kudiguard stats`; `var/kudiguard.db` |

Hardware: RTX 4070 Laptop GPU (8 GB), one request at a time; one GPU serves ~10–12 checks a minute
(`docs/CLAIMS.md` C11).

## 7. Licence: cap and attribution

The N-ATLaS licence allows at most **1,000 active end-users per rolling 30 days**. KudiGuard enforces this in code
with one admission gate shared by WhatsApp, the website and the partner API (`src/kudiguard/admission.py`;
`docs/CLAIMS.md` P1; setting `KUDIGUARD_ACTIVE_USER_CAP`). A new user beyond the cap gets a "busy" reply (API: HTTP
503 `at_capacity`, `docs/DEPLOY.md`). The partner API is off in the pilot; when on, each call must name a
pseudonymous end user (`X-KudiGuard-End-User`) who counts toward the same cap.

**Attribution wording used in the product** (WhatsApp welcome/about text, `data/languages/en.yaml` line 531, and
`README.md`):

> *N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation and Digital Economy and powered by
> Awarri Technologies.*

`README.md` adds: "Used under the N-ATLaS licence: at most 1,000 active end-users per rolling 30 days across all
channels, enforced in code (WhatsApp, website and, when switched on, the partner API)."

## 8. Evidence: one request, step by step

A real voice note from the live pilot on 12 Oct 2026 (`var/kudiguard.db`, interaction 9, web voice page, Hausa;
tester pseudonymised):

| Step | Component | What happened | Time |
|---|---|---|---|
| 1 | Web page `/` -> `POST /web/check` | Browser records the voice note and uploads it with the language `ha` | — |
| 2 | `speech/audio.py` | ffmpeg decode, silence trim, voiced-length check | decode 139 ms |
| 3 | NCAIR1/Hausa-ASR (CTranslate2 int8) | Transcript: *"wani ya kirani ya ce an toshe bibiyen ɗina."* | ASR 2,007 ms |
| 4 | `asr_terms` (`ha.yaml`, added that afternoon) | "bibiyen" -> **BVN** (a later identical typed message, interaction 10, reads "an toshe bvn dina") | — |
| 5 | N-ATLAS `ANALYSE` | English: *"Someone called and said that my account is blocked."* | |
| 6 | Rules + verdict policy | flag `account_threat` -> verdict **caution** (orange) | |
| 7 | Knowledge search + N-ATLAS `COMPOSE` (Hausa, direct) | *"Wannan na iya zama yaudara. Kada ku bi kowane hanyar da aka nuna. 1. Kar ka amsa ko danna kowace hanyar da suka bayar. 2. Tuntuɓi bankin ku ta hanyar amfani da ..."* ("This may be a scam. Do not follow any link shown. 1. Do not answer or press any link they gave. 2. Contact your bank using ...") | LLM total 4,623 ms |
| 8 | Output guard, render | Reply shown with the "I heard" transcript echo | total **6,792 ms** |

Columns `decode_ms`, `asr_ms`, `llm_ms`, `total_ms`, `transcript`, `english`, `verdict`, `flags` and `reply` of the
`interactions` table hold each step (`src/kudiguard/storage/sqlite.py`). The same flow for a text message via the
partner API (when switched on):

```http
POST /v1/check
X-API-Key: <partner key>
X-KudiGuard-End-User: c-<sha256 of the partner's customer id>
Content-Type: application/json

{"text": "...", "language": "ha"}
```

returns the verdict, flags, the reply in the user's language and the English pivot (`CheckResponse`,
`src/kudiguard/api/app.py`); `/v1/check/audio` takes a multipart voice note and runs steps 2–8.

**Reproduce:** `scripts/run_llm.ps1` (N-ATLAS), `scripts/prepare_asr.py` (NCAIR1), then
`scripts/eval_e2e.py` replays the frozen 366-case set through the real engine (`docs/VALIDATION.md`), and
`python -m pytest -m live` runs the live-model tests.
