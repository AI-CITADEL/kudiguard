# KudiGuard

**Not sure? Forward it to KudiGuard.** A WhatsApp scam verifier for Nigeria: send a voice note or forward a suspicious
message in **Hausa, Yoruba, Igbo or English** and get 🔴 / 🟠 / 🟢 with what to do and where to report, in seconds.
Yoruba and Igbo are in beta, and no native speaker has reviewed the Hausa, Yoruba or Igbo texts yet (docs/VALIDATION.md).
Built on **N-ATLAS** (LLM) and the **NCAIR1** speech models for NAIC 2026. Why and how: [docs/WHY_KUDIGUARD.md](docs/WHY_KUDIGUARD.md).

> N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation and Digital Economy and powered by
> Awarri Technologies. Used under the N-ATLaS licence: at most 1,000 active end-users per rolling 30 days across all
> channels, enforced in code (WhatsApp, website and, when switched on, the partner API).

## Channels
* **WhatsApp**: voice notes, text, forwarded messages; buttons for next steps, reporting and "warn my family".
* **Partner REST API** (`POST /v1/check`, `/v1/check/audio`, `/v1/ask`): **off in the pilot**. When switched on,
  every call names its end user with a pseudonymous id, and each end user counts toward the licence cap
  (docs/DEPLOY.md, "Partner API").
* **Web voice page** at `/` (fallback, no install).

## Results so far (small test sets; every number, source and caveat: [docs/CLAIMS.md](docs/CLAIMS.md))
| | | Caveat |
|---|---|---|
| **Frozen end-to-end set T v1** (366 cases, 5 languages; run once at 8e15852) | **pass 85%**: scams flagged 178/196 (91%), genuine messages wrongly flagged 35/96 (36%), questions answered 72/74; p50 3.5 s, p95 8.3 s | the official headline; ha/yo/ig cases DRAFT |
| Advice on T v1, judged blind by 4 LLM judges | good 94, acceptable 108, poor 131, **harmful 33** of 366; 46% of key advice points conveyed | 30 of the 33 harmful replies are genuine messages called scams |
| Advice fixes on dev set dev2 (130 cases), before → after | judged good 10 → 26, advice points 32% → 46%, harmful 25 → 23; scams flagged 96% → 96%; p50 3.6 → 4.7 s | dev set, not independent; ~22 of 45 real notices still judged harmful |
| T v2 (5503deb) | pending | re-run after fixes informed by T v1: not independent |
| Same message, same reply (shipped code) | 90% identical over 3 runs (temperature 0 + seed; 9% at the old temperature 0.1) | 53-case dev set; Yoruba (English-first) 64% |
| Knowledge retrieval, 24 independent questions | right entry in top 3: 96% (top 1: 62%) | English questions only |
| Speech (official NCAIR1 models, word error rate) | Hausa 28%, Yoruba 33%, Igbo 46% | 10-20 read FLEURS sentences per language, not voice notes |
| Scam detection by the rules alone (model down) | 80% of scams flagged, 89% precision | 40 messages; set already used once for lexicon edits |
| Answer covers the key points (0-1) | en 0.64 · ha 0.46 · yo 0.58 · ig 0.33 | 53-case dev set, ha/yo/ig AI-drafted |
| Unsafe advice caught in English/Pidgin replies | 5% → 34% (rule guard) → ~70% with the learned layer at threshold 0.85, 2.3% false alarms (verifier's held-out en+pcm set; 3.7% out of fold on our 1,135 training items) | held-out set partly used; learned layer logs only in the pilot |
| Speed (RTX 4070 laptop, text, commit 6da290d) | p50 4.0 s, p95 8.2 s (English p50 2.8 s; Yoruba +1.6 s); current master slower (dev2: p50 4.7 s, p95 10.2 s) | one request at a time; voice timed once (~7 s, English) |

Not yet measured: the full engine on the 40-message scam set (to be re-run live), T v2, real users and
native-speaker review (after the pilot). Main known weakness: real bank notices called scams (36% on T v1).

## Run (Windows, NVIDIA GPU ≥ 8 GB)
Tools and models (not in git):
* `tools\llama\`: llama.cpp [b11476 win-cuda-12.4](https://github.com/ggml-org/llama.cpp/releases/tag/b11476) (`llama-server.exe` + CUDA DLLs).
* `models\N-ATLaS-GGUF-Q4_K_M.gguf`: from [`tosinamuda/N-ATLaS-GGUF`](https://huggingface.co/tosinamuda/N-ATLaS-GGUF),
  SHA256 `1498091db609d8db8bf4637603fe8c705af17476a465b554c9c44e0ae2e69881`.
* `tools\ffmpeg\`: any LGPL ffmpeg build (or ffmpeg on PATH). `tools\cloudflared\cloudflared.exe`: 2026.9.1.

```powershell
python -m venv .venv; .venv\Scripts\pip install -e ".[dev,semantic,asr]"
copy .env.example .env               # fill in; KUDIGUARD_HASH_SECRET is required
.venv\Scripts\python scripts\prepare_asr.py   # one-time: NCAIR1 speech models (HF_TOKEN; accept model terms)
.venv\Scripts\python -c "from kudiguard.knowledge.embeddings import FastEmbedder as F; F(cache_dir='models/fastembed').embed_documents(['warm up'])"
.\scripts\watchdog.ps1               # N-ATLAS + KudiGuard + HTTPS tunnel; restarts what stops or hangs
```
Then set the tunnel URL + `/webhooks/whatsapp` as the webhook in Meta. Details: [docs/DEPLOY.md](docs/DEPLOY.md).

## Develop
`.venv\Scripts\python -m pytest` (2,963 tests, 100% line + branch coverage; `-m live` for real models) · `ruff check` · `mypy`.
Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · Validation: [docs/VALIDATION.md](docs/VALIDATION.md) ·
Sustainability: [docs/SUSTAINABILITY.md](docs/SUSTAINABILITY.md) · Demo: [docs/DEMO.md](docs/DEMO.md).

## Privacy
Phone numbers stored only as keyed hashes; most numbers and emails in messages masked (names are kept); voice notes
deleted after transcription unless the user opts in to donate them; messages deleted after about 90 days and scam
examples within 2 years (configurable, never longer; the purge runs every 6 hours while the server is on); users send
`DELETE` to erase their WhatsApp data or `STOP` to stop. Guidance only, not financial advice.
