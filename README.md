# KudiGuard

**Not sure? Forward it to KudiGuard.** A WhatsApp scam verifier for Nigeria: send a voice note or forward a suspicious
message in **Hausa, Yoruba, Igbo or English** and get 🔴 / 🟠 / 🟢 with what to do and where to report, in seconds.
Built on **N-ATLAS** (LLM) and the **NCAIR1** speech models for NAIC 2026. Why and how: [docs/WHY_KUDIGUARD.md](docs/WHY_KUDIGUARD.md).

> N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation and Digital Economy and powered by
> Awarri Technologies. Used under the N-ATLaS licence (≤ 1,000 active end-users, enforced in code).

## Channels
* **WhatsApp**: voice notes, text, forwarded messages; buttons for next steps, reporting and "warn my family".
* **REST API** for banks, fintechs and NGOs: `POST /v1/check`, `/v1/check/audio`, `/v1/ask` (docs at `/docs`).
* **Web voice page** at `/` (fallback, no install).

## Results (blind test sets, docs/SPIKES.md)
| | |
|---|---|
| Scam detection, 40 unseen messages in 4 languages | 95% of scams flagged (red/orange), 86% precision |
| Knowledge grounding, 24 unseen questions | right entry in top 3: 96% |
| Speech (official NCAIR1 models, WER) | Hausa 28%, Yoruba 33%, Igbo 46% |
| Latency (RTX 4070 laptop) | text p50 5.4 s; 11-s English voice note ~7 s end to end |

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
`.venv\Scripts\python -m pytest` (921 tests, 100% coverage; `-m live` for real models) · `ruff check` · `mypy`.
Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · Validation: [docs/VALIDATION.md](docs/VALIDATION.md) ·
Sustainability: [docs/SUSTAINABILITY.md](docs/SUSTAINABILITY.md) · Demo: [docs/DEMO.md](docs/DEMO.md).

## Privacy
Phone numbers stored only as keyed hashes; message text redacted; voice notes deleted after transcription; 90-day
retention; users send `DELETE` to erase or `STOP` to stop. Guidance only, not financial advice.
