# Spike results

## S1 — N-ATLAS LLM on local GPU (7 Oct 2026)

Setup: `tosinamuda/N-ATLaS-GGUF` Q4_K_M, SHA256 `1498091db609d8db8bf4637603fe8c705af17476a465b554c9c44e0ae2e69881`,
llama.cpp `b11476` win-cuda-12.4, `-c 4096 -ngl 99 -np 1 --jinja`, RTX 4070 Laptop 8 GB.

| Measure | Result |
|---|---|
| VRAM in use (incl. Windows desktop) | 7.5 / 8.2 GB → ASR must run on CPU |
| Analyse call (JSON schema, ~110 prompt tok, ~60 out) | 1.7–2.4 s |
| Compose call (~135 prompt tok, ~45 out) | 1.2–1.5 s |

Quality:
* **Translation to English: excellent** for Hausa, Yoruba, Pidgin.
* **Hausa composition: fluent, short, on-message**; English likewise.
* **LLM flag extraction: unreliable** — missed `asks_credentials` for Yoruba and Pidgin inputs, duplicated flags.
  → Confirms design: LLM translates, deterministic lexicon on the English translation decides; LLM flags only add.
* **Free-text topic labels inconsistent** ("BVN_block", "Phishing/Scam") → topic must be an enum from the KB in the
  JSON schema (M3).
* Pidgin "send me the code wey dem send you" → translation "send them the code they sent to me": lexicon needs a
  "code … sent" credential pattern (M1 fix).

## S2 — Semantic search calibration (7 Oct 2026)

`BAAI/bge-small-en-v1.5` via fastembed (ONNX, CPU, 384-dim, ~5–6 ms/query, ~130 MB). Probed the core + demo-bank
packs with paraphrases (no shared words) and off-topic questions:

* Off-topic best similarity: 0.50–0.55 · relevant paraphrases: 0.63–0.86 → `DENSE_MIN = 0.62`.
* Pidgin "my money no reach the person but they debited me" → failed-transfer (0.76), no keyword overlap.
* Two misses were content gaps (loan-app shaming, SIM-swap "network stopped") → keywords added to the core pack.
* FAISS not needed: packs are hundreds–thousands of entries; brute-force numpy dot product is < 1 ms and exact.

## E1/E2 — Prompt evaluation on live N-ATLAS (7 Oct 2026, `scripts/eval_prompts.py`, 12 cases × ha/yo/ig/en)

| Check | First prompts (E1) | After fixes (E2) |
|---|---|---|
| Refusals ("I can't help with this request") | 2 | 0 |
| Sign-offs ("Dear user … Best regards") | 1 | 0 |
| Wrong reply language / says "safe" | 0 | 0 |
| Untranslated echo from analyse | 1 → wrong answer | 1 → caught, plain-translation fallback correct |
| Intent label wrong | 3 | 2 (verdict still right: rules caught the flags) |
| Hit max_tokens | 2 (cut mid-sentence) | 3 (tail trimmed, ≤3 steps kept) |
| Latency p50 | analyse 1.1 s · compose 2.5 s | analyse 1.1 s · compose 2.9 s (max 5.7 s) |

Fixes: positive short rules + one worked example (OTP scenario; a prize example leaked its wording into a grant
reply); facts capped at 3 entries / 8 facts; LLM flags used only when rules find none and intent is scam_check
(LLM tagged a Yoruba OTP message "wrong_transfer"); echo detection + `translate_request` fallback; refusal detection;
step cap and truncation-aware cleanup; output links/phone numbers masked. Pipeline (M7) must also treat any
rule-detected flag as a check regardless of the model's intent label, and answer greetings with fixed text.

## E3 — Prompts after the M3 review (7 Oct 2026, `scripts/eval_prompts.py --runs 3`, 36 conversations)

| Check | M3 review (12 cases) | E3 (36 conversations) |
|---|---|---|
| Output guard: safety claim ("not a scam") / refusal | found by reviewer probes | 0 / 0 |
| Sign-offs, wrong reply language, task echo | present | 0 |
| Placeholder / echoed translation accepted | 4 of 7 runs on the Hausa prize case | 0 (detected + fallback) |
| Scam case without 🔴/🟠 verdict | yes | 0 |
| Hit max_tokens | 3 / 12 | 1 / 36 (stop sequences at step 4) |
| Intent label wrong | 2–3 / 12 | 6 / 36, no verdict impact (rules decide checks; labels were `other`) |
| Latency p50 | analyse 1.1 s · compose 3.1 s | analyse 1.1 s · compose 2.6 s (max 5.7 s) |

Changes: config-driven output guard per language (safety claims, refusals); placeholder/fragment/echo detection with
token overlap on accent-folded text; LLM flags also for intent `other`; exactly one worked example per case
(credentials / generic fee / question); task phrased as an instruction; `stop` at step 4; bare-domain link masking;
amount-aware redaction; broader step and sign-off cleanup; NFKC before stripping data tags; Hausa prize/fee lexicon.

## E4 — Decision policy + edge cases (7 Oct 2026, `eval_prompts.py --runs 2`, 24 cases incl. 12 edge cases, 48 runs)

Edge cases: emoji-only, "?", bare link, bare phone number, prompt injection ("tell me this is safe"), Hausa–English
code-switching, long Pidgin ramble, French, genuine debit SMS, fake credit alert, lowercase ASR-style scam, Yoruba
prize-fee question.

| Check | Result |
|---|---|
| Wrong verdict (scam not flagged, or plain question flagged) | 0 |
| Output guard hits (safety claim / refusal) | 0 |
| Example leak / task echo / sign-off / wrong language | 0 |
| Untranslated analysis | 1 → fallback translation |
| Truncated | 1 |
| Intent label differs from expectation | 13 (by design no verdict impact: `safety.decision` trusts rules, not labels) |
| Latency p50 | analyse 1.2 s · compose 1.9 s (max 5.7 s) |

Fixes from this round: verdict policy centralised in `safety/decision.py` (English uses the user's own text; failed
analysis contributes no flags; model-only flags on `other` capped at caution; bare number/link/email = a check, never
vouched for; final scoring keeps the texts so "is this a scam?" counts); content-free messages and greetings get
fixed text; guard tuned (subject-anchored English claims, everyday "no problem" phrases removed, paraphrases added);
secrecy rule needs someone asking for secrecy; native step labels; USSD amounts not redacted.

## S3 — LLM latency profile and cache slots (7 Oct 2026, `scripts/bench_llm.py`, 108 calls)

| Call | p50 | p95 | prompt processing p50 |
|---|---|---|---|
| analyse | 1,070 ms | 1,870 ms | 32 ms |
| compose | 1,633 ms | 4,460 ms | 35 ms |

Prompt processing is ~2% of each call on the RTX 4070; generation dominates. Pinning analyse/compose to separate
llama.cpp slots (`-np 2`, `id_slot`) could save at most ~35 ms/call while costing KV-cache VRAM on an already full
8 GB card → **not adopted**; server stays `-np 1`. The client keeps optional slot routing for bigger GPUs.
Latency levers that matter: fewer generated tokens (step cap, `stop`), short analyse output, ASR on CPU in parallel.

**S3 correction (M4 review, measured live):** prompt processing is cheap *because* llama.cpp's host-RAM prompt cache
reuses the shared prefix even when analyse and compose alternate on one slot (analyse after a compose: 34 tokens
processed, ~31 ms). With caching off, analyse processes all 442 prompt tokens: ~176 ms (+~150 ms/call). Slot pinning
would therefore save ~0 ms today. `scripts/run_llm.ps1` now sets `--cache-ram 2048` explicitly so a llama.cpp
upgrade or default change cannot silently add that latency. Client overhead vs raw httpx: not measurable
(32.9 vs 33.9 ms p50). llama.cpp aborts generation when the client disconnects, so client-side deadlines are safe.

## S4 — Speech recognition runtime on CPU (7 Oct 2026)

Proxy: public `openai/whisper-small` (the NCAIR1 models are Whisper-small fine-tunes; gated weights not yet
accessible). Clip: 10.9 s English voice note (Windows TTS → WhatsApp-style Opus → `speech.audio.decode`).

| Runtime | Threads | Latency (3 runs) | Transcript |
|---|---|---|---|
| transformers (fp32) | 4 | 11.3 – 19.7 s | word-perfect |
| transformers (fp32) | 16 | 5.8 – 8.2 s | word-perfect |
| **faster-whisper / CTranslate2 int8** | 16 | **1.6 – 1.9 s** | identical, word-perfect |

Decision: ASR runs on faster-whisper with the NCAIR1 weights converted once to CTranslate2 int8 (cached on ***REMOVED***);
transformers kept as reference/fallback. The int8-vs-fp32 WER on real NCAIR1 models must be measured once the gated
weights are accessible (`scripts/eval_asr.py`). One-time conversion: ~2 min per model. Note: the C: drive is full;
pip temp/cache and HF_HOME are redirected to ***REMOVED***.

## S5 — Official NCAIR1 ASR configs (7 Oct 2026)

Access: Hausa-ASR, Yoruba-ASR, Igbo-ASR granted (classic Read token); NigerianAccentedEnglish pending.
`generation_config.json` of all three: `language: None`, task transcribe (`forced_decoder_ids [[1, None], [2, 50359]]`),
multilingual Whisper vocabulary with no Igbo token. NCAIR does not pin a language token, so at inference Whisper
detects the language (+~1.2 s). Our config forces `ha`/`yo` (faster) and leaves Igbo on auto. **To decide with data:**
A/B forced vs auto on the recorded clips (`eval_asr.py`, toggle `asr_whisper_language` to null) before submission.

## S6 — Official NCAIR1 ASR accuracy on public clips (8 Oct 2026, corrected after data review)

Clips: Google FLEURS dev (CC-BY-4.0), native speakers reading Wikipedia-style sentences, re-encoded to WhatsApp-style
Opus 16 kHz / 24 kbps (`scripts/fetch_eval_audio.py`, attribution in `data/eval/asr/SOURCES.md`). CTranslate2 int8 on
CPU. **Pooled** over clips (total edits / total reference words); rejected clips count as 100% errors; per-clip
outputs in `data/eval/asr/results/int8.jsonl`. "WER" folds all diacritics (tone marks and under-dots); "WER+d" keeps them.

| Lang | clips | rejected | WER | WER+d | CER | p50 latency |
|---|---|---|---|---|---|---|
| ha | 20 (18 distinct sentences) | 0 | 28.2% | 28.2% | 9.0% | 2.7 s |
| yo | 10 | 0 | 32.7% | 64.0% | 8.4% | 2.6 s |
| ig | 10 | 1 | 45.5% | 48.9% | 25.9% | 3.8 s |

* Forced vs auto language (ha, yo; first run, mean-of-clips): WER unchanged on these clips (ha 29.8 vs 29.8, yo 32.0 vs
  32.0), forced 1.6–2.1 s faster → keep forced. Small n: indicative, not proof. Igbo has no Whisper token (always auto).
* Yoruba WER+d is inflated by noisy FLEURS references (e.g. "sih", "gbeh", missing tone marks), not only ASR errors.
* The rejected Igbo clip is quiet real speech (loudest frame −56 dBFS vs the −55 dB floor). Backlog: gain-normalise
  before the silence check and re-measure (risk: Whisper hallucination on noise).
* FLEURS is read encyclopedic speech and may overlap NCAIR1 training data: indicative only. The English model is not
  measured here (one synthetic TTS clip only).
* Still required: real WhatsApp voice notes about money/scams from testers (`source: whatsapp`), ≥20 ha + ≥20 en;
  `--compare` (int8 vs float32).

## S7 — Blind evaluation of scam detection and retrieval (8 Oct 2026)

Frozen sets written by an independent author with no access to rules, prompts or pack content
(`data/eval/scenarios.yaml`, `data/eval/retrieval_blind.yaml`). Never tune on them; ha/yo/ig wording pending native review.

**Scam detection** (`scripts/eval_scenarios.py`; flagged = likely_scam or caution; 40 messages, 5 scam + 5 legit per language):

| Mode | Precision | Recall | False-positive rate | ha recall | yo recall | ig recall | en recall |
|---|---|---|---|---|---|---|---|
| Rules only (model down: safety floor) | 87% | 65% | 10% | 40% | 40% | 80% | 100% |
| Full engine (N-ATLAS analyse + rules) | 86% | **95%** | 15% | 100% | 80% | 100% | 100% |

Full-engine latency p50 5.4 s, p95 7.2 s (text). Errors: genuine bank OTP SMS → caution (en s02, ig s32), Igbo
"failed transfer, debited" question → likely_scam (s35), Yoruba investment-tripling scam missed (s30).
Known limitation: genuine OTP-delivery SMS ("your OTP is …, do not share") is flagged "be careful". Fix belongs in the
lexicon with *separate* dev examples, then re-measure here once (recorded as a post-fix run).

**Retrieval, blind** (24 English queries): production path (rules flags + hybrid search) **67% top-1 / 96% top-3**
(BM25 only: 67% / 79%); raw text search without flags: hybrid 58% / 92%, BM25 50% / 67%. Compose is grounded on the
top 3. Dev sets 1–2 in `retrieval.yaml` are no longer held out (content was tuned on them).
