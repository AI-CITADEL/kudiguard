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
