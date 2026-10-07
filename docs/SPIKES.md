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
