# Architecture

```
channels                         core (no I/O assumptions)                 adapters
─────────                        ─────────────────────────                 ────────
WhatsApp webhook ─┐              pipeline.Conversation                     llm/openai_compat → N-ATLAS (llama.cpp)
Web page ─────────┼─► api/app ─► (consent, commands, buttons, limits) ─►   speech/natlas_asr → NCAIR1 (CTranslate2)
REST /v1 ─────────┘              engine.Engine (stateless check)           knowledge/* → packs (files, partner APIs)
                                 safety/* (rules, verdict policy, guard)   storage/sqlite (hashed, redacted)
```

* **Engine** (`engine.py`): text or transcript in → verdict + grounded answer out. Greeting shortcut → N-ATLAS
  analyse (JSON) → translate fallback → deterministic verdict (`safety/decision.py`) → knowledge search → N-ATLAS
  compose → output guard (retry once) → fixed fallback. Time-budgeted: model steps that no longer fit are skipped.
* **Conversation** (`pipeline.py`): versioned consent, STOP/DELETE/language commands, buttons, per-user ordering,
  message merging, rate limit, licence cap, never-silent failures, redacted evidence log.
* **Configuration, not code**: languages, lexicons, commands and UI text in `data/languages/*.yaml`; knowledge and
  verified contacts in `data/packs/*` (partners can plug in their own pack or API).
* **Safety**: the model never decides the verdict alone and never writes contacts; replies pass an output guard; the
  rules answer even when the model is down.
* **Scale**: one process per store today (SQLite). Next: Postgres + routing users to one instance (sticky by hash).

## How hallucination is kept out of replies

1. **Grounded prompt**: COMPOSE sees only the verdict, the user's message and the top knowledge facts and actions,
   with one worked example matched to the case (`prompts.py`). The verdict comes from the rules, not the model.
2. **Cleaning** (`prompts.clean_reply`): links, phone and account numbers, sign-offs and extra steps are removed;
   contacts are inserted verbatim by the renderer, never written by the model.
3. **Output guard** (`safety/output_guard.py`): "it is safe", "not a scam" and refusals → one retry, then fixed text.
4. **Grounding check** (`safety/grounding.py`): the reply is split into claims (steps, bullets, emoji lines; filler
   skipped) and each is compared by meaning (bge-small embeddings) with what the model was given, plus generic framing
   and the reviewed fixed advice. A claim below `KUDIGUARD_GROUNDING_THRESHOLD` (0.75), or naming a number or agency
   the evidence does not contain, makes the reply fall back to fixed text (`Outcome.FALLBACK`, issue `ungrounded`,
   no retry). Calibration: 100% precision, 89% recall, 0/28 false flags (`data/eval/grounding_claims.yaml`).
   Cost: p50 ~12 ms / p95 ~15 ms per English reply, hard-capped at `KUDIGUARD_GROUNDING_BUDGET_MS` (150), logged as
   `grounding_ms`; pack facts are embedded once at startup. It **fails open**: if it errors or runs out of time, the
   cleaned, guarded reply is sent (logged at warning). **Live for English only**: Hausa/Yoruba/Igbo would need an
   extra N-ATLAS back-translation call (~1–2 s) or a cross-lingual embedder that is not yet reliable enough (SPIKES S8);
   back-translation is available for offline evaluation (`GroundingChecker(llm, …)`).

Measurements behind each design choice: `docs/SPIKES.md`. Plan and milestones: `docs/PLAN.md`.
