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

Measurements behind each design choice: `docs/SPIKES.md`. Plan and milestones: `docs/PLAN.md`.
