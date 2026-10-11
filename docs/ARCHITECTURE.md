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
  Reply strategy per language (`reply_translation.py`, `KUDIGUARD_COMPOSE_STRATEGY`, default
  `english_first,ha=direct`): **Yoruba and Igbo are English-first**: composed and checked in English (every English
  check sees the text), then translated, guarded again in the language and checked for fidelity (steps, numbers,
  links, length, echo, negation kept); a failure falls back to fixed text in the language. **Hausa and English are
  direct** (composed in the language). Why (docs/SPIKES.md H, dev set, same session): English-first raised Yoruba
  coverage 0.36 → 0.56 and Igbo 0.37 → 0.42 (Igbo copying the worked example 42% → 8%) but cut Hausa 0.54 → 0.34,
  for +1.6–2.3 s p50. Direct replies get only the output guard: no fidelity or negation check, so native review of
  Hausa replies matters (11% of measured translations flipped a negation; a direct reply can err the same way unseen).
* **Conversation** (`pipeline.py`): versioned consent, STOP/DELETE/language commands, buttons, per-user ordering,
  message merging, rate limit, licence cap, never-silent failures, redacted evidence log.
* **Configuration, not code**: languages, lexicons, commands and UI text in `data/languages/*.yaml`; knowledge and
  verified contacts in `data/packs/*` (partners can plug in their own pack or API).
* **Safety**: the model never decides the verdict alone and never writes contacts; replies pass an output guard; the
  rules answer even when the model is down.
* **Scale**: one process per store today (SQLite). Next: Postgres + routing users to one instance (sticky by hash).

Measurements behind each design choice: `docs/SPIKES.md`. Plan and milestones: `docs/PLAN.md`.
