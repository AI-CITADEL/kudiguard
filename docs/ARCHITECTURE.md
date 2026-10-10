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

## How hallucination and unsafe advice are kept out of replies

1. **Grounded prompt**: COMPOSE sees only the verdict, the user's message and the top knowledge facts and actions,
   with one worked example matched to the case (`prompts.py`). The verdict comes from the rules, not the model.
2. **Cleaning** (`prompts.clean_reply`): links, phone and account numbers, sign-offs and extra steps are removed;
   contacts are inserted verbatim by the renderer, never written by the model.
3. **Output guard** (`safety/output_guard.py`, patterns in `data/languages/*.yaml`): "it is safe" / "not a scam",
   refusals, and **unsafe instructions** (share the code/PIN/BVN, pay the fee, click the link, install the app, send
   the money back) unless negated earlier in the sentence ("do not share", "kada", "má ṣe") or reported ("they said").
   A hit → one warmer retry, then fixed text. This is the deterministic safety net. ha/yo/ig patterns are DRAFT
   pending native review.
4. **Grounding check** (`safety/grounding.py`): each claim of an English reply is compared by meaning (bge-small)
   with same-polarity evidence sentences (facts, actions, verdict, flags, generic framing, reviewed fixed advice);
   its numbers and agency names must appear in the evidence or the user's message. The user's message supports a
   claim's meaning only when the claim is attributed ("They said…"), so a reply cannot pass by repeating the scam.
   `KUDIGUARD_GROUNDING_MODE`: **shadow (default)** records `grounded` / `unsupported_claims` / `grounding_ms` on the
   answer and in the logs without changing the reply; **enforce** drops an unsupported numbered step, or falls back
   to fixed text (`issue ungrounded`, no retry). Enforce needs ≥ 100 held-out real N-ATLAS English replies showing
   ≤ 5% false replacements first: on the reviewer's 13 realistic replies, 6 would have been replaced (SPIKES S8).
   Cost: ~30–55 ms per 8-sentence English reply depending on CPU load (p95 < 70 ms; flat in message length up to
   2,000 chars: the message is embedded only for attributed claims), hard-capped at
   `KUDIGUARD_GROUNDING_BUDGET_MS` (150) on one dedicated worker that is skipped, not queued, when busy; pack facts
   are embedded once at startup (~3–4 s for the default packs). It **fails open**: on error, time-out or busy, the
   cleaned, guarded reply is sent (warning logged). **English only live**: Hausa/Yoruba/Igbo would need an extra
   N-ATLAS back-translation (~1–2 s) or a cross-lingual embedder not yet reliable enough; back-translation is
   available offline (`GroundingChecker(llm, …)`).

Measurements behind each design choice: `docs/SPIKES.md`. Plan and milestones: `docs/PLAN.md`.
