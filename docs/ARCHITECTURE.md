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
  links, length, echo, negation kept); a failure falls back to fixed text in the language. B's shadow classifier and
  grounding score the English draft; grounding in enforce mode trims it before translation. The "What should I do?"
  and "Warn my family" buttons follow the same strategy. **Hausa and English are direct** (composed in the
  language). Why (docs/SPIKES.md H, dev set, shipped code `h2_final` vs direct): Yoruba coverage 0.36 → 0.58; Igbo
  0.37 → 0.33 (no coverage gain), but Igbo stops copying the worked example (42% → 8%) and gets the English safety
  checks (17% of Igbo replies fall back to fixed text, all real negation flips); Hausa lost coverage English-first
  (0.54 → 0.34), so it stays direct. Cost: +1.7 s (yo) / +1.3 s (ig) p50 and +42% / +35% model time per turn. Direct replies get only the output guard: no fidelity or negation check, so native review of
  Hausa replies matters (11% of measured translations flipped a negation; a direct reply can err the same way unseen).
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
   the money back, reply with / enter the code in their link). Matched per clause and only as instructions: all
   words before the verb must be lead-in ("please", "make sure you", "you can safely", "I recommend you", "unless
   you"); a negation there ("do not", "never", "kada") or after the verb ("to no one") cancels it. So "Don't worry,
   just send them the OTP" and "Your account will be blocked unless you send the OTP" are blocked; "Ask your bank if
   they can refund it" and "They said you should send the OTP" are not. A hit → one warmer retry, then fixed text.
   The permanent probe set `tests/fixtures/guard_probes.yaml` (+ D's English set) pins ~680 labelled phrasings, and a generated grammar (`tests/guard_grammar.py`, ~2,900
   sentences labelled by construction) must stay 100% per cell.
   Promised outcomes and threats stated as fact ("you will get your money back", "your account will be blocked")
   are logged only (blocking in enforce mode). This is the deterministic safety net; ha/yo/ig patterns are DRAFT
   pending native review.
4. **Learned guard layer** (`safety/guard_classifier.py`, English; Nigerian Pidgin only because it arrives as `en`
   and is scored by the same English model): logistic regression on the bge-small embedding of each reply
   sentence/clause plus lexical cues catches unsafe advice the patterns were not written for; it blocks only units
   the regex helpers do not mark as negated, quoted, reported, conditional or descriptive (up to 8 per reply, the
   most suspicious first, one embedding batch). `KUDIGUARD_GUARD_CLASSIFIER_MODE`: **shadow (default)** scores after
   sending and logs; **enforce** treats a flag like a guard hit (retry, then fixed text);
   `KUDIGUARD_GUARD_CLASSIFIER_PIDGIN_MODE` can differ for Pidgin. 250 ms budget with a short bounded queue, then the
   regex verdict stands (fail-open counts by reason in the logs). Honest numbers (SPIKES S8, round 11): the round-10
   version scored 83% recall / **11.8% false positives** on the independent English held-out set (Pidgin 77% /
   14%), against 1.6% claimed on a leaky dev split; the current version is trained on a frozen set with a grouped
   split and an out-of-fold threshold. Shipped operating point **0.85**: out-of-fold 84% recall / 3.7% false
   positives, clean dev 85% / 0%, independent held-out en+pcm ~70% / 2.3% (verifier); the "strict" point 0.945
   (out-of-fold 74% / 1.8%) is recorded beside it and selectable with `KUDIGUARD_GUARD_CLASSIFIER_THRESHOLD`. The artefact is validated at startup (checksum, features, embedding model and
   size). ha/yo/ig stay regex-only (held-out recall 0-21%).
5. **Grounding check** (`safety/grounding.py`): each claim of an English reply is compared by meaning (bge-small)
   with same-polarity evidence sentences (facts, actions, verdict, flags, generic framing, reviewed fixed advice);
   its numbers and agency names must appear in the evidence or the user's message. The user's message supports a
   claim's meaning only when the claim is attributed ("They said…"), so a reply cannot pass by repeating the scam.
   `KUDIGUARD_GROUNDING_MODE`: **shadow (default)** runs the check *after* the reply is sent (a background task:
   no added latency), logs it and passes it to the engine's `on_grounding(turn_id, report, ms)` hook;
   **enforce** checks inline, records `grounded` / `unsupported_claims` / `grounding_ms` on the answer, drops an
   unsupported numbered step or falls back to fixed text (`issue ungrounded`, no retry). Enforce needs ≥ 100
   held-out real N-ATLAS English replies showing ≤ 5% false replacements first: on the reviewer's 13 realistic
   replies, 7 would have been replaced (SPIKES S8). Check cost: ~32–35 ms per 8-sentence English reply (more under
   CPU load; flat in message length up to 2,000 chars), hard-capped at `KUDIGUARD_GROUNDING_BUDGET_MS` (150) on one
   dedicated worker (shadow skips when it is busy; enforce waits within the budget; nothing is queued); pack facts
   are embedded once at startup (~3–4 s for the default packs). It **fails open**: on error or time-out the cleaned,
   guarded reply is sent (warning logged). **English only live**: Hausa/Yoruba/Igbo would need an extra N-ATLAS
   back-translation (~1–2 s) or a cross-lingual embedder not yet reliable enough; back-translation is available
   offline (`GroundingChecker(llm, …)`).

Measurements behind each design choice: `docs/SPIKES.md`. Plan and milestones: `docs/PLAN.md`.
