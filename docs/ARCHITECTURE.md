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
* **Partner REST API** (`/v1`, `api/app.py`): not mounted unless `KUDIGUARD_PARTNER_API=true` (off in the pilot).
  When on, each call names a pseudonymous end user (`X-KudiGuard-End-User`, phone-shaped ids refused), who is
  admitted under the same N-ATLaS licence cap as web sessions (one lock, in-flight users counted) and WhatsApp users.
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

6. **Malicious use** (`docs/ABUSE.md`): before the model, an input screen answers crisis messages (help numbers),
   abuse and scam-tool requests with fixed text and answers prompt injections from the rules alone; after it, a
   reply screen blocks persona claims, prompt leaks and long echoes of the input. Partner packs are screened at
   load, the pattern library and donations pass a learning gate, and new users, queues, notices and voice notes are
   rate-limited. Every defence is `off | shadow | enforce`, phrases per language under `abuse:` in the YAMLs.

## Advice quality (what reaches COMPOSE)

Settings in `.env.example` ("Advice quality"); code in `knowledge/service.py` (cut-off), `engine.py`
(`AdvicePolicy`, `_genuine_notice`, `_unless_copied`) and `prompts.py`.

* **Relevance cut-off** (`KUDIGUARD_KNOWLEDGE_MIN_RELEVANCE_DENSE=0.62`, `..._BM25=0.60`; `0`/`0` = off). Search hits
  below it never reach COMPOSE; red-flag entries always do (they match the message's flags, not its words). A
  question left with no facts gets `prompts.NO_FACTS_TASK` (honest general guidance plus one short clarifying
  question) and a matching worked example, never an unrelated entry. Calibrated on the dev retrieval sets only
  (`data/eval/retrieval.yaml`, `retrieval_money.yaml`, and the English `answers.yaml` cases with their `from`
  entries), with all four production packs (core, money-skills, genuine-notices, situations; re-calibrated after the
  last two were added: scores rise as the corpus grows): the highest level that keeps every correct top-3 hit, dense
  0.62 (drops 3% of wrong hits) and BM25 0.60 (drops 6%). Cosine separates related from unrelated poorly in the
  0.6-0.7 band, so the level only removes clearly unrelated entries ("how do I repair my generator" gets no entry);
  ranking still decides among related ones. Dev sets (97 English queries, rules + all packs) before -> after: hybrid
  top-1 54.6% -> 54.6%, top-3 80.4% -> 80.4%, wrong entries passed to COMPOSE 205/291 -> 201/287; BM25 unchanged
  accuracy, wrong entries 201/289 -> 194/282. (The dev want-lists predate the new packs, so a better sit-*/notice-*
  hit counts as wrong there.)
* **Entry model answers** (`example:` inline in `topics.yaml`, or `examples.yaml` in the pack: `id: text`). The top
  entry's example is shown to COMPOSE labelled "a model answer for a similar situation; do not copy it, adapt it to
  the user's message", and replaces the fixed worked example (one example per request). Copy guard: a reply (the
  English draft for English-first languages) sharing more than `KUDIGUARD_COMPOSE_EXAMPLE_COPY_MAX` (0.5) of its word
  trigrams with the example is composed once more without it. `KUDIGUARD_COMPOSE_EXAMPLES=false` turns it off. The
  example is used only when its entry is the top search hit at `KUDIGUARD_COMPOSE_EXAMPLE_MIN_RELEVANCE_DENSE` /
  `_BM25` (0.70, above the cut-off: keeps 85% / 81% of correct and 51% / 55% of wrong top-3 hits on the dev sets).
  0.5 is a starting level, not yet measured on live replies (no pack had examples when it was set).
* **Genuine notices** (entries `notice-*`, `KUDIGUARD_NOTICE_PREFIX`; empty = off). Notice keywords overlap scam
  wording (credit alert, reversal, BVN update, delivery), so retrieval alone never lowers a verdict. Model-only
  suspicion is lowered only when ALL hold: the best search hit is a notice at or above
  `KUDIGUARD_NOTICE_MIN_RELEVANCE_DENSE=0.80` (`..._BM25=0.84`; on the dev sets 7% / 5% of wrong hits reach these
  levels); the rules (`scam_rules.assess`) found no flag at all; the message has no link, email, phone or account
  number (`prompts.strip_contacts`; an OTP SMS with its code therefore keeps its verdict); the model's flags are only
  urgency, impersonation or secrecy (no request to send money, codes or details). Then: no red flags (the notice's
  own check; red-flag entries dropped) when ANALYSE's `user_initiated` says the user started it, else at most
  caution. Otherwise the model's verdict stands. Typical look-alike scams are unit-tested not to be lowered.
* **Copy guard on the fixed worked examples too**: a reply that shares more than the copy level of its trigrams with
  the fixed worked example it was shown (live: a caution reply copied "Real offers do not ask you to pay first" word
  for word) is composed once more with no example at all.
* **Rules: a credential REQUEST, not a warning** (`data/languages/en.yaml`, `asks_credentials`): the request verb must
  not be negated ("do not / don't / never / ever / nor / Pidgin no go" just before it), so a bank's own "Your OTP is
  123456. Do not share it with anyone" no longer flags; "send me the OTP", "read me the code that came to your phone",
  "share your PIN to unlock" and Pidgin "if I no send the OTP" still do. First-person wrong transfers ("sent money to
  my account by mistake", "reverse my money to her", "send back the excess") were added to `wrong_transfer`.
  Rules-only, before -> after (any flag on a message the user asks about): answers.yaml scams 14/20 -> 15/20,
  non-scams flagged 2/21 -> 2/21; prompts.yaml scams 17/18 -> 17/18 (asks_credentials 5 -> 5), non-scams 2/16 ->
  2/16; own-OTP / wrong-transfer probes: scams 3/5 -> 5/5, genuine notices flagged 3/4 -> 1/4 (the remaining one is
  `impersonation` on a bank name).
* **Situational packs in general search** (`uncued_weight` in pack.yaml; 0.9 for genuine-notices and situations, 1 =
  off): when the query has no red flag and contains none of the entry's own keywords, those packs' hits count 0.9 of
  their relevance when the merged hit list is ranked, so general questions ("keep money aside every month") stay with
  general answers. Dev sets, all four packs, rules + search, top-1 / top-3 before -> after: production pipeline BM25
  67%/91% -> 83%/96%, hybrid 72%/93% -> 89%/96%; money-skills BM25 59%/95% -> 77%/100%, hybrid 77%/91% -> 86%/95%;
  core-only sets unchanged. The 15 accepted sit-*/notice-* alternatives: 15/15 in the top 3 with BM25, 13/15 hybrid.
* **Impersonation needs a claim**: "customer care/service/support" flags only as a claim ("this is X customer care",
  "he said he is from customer care", "him be customer care") or as the one who contacted the user ("customer care
  called/asked/DMed"), not in a question ("what is my bank's customer care number?"). Rules-only on the dev sets:
  impersonation on scams answers.yaml 3/20 -> 3/20, prompts.yaml 5/18 -> 5/18; customer-care questions 2/2 -> 0/2.
* **Rule fixes after review (53bdeed)**: a negated "do not share" that makes an exception ("... except our agent",
  "give it to him only") and "read the code to the officer" still flag `asks_credentials`; "send back the excess / the
  balance" needs a wrong-transfer cue (by mistake, in error, wrong account...); named or headed customer care ("GTBank
  customer service:", "Customer care desk.", "message from customer service", "contact our customer care") and "this
  is / I am from / ... here / Regards, <bank>" count as impersonation claims, while "what is my bank's customer care
  number?" and "how do I reach GTBank customer care?" stay clean. Probe table: `tests/unit/test_rules_review.py`.
  Rules-only on the dev sets, master -> new: answers.yaml scams 14/20 -> 15/20, non-scams 2/21 -> 2/21; prompts.yaml
  17/18 -> 17/18, 2/16 -> 2/16 (impersonation and asks_credentials recall unchanged).
* **Pidgin, stabilised**: Pidgin input with a scam verdict gets a Pidgin worked example; a reply to Pidgin input that
  reads as Hausa, Yoruba or Igbo (`prompts.reads_as_hausa_yoruba_igbo`) is composed once more in English.
* **Pidgin replies** (`KUDIGUARD_PIDGIN_REPLIES`): Pidgin arrives as language `en`; when
  `guard_classifier.is_pidgin` detects it, COMPOSE is asked to "Reply in Nigerian Pidgin".
* **Stock answer fix**: the fixed question example (NIN, "dial *346#") was the only worked example for every
  question; with no matching facts N-ATLAS copied it (a stolen-card question got "*346#"). Questions without facts now
  get the no-facts example, and an entry example replaces the fixed one.

Measurements behind each design choice: `docs/SPIKES.md`. Plan and milestones: `docs/PLAN.md`.
