# Validation

Evidence for NAIC 2026: real usage by people outside the team, plus blind accuracy tests. Numbers below are filled
in from `kudiguard stats --exclude-users team.txt` (team testers' numbers, one per line, are hashed and excluded).

## Real usage (to fill in after the pilot)
| Measure | Value |
|---|---|
| Period | _from – to_ |
| Interactions (WhatsApp + web) | _n_ (target ≥ 50) |
| Unique WhatsApp users / web sessions | _n_ / _n_ |
| Languages | ha _n_ · yo _n_ · ig _n_ · en _n_ |
| Verdicts | 🔴 _n_ · 🟠 _n_ · 🟢 _n_ · questions _n_ |
| Helpful (👍 of rated) | _x_ / _y_ (_z_%) |
| Latency p50 / p95 | text _a_ / _b_ s · voice _c_ / _d_ s |

Ten de-identified exchanges with English glosses, and tester quotes, go here (from `kudiguard export`, already redacted).

## Native-speaker review (to fill in after the pilot)
Native speakers grade a stratified sample of 30–50 real replies (rubric and process: [NATIVE_REVIEW.md](NATIVE_REVIEW.md)).
Paste the output of `kudiguard review score var/review/<lang>-*.csv ... --markdown` here (it also prints an
inter-rater agreement table: pairwise agreement, Gwet's AC1 and splits per language).

| Language | Replies graded | Accuracy (1-5) | Fluency (1-5) | Faithful to facts | Harmful or misleading |
|---|---|---|---|---|---|
| ha | _n_ | _x_ | _x_ | _x_% | _x_% |
| yo | _n_ | _x_ | _x_ | _x_% | _x_% |
| ig | _n_ | _x_ | _x_ | _x_% | _x_% |
| **All** | _n_ | _x_ | _x_ | _x_% | _x_% |

_n_ native-speaker reviewers; inter-rater agreement on _n_ double-graded replies: _x_. Replies marked harmful or
unfaithful, and what was changed: _…_

## Blind accuracy (never tuned on; docs/SPIKES.md S6–S7)
| Test | Result |
|---|---|
| Scam detection, 40 messages (ha/yo/ig/en, 50% legit) | 95% of scams flagged red/orange, 86% precision; rules alone 65% recall |
| Knowledge retrieval, 24 questions | right entry in top 3: 96% |
| Speech, Google FLEURS clips as WhatsApp audio (WER) | Hausa 28% · Yoruba 33% · Igbo 46% |

## Speech on real voice notes (donated, opt-in)
Users who opt in donate voice notes (adults, own voice; docs/DPIA_VOICE.md). The transcription loop:

1. **Export a sample:** `kudiguard voice export --out var/voice-export/2026-10 --n 50 --seed 1 [--languages ha,yo]`.
   * This draws a seeded random sample per language.
   * It writes the decrypted audio to `audio/<id>.<ext>`, plus `manifest.csv` (UTF-8 BOM, formula-safe) with the
     automatic `asr_hypothesis` and an empty `reference` column.
   * Only `var/` or `exports/` are accepted, never a cloud-synced folder.
2. **Transcribe:** native-speaker transcribers work under written confidentiality, locally, with no cloud drives
   and no copies abroad.
   * Listen to the audio and type exactly what was said into `reference`. Do not edit the hypothesis.
   * Type every number as `[number]`. References are masked like the hypotheses anyway.
   * Keep tone marks where you hear them.
   * Save the file as **CSV UTF-8**.
   * Report sample ids that contain a child's voice, another person's voice, or a PIN, OTP, BVN or account number
     read out. Delete them with `kudiguard voice remove <id> ...`.
3. **Score:** `kudiguard voice wer var/voice-export/2026-10/manifest.csv` gives pooled WER per language (diacritics
   folded and kept) and CER, with the same scoring as `scripts/eval_asr.py` (`kudiguard.speech.metrics`). Copy the
   table here. The hypotheses come from the model that ran at donation time. To compare a new model, re-transcribe
   the same clips with it (and keep the human references).
4. **Clean up:** delete the export folder and every transcriber's copy.

`kudiguard voice stats` counts donors, samples and minutes per language.

| Language | Clips | WER | WER+d | CER |
|---|---|---|---|---|
| ha / yo / ig / en | _n_ | _x_ | _x_ | _x_ |

## Known limitations
* A language tap within 5 s of the welcome gets "please read the message above" once, because consent must
  come after the notice could be read (docs/DEPLOY.md).
* Genuine bank OTP SMS can be flagged 🟠 "be careful" (safe direction; fix pending separate examples).
* Hausa/Yoruba/Igbo fixed text and lexicons await native-speaker review (model replies: see above).
* **Known risk: translations flip negations.** Yoruba and Igbo replies are composed in English and machine-translated
  (default `KUDIGUARD_COMPOSE_STRATEGY=english_first,ha=direct`; docs/SPIKES.md H). Of 93 measured translations, 10
  (**11%**) lost or added a negation (Igbo "Do not pay anything" → "give any money"; Yoruba "You should not keep all
  your money at home" → "keep all your money at home"; Hausa "block your SIM" → "do not block your SIM"). A negation
  check turns those into fixed text in the language (8–14% of dev replies), but other meaning errors ("three months"
  → "three weeks") are not detected. Negator lists, translations and the new yo/ig scam terms are DRAFT, unreviewed.
* **Hausa replies are composed directly in Hausa** (English-first cut Hausa coverage 0.54 → 0.34 on the dev set, while
  it raised Yoruba 0.36 → 0.56 and Igbo 0.37 → 0.42). A direct reply only passes the output guard (Hausa and English
  patterns): no English-side checks and no fidelity or negation check, so a flipped instruction in a Hausa reply is
  not caught automatically. Native-speaker review of Hausa replies is the control for this.
* English-first adds ~1.6–2.3 s per Yoruba/Igbo reply (p95 up to ~16 s with a retry); Hausa and English replies are
  not slowed.
* Every language's answer depends on the model's English translation of the message, which still mistranslates some
  Yoruba/Igbo/Hausa words (ajo/adashe, "blocked" read as "hacked").
* Yoruba and Igbo stay **beta**; Igbo speech forces Whisper's `yo` token (the model's own choice on every test clip,
  10 read clips only).
* Speech accuracy measured on read speech; real voice notes from the pilot will be added.
