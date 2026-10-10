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

## Blind accuracy (never tuned on; docs/SPIKES.md S6–S7)
| Test | Result |
|---|---|
| Scam detection, 40 messages (ha/yo/ig/en, 50% legit) | 95% of scams flagged red/orange, 86% precision; rules alone 65% recall |
| Knowledge retrieval, 24 questions | right entry in top 3: 96% |
| Speech, Google FLEURS clips as WhatsApp audio (WER) | Hausa 28% · Yoruba 33% · Igbo 46% |

## Speech on real voice notes (donated, opt-in)
Users who opt in donate voice notes (adults, own voice; docs/DPIA_VOICE.md). The transcription loop:

1. `kudiguard voice export --out var/voice-export/2026-10 --n 50 --seed 1 [--languages ha,yo]`: a seeded random
   sample per language, decrypted to `audio/<id>.<ext>` plus `manifest.csv` (UTF-8 BOM, formula-safe) with the
   automatic `asr_hypothesis` and an empty `reference` column. Only git-ignored folders (`var/`, `exports/`) or
   outside the checkout are accepted.
2. Native-speaker transcribers (written confidentiality; work locally, no cloud drives, no copies abroad) type
   exactly what was said into `reference`, listening to the audio, not editing the hypothesis. They report sample
   ids that contain a child's or another person's voice: `kudiguard voice remove <id> ...` deletes them.
3. `kudiguard voice wer var/voice-export/2026-10/manifest.csv`: pooled WER per language (diacritics folded and
   kept) and CER, the same scoring as `scripts/eval_asr.py` (`kudiguard.speech.metrics`). Copy the table here.
4. Delete the export folder and every transcriber's copy.

`kudiguard voice stats` counts donors, samples and minutes per language.

| Language | Clips | WER | WER+d | CER |
|---|---|---|---|---|
| ha / yo / ig / en | _n_ | _x_ | _x_ | _x_ |

## Known limitations
* Genuine bank OTP SMS can be flagged 🟠 "be careful" (safe direction; fix pending separate examples).
* Hausa/Yoruba/Igbo fixed text and lexicons await native-speaker review.
* Speech accuracy measured on read speech; real voice notes from the pilot will be added.
