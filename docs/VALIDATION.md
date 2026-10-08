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

## Known limitations
* Genuine bank OTP SMS can be flagged 🟠 "be careful" (safe direction; fix pending separate examples).
* Hausa/Yoruba/Igbo fixed text and lexicons await native-speaker review.
* Speech accuracy measured on read speech; real voice notes from the pilot will be added.
