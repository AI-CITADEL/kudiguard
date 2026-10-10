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
Paste the output of `kudiguard review score var/review/graded-*.csv --markdown` here.

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

## Known limitations
* Genuine bank OTP SMS can be flagged 🟠 "be careful" (safe direction; fix pending separate examples).
* Hausa/Yoruba/Igbo fixed text and lexicons await native-speaker review (model replies: see above).
* Speech accuracy measured on read speech; real voice notes from the pilot will be added.
