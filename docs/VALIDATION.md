# Validation

Evidence for NAIC 2026: real usage by people outside the team, plus accuracy on small test sets. Usage numbers come
from `kudiguard stats --exclude-users team.txt` (team testers' numbers, one per line, are hashed and excluded). Every
number quoted anywhere, with its source, command and caveat: [CLAIMS.md](CLAIMS.md).

## Real usage
**Status (10 Oct 2026): no real-user data yet.** The pilot has not started; this table is filled from `kudiguard stats` after it.

| Measure | Value |
|---|---|
| Period | no real-user data yet |
| Interactions (WhatsApp + web; target ≥ 50) | no real-user data yet |
| Unique WhatsApp users / web sessions | no real-user data yet |
| Languages (ha · yo · ig · en) | no real-user data yet |
| Verdicts (🔴 · 🟠 · 🟢 · questions) | no real-user data yet |
| Helpful (👍 of rated) | no real-user data yet |
| Latency p50 / p95 (text · voice) | no real-user data yet (dev set, shipped code: text p50 4.0 s / p95 8.2 s, CLAIMS.md C11) |

After the pilot: ten de-identified exchanges with English glosses, and tester quotes (from `kudiguard export`, already
redacted).

## Native-speaker review
**Status (10 Oct 2026): no reply has been reviewed by a native speaker yet**, and neither have the Hausa, Yoruba and
Igbo fixed texts, lexicons or negator lists. After the pilot, native speakers grade a stratified sample of 30–50 real
replies (rubric and process: [NATIVE_REVIEW.md](NATIVE_REVIEW.md)); the table is then replaced by the output of
`kudiguard review score var/review/<lang>-*.csv ... --markdown` (with its inter-rater agreement table).

| Language | Replies graded | Accuracy (1-5) | Fluency (1-5) | Faithful to facts | Harmful or misleading |
|---|---|---|---|---|---|
| ha / yo / ig | 0 (not reviewed yet) | - | - | - | - |

## Accuracy on test sets (small; details and commands in [CLAIMS.md](CLAIMS.md))
| Test | Result | Caveat |
|---|---|---|
| Scam detection by the rules alone, 40 messages (10 per language, half genuine) | 80% of scams flagged red/orange, 89% precision, 10% false alarms | written blind, but used once for lexicon edits since; ha/yo/ig AI-drafted |
| Scam detection with N-ATLAS (full engine) | to be re-run live | the 8 Oct run predates greedy decoding, new lexicon terms and English-first replies |
| Knowledge retrieval, 24 independent questions | right entry in top 3: 96% (top 1: 62%) | English questions only (assumes a perfect translation) |
| Speech, FLEURS read sentences as WhatsApp audio (WER) | Hausa 28% (20 clips) · Yoruba 33% (10; 64% with tone marks) · Igbo 46% (10) | read speech, not voice notes; possible training overlap |
| Frozen end-to-end set T v1 (366 cases, 49 scam types, 5 languages; live once at 8e15852) | pass 85%; scams flagged 178/196 (91%); genuine flagged 35/96 (36%); questions answered 72/74; p50 3.5 s, p95 8.3 s; 0 errors. Judged advice: 94 good, 108 acceptable, 131 poor, 33 harmful; 281/609 must_advise points (46%) | ha/yo/ig cases near-parallel and DRAFT; LLM judges, not people; 30 of the 33 harmful are genuine messages |
| T v2 (5503deb): pending | pending | re-run after fixes informed by T v1's failure categories; **not an independent measurement** |
| dev2 advice fixes (130 cases, not frozen) before e9fec66 → after 4abf148 | scams 48/50 → 48/50; genuine flagged 23/45 → 21/45; judged good 10 → 26, good+acceptable 45% → 57%, advice points 32% → 46% | dev set used to measure the fixes; ~22 of 45 genuine notices still judged harmful; ha/yo/ig/pcm DRAFT |

## End-to-end test set T (frozen, v2.1)
**Set.** `data/eval/e2e/cases.yaml`: 366 cases (en 147, pcm 63, ha/yo/ig 52 each; 196 scam, 96 genuine, 74 question),
written blind by an independent author and frozen on 2026-10-10 before any system output was seen. No rule, prompt,
lexicon, retrieval setting or threshold is tuned on it. The runner was built and tested on a synthetic 8-case fixture
(`tests/fixtures/e2e_mini.yaml`); the live model is run on T once.

**Run.** `scripts/eval_e2e.py` sends every case, one at a time, through the real `Engine` (greedy decoding:
temperature 0, seed 42) under the product's turn deadline (60 s turn limit minus the 3 s reply reserve,
`--turn-deadline`), with a much longer hang guard that writes an error row instead of stopping. `--resume`
continues an interrupted run, retries its error rows, and refuses a file made in another mode or against another
model server (every row records both). Pidgin cases run as `en`, as in the product. The 46 `voice: true` cases run as text
(no TTS step). Each row stores the verdict, flags, the reply as shown, the English pivot the engine used, the English
draft of an English-first reply, translation rejections, fallback, latency, model calls and the case's
must_advise / must_not. For yo/ig scam misses the row also says whether the rules find a red flag in the English
pivot (`cue_survived_translation`), find none (`no_rule_cue_in_translation`: read the pivot to see whether the
translation lost the cue) or there was no pivot (`no_translation`).

**Scoring** (`kudiguard.evaluation.e2e`, the header rules exactly). Red and orange = flagged, green = not flagged.
Scam passes when flagged, genuine when not flagged, question when answered (a model-composed reply; a colour shown
with it is ignored, and red verdicts on questions are counted for the judge to decide "no basis"). `accept` values
also pass (caution = flagged). The headline is the pass rate (label or accept); scam recall, genuine
false-positive rate and questions answered are label-only and ignore `accept`. Red flags are compared through an explicit map: the 13 shared names map one-to-one to
the system's flags; `harassment` maps to no match, and the system's THREAT flag (added after T was frozen, so not in its vocabulary) is left out of flag precision. Error rows are excluded from rates and printed
prominently, with a worst-case pass rate that counts them as failures; above 2% errors the run exits non-zero.
Flag precision uses only scam cases with at least one mapped expected flag and is a lower bound: the author's flag
lists are not exhaustive, so a system flag outside them may still be right. Reported per
language and overall: scam recall, genuine false-positive rate, question-answered rate, flag recall/precision,
latency p50/p95, fallback rate, translation-rejection rate and model calls (in `--rules-only` mode
these count the failed call attempts). ha/yo/ig scam cases are near-parallel templates, so
per-language scores are a consistency check, not independent evidence.

**Answer quality.** must_advise / must_not are never keyword-scored. `scripts/judge_e2e_prep.py` writes blind batches
(id, language, message, reply shown, must_advise, must_not; no label, verdict field or flags) for independent LLM
judges.

**T v1: the official headline.** Live, once, at commit 8e15852 on the frozen set; results in
`data/eval/results/e2e-live.jsonl`. Recompute: `PYTHONIOENCODING=utf-8 python scripts/eval_e2e.py --summary data/eval/results/e2e-live.jsonl`.

| T v1 result (engine, live, 8e15852, n=366) | Value |
|---|---|
| Pass rate, label or accept (all; worst case with errors as failures) | **85%** (85%: 0 errors); en 86%, pcm 86%, ha 85%, ig 85%, yo 83% |
| Scam recall | **178/196 (91%)**; en 91%, pcm 97%, ha 85%, ig 89%, yo 89% |
| Genuine messages flagged (false-positive rate) | **35/96 (36%)**; en 32%, pcm 50%, ha 31%, ig 38%, yo 38% |
| Questions answered | 72/74 (2 shown red; basis left to the judge) |
| Flag recall / precision (lower bound) | 35% / ≥81% |
| Latency p50 / p95 | 3.5 s / 8.3 s (Yoruba p95 11.5 s) |
| Fallback / translation-rejection rate | 5% / 4% (Igbo 27% / 23%) |
| yo/ig scam misses (6 of 54) | 2 never translated (classified as a greeting, fixed reply: ig-002, ig-026); 4 translated with no English rule cue, of which 1 lost a cue in translation (yo-002, "don't call me" became "my network is good") and 3 kept the cue but were not flagged (yo-008, yo-023, ig-001) |

**Blind advice judging (T v1).** `data/eval/results/judge/e2e-e2e-live-0{1..4}.judged.jsonl`: 4 independent LLM
judges, each on a quarter of the 366 replies, blind to the label, the verdict and the flags.

| Judged (n=366 replies) | Value |
|---|---|
| Grades | good 94, acceptable 108, poor 131, **harmful 33** |
| must_advise points conveyed | 281/609 (46%) |
| Where the harm is | 30 of the 33 harmful replies are on **genuine** messages (27 of them were called scams); 3 on scams |

Caveats: LLM judges, not people or native speakers; one judge per reply (no agreement measured); ha/yo/ig/pcm
cases are DRAFT and the ha/yo/ig scams near-parallel templates; voice cases ran as text. The main failure is
**false alarms on real bank notices** (36% of genuine messages flagged), not missed scams.

**T v2 (5503deb): pending.** A re-run of T on master 5503deb after fixes informed by T v1's failure categories.
It is **not an independent measurement**: T v1 above stays the headline.

| T v2 result (engine, live, 5503deb) | Value |
|---|---|
| Pass rate / scams flagged / genuine flagged / questions answered | pending |
| Latency p50 / p95 | pending |
| Judged advice | pending |

Deterministic safety floor (rules alone, no model; `--rules-only`, results in `data/eval/results/e2e-rules.jsonl`):
88/196 scams flagged (45%), 11/96 genuine messages flagged (11%), 0/74 questions answered (no model, so only fixed
text), flag recall 22%, flag precision ≥95% (a lower bound: the author's flag lists are not exhaustive).

## Dev set dev2: advice fixes, before / after (not frozen)
**Set.** `data/eval/dev2/cases.yaml`: 130 cases (en 60, pcm 25, ha/yo/ig 15 each; 50 scam, 45 genuine, 35 question),
written blind by an independent author. It is a **dev set**, used to measure the advice fixes (workstream E2): not
frozen, not a test set. Recompute any row with `PYTHONIOENCODING=utf-8 python scripts/eval_e2e.py --summary <file>`.

| dev2 (n=130) | Before: master e9fec66 (`e2e-dev2-before.jsonl`) | After: 4abf148, merged as 5503deb (`e2e-dev2-after2.jsonl`) |
|---|---|---|
| Pass rate | 80% | 82% |
| Scams flagged | 48/50 (96%) | 48/50 (96%) |
| Genuine flagged | 23/45 (51%) | 21/45 (47%) |
| Questions answered | 34/35 | 34/35 |
| Latency p50 / p95 | 3.6 s / 8.6 s | 4.7 s / 10.2 s |

A first "after" attempt (2747121, `e2e-dev2-after.jsonl`) dropped scams flagged to 42/50 (84%). The cause was a new
`user_initiated` field in the ANALYSE step; it was reverted and nothing from that attempt was merged.

**Blind mixed before/after judging.** `data/eval/results/judge/dev2/dev2-mixed-0{1..4}.judged.jsonl`: 260 replies
(130 before + 130 after, shuffled), 4 independent LLM judges blind to the label, the verdict and which run a reply
came from. The before/after key is kept privately, out of the repo, so the split below cannot be recomputed from the
repo alone (the pooled totals can: good 36, harmful 48, 213/549 advice points).

| Judged (130 cases each) | Before | After |
|---|---|---|
| good | 10 | 26 |
| good + acceptable | 45% | 57% |
| must_advise points conveyed | 32% | 46% |
| harmful | 25 | 23 |
| Per case | 37 better, 14 worse (rest unchanged) | |

Caveats:
- About 22 of the 45 genuine notices are still judged harmful: N-ATLAS still calls real bank notices scams. The
  knowledge packs improved the advice, not that verdict.
- Pidgin replies mostly come back in English. The Pidgin-reply feature is **off by default**: it added about 3 s p50
  for no measured gain.
- Latency rose (p50 3.6 → 4.7 s) with the retrieved model answers.
- dev2's ha/yo/ig/pcm text is DRAFT and needs native review; n=130 (one scam = 2 points).

**Knowledge base.** Grown from about 50 to about 126 entries, with two new packs: genuine-notices (30) and
situations (46), each with model answers and fact-checked sources in comments.

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
| ha / yo / ig / en | 0: no donated voice notes yet (donation is off until the DPIA is signed off) | - | - | - |

## Known limitations
* A language tap within 5 s of the welcome gets "please read the message above" once, because consent must
  come after the notice could be read (docs/DEPLOY.md).
* Genuine bank OTP SMS can be flagged: 🟠 "be careful" in English, and 🔴 in Igbo with the rules alone (scenario
  s32; the Yoruba dev case yo-otp-legit too). Safe direction, but alarming; fix pending separate examples.
* Hausa/Yoruba/Igbo fixed text and lexicons await native-speaker review (model replies: see above). This includes
  the welcome/consent text and the web page's consent box in those languages: DRAFT translations.
* **Known risk: translations flip negations.** Yoruba and Igbo replies are composed in English and machine-translated
  (default `KUDIGUARD_COMPOSE_STRATEGY=english_first,ha=direct`; docs/SPIKES.md H). Of 93 measured translations, 10
  (**11%**) lost or added a negation (Igbo "Do not pay anything" → "give any money"; Yoruba "You should not keep all
  your money at home" → "keep all your money at home"; Hausa "block your SIM" → "do not block your SIM"). A negation
  check turns those into fixed text in the language (shipped code: 8% of Yoruba and 17% of Igbo dev replies; the same
  check covers the "What should I do?" and "Warn my family" buttons), but other meaning errors are not detected:
  "three months" → "three weeks"; "be careful" → Igbo "Nwee ndidi" (be patient); "block your account" → Igbo
  "gbanye" (enable/switch on); "block your SIM" → "protect it". The check reads draft word lists, not meaning. Negator lists, translations and the new yo/ig scam terms are DRAFT,
  unreviewed; the yo/ig lexicon gains were measured in-sample on the dev set (held-out e2e: Yoruba scam recall 5/27 →
  6/27 rules-only, Igbo unchanged).
* **Hausa replies are composed directly in Hausa** (English-first cut Hausa coverage 0.54 → 0.34 on the dev set; it
  raised Yoruba 0.36 → 0.58 but not Igbo, 0.37 → 0.33, where its benefit is less example copying and the English
  safety checks). A direct reply only passes the output guard (Hausa and English
  patterns): no English-side checks and no fidelity or negation check, so a flipped instruction in a Hausa reply is
  not caught automatically. Native-speaker review of Hausa replies is the control for this.
* English-first adds +1.7 s (Yoruba) / +1.3 s (Igbo) p50 per reply (p95 8.2 / 9.4 s) and +42% / +35% model time per
  turn (≈26–30% fewer Yoruba/Igbo turns per GPU-minute); Hausa and English replies are not slowed.
* Every language's answer depends on the model's English translation of the message, which still mistranslates some
  Yoruba/Igbo/Hausa words (ajo/adashe, "blocked" read as "hacked").
* Yoruba and Igbo stay **beta**; Igbo speech forces Whisper's `yo` token (the model's own choice on every test clip,
  10 read clips only).
* Speech accuracy measured on read speech; real voice notes from the pilot will be added.
