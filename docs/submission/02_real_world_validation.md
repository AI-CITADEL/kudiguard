# Real-World Validation

**KudiGuard — NAIC 2026, Problem Statement 2 (Voice-First Access)**

Document 3 of the submission: benchmark results on frozen test sets, speech accuracy, and the live pilot of
12 Oct 2026 with what it found and what we fixed the same day. Every number names its source file; the full claims
register with commands to reproduce is `docs/CLAIMS.md`.

---

## 1. Summary

| Evidence | Result | Source |
|---|---|---|
| Frozen end-to-end benchmark **T v1** (366 cases, 5 languages, run once) | pass **85%**; scams flagged **178/196 (91%)**; genuine messages wrongly flagged **35/96 (36%)**; questions answered 72/74; p50 3.5 s, p95 8.3 s; 0 errors | `docs/CLAIMS.md` C10, `data/eval/results/e2e-live.jsonl` |
| Advice fixes on dev set dev2 (130 cases) | judged good 10 -> 26; advice points 32% -> 46%; scams flagged 48/50 -> 48/50 | `docs/CLAIMS.md` C14 |
| NCAIR1 speech, word error rate | Hausa 28%, Yoruba 33%, Igbo 46% (read speech) | `docs/CLAIMS.md` C3 |
| **Live pilot, 12 Oct 2026** | 14 real interactions (13 voice notes) from 3 testers on the web voice page; 8 issues found, 7 fixed and 1 experiment rejected the same day; 11 live messages replayed as text after the fixes: 10 flagged | `var/kudiguard.db`, `kudiguard stats`, git log |

The live sample is **small** (section 5). We state it plainly rather than extrapolate from it.

## 2. Headline benchmark: frozen set T v1 (`docs/CLAIMS.md` C10, `docs/VALIDATION.md`)

**Set.** `data/eval/e2e/cases.yaml`: 366 cases (English 147, Pidgin 63, Hausa/Yoruba/Igbo 52 each; 196 scams,
96 genuine messages, 74 questions; 49 scam types), written blind by an independent author and **frozen on
2026-10-10 before any system output was seen**. No rule, prompt, lexicon, retrieval setting or threshold was tuned
on it. **Run once**, live, at commit 8e15852, through the real engine with greedy decoding (temperature 0, seed 42)
and the product's turn deadline.

| Measure | T v1 |
|---|---|
| Pass rate | **85%** |
| Scams flagged (red or orange) | **178 / 196 (91%)** |
| Genuine messages wrongly flagged | **35 / 96 (36%)** |
| Questions answered | 72 / 74 |
| Latency (text) | p50 3.5 s, p95 8.3 s |
| Errors | 0 |
| Advice judged blind by 4 LLM judges | good 94, acceptable 108, poor 131, **harmful 33** of 366 |
| Key advice points conveyed (must_advise) | 281 / 609 (46%) |
| Rules-only floor (model off) | 88/196 scams flagged, 11/96 genuine flagged |

**Caveats (as stated in `docs/CLAIMS.md` C10 and `docs/VALIDATION.md`):**

* Run once; nothing tuned on T before it.
* **30 of the 33 harmful replies are on genuine messages** (27 called scams): false alarms on real notices are the
  main failure. About a third of genuine notices are over-flagged.
* Judges are LLMs (one per reply, blind to label/verdict/flags), not people.
* Hausa/Yoruba/Igbo cases are near-parallel templates (not independent samples) and DRAFT wording; the 46 voice
  cases were run as text.
* **T v2** (re-run at 5503deb, after fixes informed by T v1's failure categories) is **not independent**: same
  totals (pass 85%, 178/196, 35/96, 72/74; 27 verdicts changed, 6 passes gained and 6 lost), slower (p50 4.3 s,
  p95 10.9 s); judged good+acceptable 57% -> 60%, harmful 33 -> 36, gains within judge noise; known regression
  ig-001 (an Igbo vishing SMS now shown green).

Translation analysis on T v1 (`docs/MODEL_STRATEGY.md` §1): the engine flagged 48 of 54 Yoruba and Igbo scams; of the
6 misses, 3 involved translation (2 never translated, 1 cue lost). Translation is not the main cause of missed scams.

## 3. Advice fixes measured on dev2, before -> after (`docs/CLAIMS.md` C14)

dev2: `data/eval/dev2/cases.yaml`, 130 blind cases (45 genuine, 50 scam, 35 question). Master e9fec66 -> 4abf148
(merged as 5503deb).

| Measure | Before | After |
|---|---|---|
| Judged good (4 blind LLM judges, mixed) | 10 | **26** |
| Good + acceptable | 45% | 57% |
| must_advise points conveyed | 32% | **46%** |
| Harmful | 25 | 23 |
| Scams flagged | 48/50 (96%) | 48/50 (96%) |
| Genuine flagged | 23/45 | 21/45 |
| Pass | 80% | 82% |
| Latency p50 / p95 | 3.6 s / 8.6 s | 4.7 s / 10.2 s |

37 cases better, 14 worse. Caveat: **a dev set used to measure the fixes, so not independent evidence**; about 22 of
45 genuine notices are still judged harmful; LLM judges; ha/yo/ig/pcm text DRAFT, n=130.

## 4. Speech recognition, per language (`docs/SPIKES.md` S6, `docs/CLAIMS.md` C3)

| Language | NCAIR1 model | Clips | WER | WER with diacritics | CER |
|---|---|---|---|---|---|
| Hausa | NCAIR1/Hausa-ASR | 20 | 28.2% | 28.2% | 9.0% |
| Yoruba | NCAIR1/Yoruba-ASR | 10 | 32.7% | 64.0% | 8.4% |
| Igbo | NCAIR1/Igbo-ASR | 10 (1 rejected, counted as 100% errors) | 45.5% | 48.9% | 25.9% |
| English | NCAIR1/NigerianAccentedEnglish | 1 synthetic clip | 0% | — | — |

Google FLEURS read sentences re-encoded as WhatsApp audio; not real voice notes; may overlap NCAIR1 training data.

## 5. Live pilot, 12 Oct 2026

**Setup.** Live server on the team's PC (RTX 4070 Laptop GPU), N-ATLAS Q4_K_M via llama.cpp, NCAIR1 speech models,
public HTTPS via a Cloudflare tunnel, **web voice page** (`/`). Testers recorded voice notes in their own language
about scam situations. WhatsApp was not used by testers on this day (0 WhatsApp users, `kudiguard stats`).

**Database totals** (`.venv\Scripts\python -m kudiguard.cli stats`, run 12 Oct 2026 15:27 EDT, read-only):

```
Interactions: 15   unique users: 4 (whatsapp users: 0, web sessions: 4)
Languages: {'ha': 10, 'yo': 4, 'en': 1}
Verdicts: {'caution': 6, 'no_flags': 4, 'not_a_check': 3, 'likely_scam': 2}
Outcomes: {'answered': 14, 'fallback': 1}
Helpful: 0/0 (n/a)
Latency audio: n=13 p50=7475 ms p95=17686 ms
Latency text: n=2 p50=3793 ms p95=4103 ms
Per day: {'2026-10-08': 1, '2026-10-12': 14}
```

The 2026-10-08 row is the team's internal pre-pilot text test. **12 Oct only** (`var/kudiguard.db`,
`interactions`, opened read-only):

| Language | Voice notes | Typed | Total |
|---|---|---|---|
| Hausa | 9 | 1 | 10 |
| Yoruba | 4 | 0 | 4 |
| **Total** | **13** | **1** | **14** |

Verdicts on 12 Oct (as recorded at the time, before the same-day fixes): caution 6, no_flags 4, not_a_check 3,
likely_scam 1. Outcomes: 13 answered, 1 fallback. Voice-note latency end to end: p50 7,475 ms, p95 17,686 ms
(n=13; Hausa p50 6,417 ms, Yoruba p50 13,667 ms); speech recognition alone p50 2,343 ms. Testers (pseudonymous labels
from `var/testers.csv`): **Tester 1** (team member, Yoruba, collecting ASR terms), **Tester 2** and **Tester 3**
(remote Hausa testers outside the team). No helpful/unhelpful ratings were given.

### 5.1 The real messages (12 Oct 2026, times UTC; transcripts as heard by the NCAIR1 model)

| # | Time | Tester | Lang | Input | Transcript (ASR output) | N-ATLAS English | Verdict then |
|---|---|---|---|---|---|---|---|
| 2 | 16:42 | Tester 1 | yo | voice | ẹnikan pẹrin o so wipe ki mu bifi ẹni mi wa, bo irun bami lati fun ni bifi ẹni mi naa, ṣe skamu lele yii aabi bẹkọ? | — (timed out: model loading) | not_a_check (fallback) |
| 3 | 16:46 | Tester 1 | yo | voice | ẹnikan te mi, oni ki mu bifi eni mi wa kinsẹndẹ sa wọn, ... ki tun fi account-number mi sinu bifi eni mi naa ki wa fẹransẹ si oun | Someone is threatening me ... | likely_scam |
| 4 | 17:00 | Tester 1 | yo | voice | ẹnikọ́pẹ́ mi, o ni ki n fun ohun ni bífi ẹni mi, wọ́n ni bífi ẹni mi ti dina kí n fi ránṣẹ sí wọn. | My friend, I have to give something up as a favor ... | not_a_check |
| 5 | 17:01 | Tester 1 | yo | voice | wọ́n ní bífí ẹ̀nì mi ti díná kí n fi ránṣẹ́ sí wọn. | They have blocked my debit card ... | caution |
| 6 | 17:12 | Tester 2 | ha | voice | shugaban ƙarar ƙarar ƙarar ƙanar | (garbled input) | caution |
| 7 | 17:13 | Tester 2 | ha | voice | mun ya kirani ya ce ma tura mai kudi. | they called me and said that they are sending money. | no_flags |
| 8 | 17:14 | Tester 2 | ha | voice | wani ya kirani ya ce na tura mai kuɗi. | Someone called me and said they were sending money. | no_flags |
| 9 | 18:28 | Tester 3 | ha | voice | wani ya kirani ya ce an toshe bibiyen ɗina. | Someone called and said that my account is blocked. | caution |
| 10 | 18:29 | Tester 3 | ha | text | Wani yakirani yace an toshe bvn dina | A stranger is saying that my BVN has been blocked | caution |
| 11 | 18:50 | Tester 2 | ha | voice | wani ya ƙira ni ya ce na tura mai otp na. | someone is calling me and said that I sent an OTP | caution |
| 12 | 18:50 | Tester 2 | ha | voice | wani ya kirani ya ce na tura mai fim ɗina. | An kira ni don aika fim dinta. | no_flags |
| 13 | 18:51 | Tester 2 | ha | voice | wani ya ƙira ne ya ce na tura mai akawun nan ba na. | Someone created a character who said that I sent this accountant here ... | not_a_check |
| 14 | 18:53 | Tester 3 | ha | voice | wani ya kirani yake ce mun na duba wani sako an turo min wasu lambobi in tura mu shi. | Someone called and said they were checking a message and sent me some numbers to forward to them. | caution |
| 15 | 18:54 | Tester 3 | ha | voice | wani ya kira ne yake ce an tura mu wani sako masu wasu lambobi guda shida, to yace in tura masu lambobi. | Someone called and said that there were some numbers with six digits, so he said to send those numbers. | no_flags |

(# = interaction id. English glosses shortened. "Verdict then" is the verdict recorded live, before the fixes in 5.2.)

What the table shows honestly: the NCAIR1 models heard most Hausa sentences well, but bank words came out phonetically
("bifi ẹni", "bibiyen" for BVN; "fim" for PIN; "akawun" for account), and N-ATLAS translation sometimes flipped the
meaning ("send him money" became "they are sending money", interactions 7–8). Several scam situations therefore got
no_flags or not_a_check live. These are exactly the cases fixed below.

### 5.2 Found in live testing -> fixed the same day

| | Found | Fix | Commit(s) |
|---|---|---|---|
| a | The first Yoruba voice note timed out (interaction 2: 17.7 s, fallback) because its speech model was loaded on first use | Speech models preloaded at startup (`KUDIGUARD_PRELOAD_LANGUAGES`, `NatlasASR.preload`) | `src/kudiguard/config.py`, `api/services.py` |
| b | "BVN" heard as "bifi ẹni" (Yoruba) and "bibiyen" (Hausa) | `asr_terms`: per-language regex corrections of phonetically spelled bank terms, applied after ASR; donated audio keeps the recogniser's own words | 38e3ac8, 9025acf, 920c415 |
| c | Hausa "ya ce na tura mai kuɗi" (send him money) translated "they are sending money", verdict green | Hausa `money_request` rule on the original text, narrowed to an unknown caller ("wani/wata/wasu ... kira") after review | e736303, 5cca936 |
| d | A 0.46 s accidental tap was transcribed as garbage and answered | Voice notes need at least 0.5 s of voiced speech (`MIN_VOICED_SECONDS`, measured on voiced frames) | e736303, 5cca936 |
| e | Yoruba "give him my BVN" / "my BVN is blocked, send it to them" not flagged | Yoruba caller-framed BVN/NIN/card rules (`asks_credentials`, `account_threat`), only from an unknown caller; bank-branch, NIMC and family requests spared | 256b0dc, eaade33, e3f816d |
| f | Hausa requests for OTP, PIN ("fim"), account number ("akawun nan ba") and a six-digit code not flagged | Hausa rules for those requests from an unknown caller | 451629c, 07a1ec6 |
| g | An uncertain non-English voice note with no flags got a green "no warning signs" reply | `KUDIGUARD_VOICE_NO_REASSURANCE` (default true): the reply says some words may have been mis-heard and asks the user to type; stored verdict unchanged | a66a94d, 8d1f551 |
| h | Experiment: a "focused risk check" (an extra N-ATLAS call on the original ha/yo/ig text) | Measured on dev2 (ha/yo/ig/pcm, 70 cases), off -> enforce: scams flagged 25/27 -> 25/27 (no recall gain); genuine flagged 14/25 -> 15/25; Hausa genuine flagged 2/6 (33%) -> 3/6 (50%), n=6; p50/p95 5.5/11.2 s -> 6.5/12.0 s; model calls 187 -> 231. **Rejected** by the pre-set rule (ship only if recall rises and genuine false positives do not): stays off (`KUDIGUARD_FOCUSED_RISK_CHECK=off`), not merged | f961e32 (experiment branch); `data/eval/results/live-2026-10-12/focused-risk-dev2-off.jsonl`, `-enforce.jsonl` (`scripts/eval_e2e.py --summary`) |

**Tester feedback, acted on.** A remote Hausa tester wrote: *"It can't decipher a Hausa pronunciation of words like
account. Most Hausa people don't use the Hausa word for account."* The team followed up: the tester meant the English
word "account" in Hausa pronunciation, spelt for example "akawunt". The `asr_terms` corrections now cover Hausa
"akawun" / "akawunt" / "akaunt" -> account (and "nan ba" / "namba" after it -> account number) and Yoruba
"akaunti" / "akanti" / "akaounti" / "akant" -> account (`data/languages/ha.yaml`, `yo.yaml`; commits 8fc2974, 273c7f6).

**How each fix was checked.** Each change was reviewed, then false-positive checked on the frozen Hausa and Yoruba
cases: **67 Hausa** (52 in T + 15 in dev2) and **67 Yoruba** (52 + 15). Commit messages record the result: Yoruba
rules "No change on any Yoruba e2e/dev2 case" (256b0dc, eaade33, e3f816d); Hausa `money_request` "0 new hits on
Hausa genuine/question cases" and, after narrowing, "no genuine/question case changes" (one scam case, ha-003, moved
from not_a_check to no_flags through the new "someone called me" intent term; 5cca936). All rules remain DRAFT
pending native review. The latest full test run, on master 273c7f6: **5,517 passed**, 1 skipped, 12 xfailed, **100% line and branch
coverage** (9,447 statements, 2,256 branches), mypy clean on 73 source files, ruff format clean.

### 5.3 Replay after the fixes

Today's failed live messages were re-run **as text** through the fixed engine (`scripts/eval_e2e.py`; cases and
results in `data/eval/results/live-2026-10-12/`). Verdicts below are read from the result files.

**Round 1, Yoruba and Hausa** (`replay-yo-ha-cases.yaml` -> `replay-yo-ha-after.jsonl`, 4 cases; the Yoruba texts
carry "BVN" as the voice path now writes it after `asr_terms`):

| Case (live interaction) | Text replayed | Live verdict | After the fixes |
|---|---|---|---|
| live-ha-2 (7) | mun ya kirani ya ce ma tura mai kudi. | no_flags | **caution** (money_request) |
| live-ha-3 (8) | wani ya kirani ya ce na tura mai kuɗi. | no_flags | **caution** (money_request) |
| live-yo-2 (5) | wọ́n ní BVN mi ti díná kí n fi ránṣẹ́ sí wọn. | caution | **likely_scam** (asks_credentials, account_threat) |
| live-yo-3 (4) | ẹnikọ́pẹ́ mi, o ni ki n fun ohun ni BVN mi, wọ́n ni BVN mi ti dina kí n fi ránṣẹ sí wọn. | not_a_check | **likely_scam** (asks_credentials, account_threat) |

**Round 2, Hausa** (`replay-ha-round2-cases.yaml` -> `replay-ha-round2-after.jsonl`, 7 cases): **6 of 7 flagged.**

| Case (live interaction) | Text replayed | Live verdict | After the fixes |
|---|---|---|---|
| r2-1 (9) | wani ya kirani ya ce an toshe bibiyen ɗina. | caution | no_flags (see note) |
| r2-2 (10) | Wani yakirani yace an toshe bvn dina | caution | **caution** (account_threat) |
| r2-3 (11) | wani ya ƙira ni ya ce na tura mai otp na. | caution | **likely_scam** (asks_credentials) |
| r2-4 (12) | wani ya kirani ya ce na tura mai fim ɗina. | no_flags | **likely_scam** (asks_credentials) |
| r2-5 (13) | wani ya ƙira ne ya ce na tura mai akawun nan ba na. | not_a_check | **likely_scam** (asks_credentials) |
| r2-6 (14) | wani ya kirani yake ce mun na duba wani sako an turo min wasu lambobi in tura mu shi. | caution | **caution** (suspicious_link) |
| r2-7 (15) | wani ya kira ne yake ce an tura mu wani sako masu wasu lambobi guda shida, to yace in tura masu lambobi. | no_flags | **likely_scam** (asks_credentials) |

Note on r2-1: it was replayed as **text**, which skips the voice-only `asr_terms` correction ("bibiyen" -> BVN); its
typed twin r2-2 ("an toshe bvn dina") is flagged. As a voice note the correction applies (section 5.2 b).

**Read this plainly:** the replay uses the same messages the fixes were written from. It shows the fixes work on
those messages; it is **not** a measure of general accuracy. General accuracy is T v1 (section 2), and the fixes'
false-positive check is the 67 + 67 frozen cases above.

## 6. Known limitations (stated plainly)

* **Small live sample**: 4 users and 15 interactions in the database, of which 14 on 12 Oct from 3 testers (one a
  team member) on the web page; no WhatsApp users and no ratings yet. It shows real failure modes, not accuracy.
* **Hausa ASR ~28% WER** on read speech (Yoruba 33%, Igbo 46%); real voice notes are harder, and bank words come out
  phonetically. `asr_terms` cover only the terms observed so far.
* **Translation can flip meaning** (live interactions 7–8; 11% of measured translations reversed a negation,
  `docs/MODEL_STRATEGY.md`). Rules on the original text are the safeguard.
* **About a third of genuine notices are over-flagged** on T (35/96), the main failure; 30 of 33 harmful replies are
  there.
* **Hausa, Yoruba and Igbo rules, texts and test cases are DRAFT**, pending native-speaker review (none done yet,
  `docs/VALIDATION.md`).
* Benchmarks use LLM judges, not people; T's ha/yo/ig cases are near-parallel templates.
