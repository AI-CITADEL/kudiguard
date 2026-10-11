# Model strategy: where N-ATLAS is strong, where it falls short, and what we will (and won't) change

Status: 11 Oct 2026. Every claim below cites its evidence; anything not yet measured says so. Hausa, Yoruba and Igbo test
texts are AI-drafted pending native-speaker review, so per-language numbers are indicative.

## 1. What the measurements show about N-ATLAS

| Capability | Finding | Evidence |
|---|---|---|
| Translating user messages into English (what scam detection relies on most) | ✅ Rated **excellent** for Hausa, Yoruba and Pidgin in the first spike; Igbo not separately rated | `docs/SPIKES.md` S1 |
| Writing replies in Hausa and English | ✅ Good: coverage of key advice en 0.64, ha 0.46 (direct) | `docs/SPIKES.md` H; `data/eval/results/answers-h2_final.jsonl` |
| Yoruba replies, written in English first then translated | ✅ Good: coverage 0.36 → **0.58**, close to English | same |
| Writing Igbo directly | ❌ Weak: 42% of replies copied the built-in example; thin answers | `answers-h_after_direct.jsonl` |
| Igbo replies via English-first | ⚠️ Safer, not fuller: coverage 0.33 (0.37 direct); copying 42% → 8% | `answers-h2_final.jsonl` |
| Translating replies into Yoruba and Igbo | ❌ Reverses negations: 10 of 93 translations (**11%**) in the first offline check; in the shipped run **8%** (yo) and **17%** (ig) of replies were caught by our negation check and replaced with fixed text | `docs/SPIKES.md` H; `docs/VALIDATION.md` |
| Other meaning errors in translation | ❌ Not caught by any check, e.g. "be careful" → Igbo "be patient"; "block your SIM" → Yoruba "protect it" | `docs/VALIDATION.md` |
| Compared with other models | ❓ **Not measured.** We cannot claim another model would do better; most general models are weaker in Yoruba and Igbo, not stronger | — |

Related measurement: with the rules alone (no model), the native scam-word lists caught about 6 of 27 Yoruba and 6 of 27
Igbo scams on the frozen end-to-end set (`data/eval/e2e`, measured during the workstream-H review).

**End-to-end evidence on translation (T v1, live once at 8e15852, `data/eval/results/e2e-live.jsonl`, `miss_diagnosis`
plus reading the English pivot).** The full engine flagged 48 of 54 Yoruba and Igbo scams (yo 24/27, ig 24/27). Of the
**6 misses**, 3 involved translation: 2 were never translated (the message was classified as a greeting and got a fixed
reply: ig-002, ig-026) and 1 lost a cue in translation (yo-002: "don't call me" became "my network is good"; the
money request survived). The other 3 had a clean translation that kept the cue and were still not flagged (yo-008,
yo-023, ig-001). n=6, so this is indicative only; ha/yo/ig cases are DRAFT near-parallel templates. Translation is
**not the main cause** of missed scams; the larger failure on T is false alarms on genuine messages (36% flagged,
all languages).

## 2. The constraint

The NAIC 2026 challenge is built on N-ATLAS, and judges score how deeply a project uses it (`docs/WHY_KUDIGUARD.md`).
Replacing N-ATLAS as the core model would very likely break the challenge rules. **Team action: confirm against the PS2
rules text.** Until then, N-ATLAS stays the main model.

## 3. Strategy: N-ATLAS at the core; specialist models only where it measurably fails

| Step | Today | Possible specialist | Notes |
|---|---|---|---|
| Speech → text | Official NCAIR1 ASR models | — | Already specialist models |
| Translation for scam detection | N-ATLAS | Meta NLLB-200 (supports Hausa, Yoruba, Igbo) as a **second opinion** | Runs locally. ⚠️ **Licence CC-BY-NC 4.0 (non-commercial)**, and Meta's model card says it is "a research model and is not released for production deployment". Acceptable for the competition and pilot; **must be replaced before any commercial (bank/telco) use**. |
| Translating replies into Yoruba and Igbo | N-ATLAS + our fidelity and negation checks | NLLB or similar | Same licence caveat; adopted only if it beats N-ATLAS on our fidelity checks |
| Writing the reply | N-ATLAS (English-first for yo/ig; direct for en/ha) | — | Kept: it is the challenge's core |

Rules for adding any specialist model: it must win on our own measurements, be configurable (and switchable off), run
locally (no cross-border transfer of messages; a cloud translator would send Nigerians' messages abroad and is ruled out),
and carry its licence in `docs/` and the settings.

## 4. Plan: measure, don't switch on a hunch

1. **Done: run the frozen end-to-end set (T) live, once** (8e15852; results in §1) (`data/eval/e2e/cases.yaml`, 366 cases). For every Yoruba and Igbo scam
   it misses, record whether the warning sign was lost in **translation** or missed by the **rules**.
2. **Translation A/B: lower priority.** Step 1 is done (T v1, §1): only 3 of 6 Yoruba/Igbo scam misses involved
   translation, and 2 of those were intent routing (no translation at all), not translation quality. The A/B stays a
   possible later measurement, behind fixing false alarms on genuine notices. If run: translate T's Yoruba and Igbo
   cases (54 scams, 50 genuine messages and questions) with N-ATLAS and with NLLB-200, locally.
   - Score: do the scam signals survive (OTP/PIN/BVN, fee, link, "blocked", urgency, secrecy)? Which translation do the
     English rules catch? False alarms on the genuine messages; latency; memory.
   - Measurement only: nothing is tuned on T for it.
3. **Decide by the numbers:** if NLLB clearly wins, add it as a configurable second opinion **for scam detection only**,
   flagged with its licence. Otherwise keep N-ATLAS alone. Either way N-ATLAS remains the core model.

## 5. What we tell judges

> We built on N-ATLAS, measured where it falls short for Yoruba and Igbo, and added safeguards: English-first replies with
> negation checks, rule-based verdicts that don't depend on the model, and a reply guard. Where a specialist model helps,
> we add it only on evidence, locally, and with its licence stated.

Sources: Meta NLLB-200 model card, https://huggingface.co/facebook/nllb-200-distilled-600M (licence cc-by-nc-4.0; "not
released for production deployment"); this repository's `docs/SPIKES.md`, `docs/VALIDATION.md`, `data/eval/results/`.
