# DPIA: opt-in voice-note donation

Data protection impact assessment under the Nigeria Data Protection Act 2023 (NDPA) s.28, and the NDPC's General
Application and Implementation Directive (GAID 2025, in force 19 September 2025). Status: **draft for the operator's
sign-off**. This is an engineering assessment, not legal advice. **Before scaling beyond the pilot, have a qualified
Nigerian data-protection lawyer or a licensed DPCO review it.** Items marked **DECIDE** need the operator or a lawyer.

## 1. What is processed, and why
* **Purpose (only one):** test and improve speech recognition for Hausa, Yoruba, Igbo and Nigerian English.
  Native-speaker transcribers type what was said, and `kudiguard voice wer` measures the word error rate.
  The recordings are never used for anything else (s.24(1)(b)).
* **Who:** adult users who opt in. On WhatsApp the offer comes at most once, after an answered voice note that was
  not a scam warning, or when the user sends "donate voice". On the website, the user ticks a separate box, which is
  unticked by default.
* **What (s.24(1)(c), minimisation):**
  * the original audio as received, at most 16 MB and 90 s of speech;
  * a random sample id;
  * the donor's keyed hash (`privacy.UserHasher`), never the phone number;
  * language, channel and the ISO week (no timestamp);
  * the version of the donation text accepted;
  * the duration, when known;
  * the automatic transcript, with numbers and emails masked (`text.redact`).
* **Consent record (s.26(1)):** the user row stores `donation_version` and `donation_at`. Each sample row stores the
  version too. A donor's user row is kept while they have samples.
* **Retention (s.24(1)(d)):** 12 months (`KUDIGUARD_VOICE_RETENTION_DAYS=365`). The purge loop runs every 6 hours, and
  `kudiguard purge` can also be run. A sample is deleted when the week it arrived in is 12 months old, so none is
  kept longer than 12 months.
* **Why the week and not the date:** the week is enough to apply retention and to compare model versions over time.
  It also makes linking a sample to an exact entry in the 90-day evidence log harder. The donor hash still links the
  two, deliberately, so that withdrawal, erasure and access requests can be honoured.

## 2. Lawful basis and conditions
* **Consent (s.25(1)(a), s.26):**
  * *Separate:* its own message, buttons, version and record. KudiGuard works the same without it, so consent is not
    bundled with the service (s.26(2)).
  * *Explicit, affirmative:* a "Yes, I am 18+" button or a ticked box. Silence and "No" mean no (s.26(3), s.26(7)).
  * *Informed:* the offer states the following, in plain words in all four languages (s.26(6)):
    * the purpose;
    * the age limit;
    * own voice only, and forwarded voice notes are never kept;
    * who listens to the recordings;
    * that we never identify speakers;
    * encryption and retention;
    * how to withdraw (s.26(4)).
  * *Versioned:* `DONATION_VERSION`. A "Yes" button under an older offer text shows the current offer again.
* **Withdrawal (s.35):** as easy as giving consent. The user sends "STOP DONATING" (or a phrase in their language),
  taps "No, thanks", or unticks the web box. This stops collection and deletes every donated sample (rows and files),
  and the user gets a confirmation. DELETE does the same and erases everything else too. STOP also ends donation.
  Earlier processing stays lawful (s.26(5)).
* **Sensitive data (s.30, s.65):** s.65 counts biometric data as sensitive when it is processed "for the purpose of
  uniquely identifying a natural person". Its definition lists "voice recognition" as an example. KudiGuard never
  uses voices to identify or verify anyone and will never build speaker identification. We still apply the
  stricter regime: explicit, specific consent (s.30(1)(a)), encryption at rest, and restricted access. **DECIDE:**
  a lawyer should confirm whether s.30 applies.
* **Children (s.31):** the age of majority is 18. Donation is offered to adults only, and the "Yes" button itself
  confirms the age. A "No", or no answer, means no donation. We do not verify age with an ID (s.31(2)–(3)): asking
  for an ID would collect far more data than the risk warrants. **DECIDE:** whether self-declaration is enough.
  Reviewers must delete, and report, any sample that is clearly a child's voice.
* **Third parties:** the offer asks for the donor's own voice only. Forwarded WhatsApp voice notes, flagged by the
  Cloud API's `context.forwarded` / `frequently_forwarded`, are never kept. Forwarded scam audio is usually a
  scammer's or a victim's voice, and those people have not consented. Reviewers delete any sample that clearly
  contains someone else's voice. Text is never donated. The web page records live from the microphone; it cannot
  upload files.

## 3. Necessity and proportionality
* Real WhatsApp voice notes have noise, codecs and code-switching that read-speech sets such as FLEURS lack. Without
  references from these voice notes we cannot measure or improve accuracy, and speech errors lead to wrong scam
  advice.
* Less intrusive options were considered:
  * Asking the team to record speech: done already (FLEURS, TTS clips). It does not represent users.
  * Keeping only transcripts: this cannot measure ASR errors, because the reference must come from the audio.
* The scope is proportionate: opt-in, adults only, own voice only, a sample is transcribed rather than everything,
  12 months, and no identification.

## 4. Security (s.39) and access
* **Encryption at rest:** AES-256-GCM, with a random nonce per file and the sample id authenticated. The key comes
  from `KUDIGUARD_HASH_SECRET` through HKDF-SHA256 with its own label.
  * Files are named by random ids under `KUDIGUARD_VOICE_DIR` (git-ignored `var/voice/`), with mode 0600 on POSIX.
  * A backup of `var/` without `.env` cannot be decrypted.
  * The secret sits on the same host, so this does not protect against a full compromise of the host.
* **Access:** only named reviewers and transcribers under written confidentiality agreements.
  * `kudiguard voice export` writes decrypted copies only to git-ignored folders (`var/`, `exports/`) or outside the
    checkout, and prints a privacy reminder.
  * Transcribers work locally, with no cloud drives, and delete their copies once the references are typed.
  * The operator deletes the export folder after `kudiguard voice wer`.
* **Ordering and robustness:**
  * Writes happen only after the reply is sent and never affect it.
  * Writes and erasure are serialised, so a withdrawal can never be overtaken by a write that is still in progress.
  * Erasure deletes rows, then files. A file left behind by a crash is swept by the next purge.
  * SQLite runs with `secure_delete` and a WAL checkpoint after erasure.
  * **Note:** `KUDIGUARD_VOICE_DIR` must belong to `KUDIGUARD_DB_PATH`. The purge deletes files the database does not
    know about.

## 5. Where the data is (ss.41–43)
* Donated audio is stored only on KudiGuard's own server and never in third-party cloud storage. Transcribers must
  not take copies abroad or upload them anywhere.
* This is a **design choice** that keeps us out of the cross-border transfer rules (NDPA ss.41–43 and GAID 2025,
  already in force). It is not a localisation mandate.
* The CBN's payment-data localisation circular (in force January 2027) covers payment-system participants. It does
  not apply to KudiGuard's donations, though it is relevant to partner banks' own data.
* **In transit:** WhatsApp voice notes pass through Meta's servers, and web and webhook traffic passes through
  Cloudflare's tunnel, before they reach us. Both are disclosed on /privacy.
* **DECIDE:** confirm the physical location of the server (currently the team's PC). If it is outside Nigeria,
  storing donations there is itself a transfer that needs a basis under s.41/s.43 (e.g. informed consent to the
  transfer, stated in the offer), because no NDPC adequacy decisions had been published as of mid-2026.

## 6. Data-subject rights (ss.34–38)
* **Access (s.34(1)(a)–(b)):** `kudiguard voice access --user <number | web:<session id>> [--out var/...]`. This prints
  the consent record and the samples, and copies the audio for a secure hand-over.
* **Erasure and withdrawal (s.34(1)(d), s.35):**
  * On WhatsApp: in-chat commands or buttons.
  * On the web: untick the box (`POST /web/donation/withdraw`).
  * By email: `kudiguard voice erase --user ...`.
* **Objection (s.36):** treated as withdrawal.
* **Automated decisions (s.37):** none are made from donations.
* **Rectification (s.34(1)(c)):** the only free text is the automatic transcript, which is not used about the person.
* **Contact:** `KUDIGUARD_PRIVACY_CONTACT`, shown on /privacy.

## 7. Risks
| Risk | Likelihood / impact | Mitigations | Residual |
|---|---|---|---|
| Re-identification by voice (someone recognises the speaker) | Medium / medium | Few named listeners under confidentiality; no phone numbers; random ids; no speaker identification; 12-month limit | Low–medium (voice is inherently identifying) |
| Third-party voices (forwarded scam calls, family members) | Medium / medium | Forwarded voice notes excluded; own-voice wording; reviewers delete others' voices | Low |
| Children donating | Low / high | 18+ in the offer, button and web box; no answer = no; reviewers delete | Low (self-declared) |
| Sensitive content (health, money troubles) in the speech | Medium / medium | Purpose limited to ASR; hypothesis redacted; access limited; withdrawal deletes | Low–medium |
| Breach (stolen disk, leaked backup, leaked export) | Low / high | Encryption at rest; key derived from a secret kept out of backups; git-ignored exports; deleting export copies | Low |
| Function creep (training other models, sharing) | Low / medium | Single stated purpose; a new use needs a new consent version | Low |
| Consent not demonstrable | Low / medium | Versioned record per user and per sample; record kept while samples exist | Low |

Overall residual risk: **low to medium, acceptable** with the controls above. It is not high, so prior consultation
with the NDPC (s.28(2)) is not expected. **DECIDE:** the lawyer confirms this.

## 8. Incident procedure (s.40)
1. **Contain:**
   * Revoke access.
   * Rotate `KUDIGUARD_HASH_SECRET` if the secret may have leaked. Rotation makes stored samples undecryptable;
     purge them.
   * Stop the server if the leak is ongoing.
2. **Assess:** what leaked (audio, rows, exports), how many donors, and whether the data was encrypted.
3. **Notify the NDPC within 72 hours** of becoming aware, if the breach is likely to risk people's rights. Partial
   information may be given in phases (s.40(2), (9)).
4. **Tell affected users** in plain language if the risk is high (s.40(3)). WhatsApp donors can be messaged through
   the bot only within 24 h of their last message. Otherwise post a notice on /privacy.
5. **Record** every breach, including ones not notified (s.40(8)), in `var/incidents.md`. Keep it out of git.

## 9. Controller obligations
* **Controller:** `KUDIGUARD_OPERATOR_NAME`. **Privacy contact or DPO:** `KUDIGUARD_PRIVACY_CONTACT`. Under s.32, a DPO
  is mandatory only for data controllers of major importance (DCPMIs).
* **Registration (s.44; GAID Arts. 8–10; NDPC Guidance Notice NDPC/HQ/GN/VOL.02/24):** the tiers count data subjects
  over 6 months.

  | Tier | Data subjects in 6 months |
  |---|---|
  | OHL | more than 200 |
  | EHL | more than 1,000 |
  | UHL | more than 5,000 |

  A pilot with more than 200 users in 6 months probably reaches OHL: register, and renew every year. EHL and UHL
  also file annual compliance audit returns through a licensed DPCO. **DECIDE:** registration before the pilot
  exceeds 200 users.
* **Record of processing activities** (GAID Art. 13 semi-annual report). Entry:

  | Field | Entry |
  |---|---|
  | Activity | Voice-note donation for speech-recognition testing |
  | Data subjects | Adult users who opt in |
  | Categories | Voice audio, pseudonymous id, language, channel, week, redacted transcript, consent record |
  | Basis | Consent (s.25(1)(a), s.30(1)(a)) |
  | Recipients | Named reviewers and transcribers under written confidentiality |
  | Transfers | None by design (see §5) |
  | Retention | 12 months |
  | Security | §4 |

* **Note on GAID article numbers:** check them against the official GAID 2025 text before filing. Secondary sources
  disagree on the numbering from Art. 19 onwards.

## 10. Sign-off
| Role | Name | Date | Decision |
|---|---|---|---|
| Operator (controller) | | | |
| Privacy contact / DPO | | | |
| Legal review (Nigerian data-protection counsel) | | | |

Review this assessment when the purpose, the retention period, the listeners, the server location or the offer text
changes. Bump `DONATION_VERSION` whenever the offer text changes.

Sources:
* NDPA 2023, section by section: <https://www.lawglobalhub.com/?p=66857>. Definitions in s.65:
  <https://www.lawglobalhub.com/section-65-nigeria-data-protection-act-2023/>.
* FPF summary: <https://fpf.org/blog/nigerias-new-data-protection-act-explained/>.
* GAID 2025:
  * <https://www.aelex.com/the-nigeria-data-protection-act-general-application-and-implementation-directive-2025/>
  * <https://www.doa-law.com/wp-content/uploads/2025/10/Gaid-Serie-1.pdf>
  * <https://blog.lawpavilion.com/ndp-act-2023-gaid-2025-a-comprehensive-guide-to-nigerias-new-data-protection-landscape>
* Registration tiers:
  * <https://www.michalsons.com/blog/nigeria-data-protection-act-ndpa-controller-and-processor-registration/77465>
  * <https://ndpc.gov.ng/Files/registration.pdf>
* Cross-border transfers:
  * <https://www.mondaq.com/nigeria/privacy-protection/1793876/cross-border-data-transfers-under-nigerias-ndpa-a-compliance-guide>
  * <https://www.dlapiperdataprotection.com/?t=transfer&c=NG>
* CBN circular, not applicable:
  * <https://techeconomy.ng/cbn-orders-banks-fintechs-to-store-payment-data-in-nigeria-by-january-2027/>
  * <https://digitalpolicyalert.org/event/40846-circular-introducing-systemic-oversight-measures-in-payments-system-including-data-localisation-obligations-enters-into-force>
