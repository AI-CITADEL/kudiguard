# DPIA: opt-in voice-note donation

This is a data protection impact assessment under the Nigeria Data Protection Act 2023 (NDPA) s.28, and under the
NDPC's General Application and Implementation Directive (GAID 2025), in force since 19 September 2025.

**Status: draft for the operator's sign-off.** The feature stays off (`KUDIGUARD_VOICE_DONATION=false`) until §10 is
signed. This is an engineering assessment, not legal advice. **Before scaling beyond the pilot, have a qualified
Nigerian data-protection lawyer or a licensed DPCO review it.** Items marked **DECIDE** need the operator or a lawyer.
Section numbers come from reproductions of the Act, because the official PDF could not be fetched; check them
against the gazetted text before filing.

## 1. What is processed, and why
* **Purpose (only one):** to test and improve speech recognition for Hausa, Yoruba, Igbo and Nigerian English.
  Native-speaker transcribers type what was said, and `kudiguard voice wer` measures the word error rate. Nothing
  else is done with the recordings (s.24, purpose limitation).
* **Who:** adults who opt in. There are three ways to say yes. Each one confirms 18+ and own voice, and shows the
  same text and version (`DONATION_VERSION`).
  * *WhatsApp:* the offer comes after an answered voice note that was not a scam warning, or when the user sends
    "donate voice". The user taps "Yes, I am 18+". The offer is shown once per offer text.
  * *Website:* a separate checkbox, unticked by default, sent with its own version header.
  * *Tester sign-up form:* see `docs/TESTER_SIGNUP.md`. The operator imports the form with `kudiguard voice
    preconsent`.
    * Each row needs a valid Nigerian mobile number, all three boxes ticked and a past timestamp; otherwise
      nothing is imported.
    * A form answer never overrides a later STOP, withdrawal, decline or chat activity, and answers older than 30
      days are skipped.
    * A STOPped user who returns starts without donation consent.
    * Imported testers are not asked again in the chat.
    * An import never implies main consent: the tester's first message, whatever it is, gets the welcome, which
      carries the retention, reviewer and transfer notice. Main consent is recorded only after the current welcome
      was *delivered* (`welcome_version`, `welcome_at`).
    * Consent comes only from a message sent at least 5 s after that delivery, judged by WhatsApp's timestamp. A
      message sent earlier gets a pointer to the notice and is not processed.
    * The import itself is a compare-and-set: it writes nothing if the chat changed the row after the import read
      it.
* **What we keep (s.24, minimisation):**
  * the original audio, at most 16 MB and 90 s of speech;
  * a random sample id;
  * the donor's keyed hash (`privacy.UserHasher`), never the phone number;
  * language, channel and ISO week (no timestamp);
  * the version of the donation text accepted;
  * the duration, when known (not recorded on the web);
  * the automatic transcript, with numbers and emails masked (`text.redact`).
* **What is never kept:**
  * Text messages.
  * WhatsApp voice notes not recorded in the app (Meta's `audio.voice` is not true), or marked forwarded
    (`context.forwarded` / `frequently_forwarded`).
  * Anything received before the yes.
* **Consent record (s.26, burden of proof):**
  * *Chat and form:* the user row stores `donation_version`, `donation_at` and `donation_source` (chat or form).
  * *Web:* web donors have no account. Their proof is the `consent_version` stored on every sample (N6).
  * A donor's user row is kept while they have samples.
* **Retention (s.24):** 12 months. `KUDIGUARD_VOICE_RETENTION_DAYS` is capped at 365, the figure the offer promises.
  * The purge runs every 6 hours and on `kudiguard purge`.
  * A sample is deleted once the week it arrived in is 12 months old, so none outlives the promise.
* **Why the week and not the date:** the week is enough for retention and for comparing model versions. It also
  makes it harder to link a sample to an exact entry in the 90-day evidence log. The donor hash still links the
  two, deliberately, so that withdrawal, erasure and access requests can be honoured.

## 2. Lawful basis and conditions
* **Consent (s.25, s.26):**
  * *Separate:* its own text, buttons or boxes, version and record. KudiGuard works the same without it, so consent
    is not a condition of the service.
  * *Explicit and affirmative:* silence, no answer and "No" all mean no. No box is pre-ticked.
  * *Informed:* the offer states the following, in plain words in all four languages:
    * the purpose;
    * 18+ only, own voice only;
    * who listens: reviewers and transcribers under written confidentiality;
    * that we never identify speakers;
    * encryption, 12-month retention, and the storage location (§5);
    * how to withdraw.
  * *Versioned:* when the offer text changes, the new `DONATION_VERSION` pauses collection for old donors. They are
    offered the new text once, and nothing more is kept until they answer.
  * *Survives the main notice:* a change of the main consent text does not end donation. Only STOP, STOP DONATING,
    "No" and DELETE end it.
* **Withdrawal (s.35):** as easy as giving consent.
  * WhatsApp: STOP DONATING, the "No, thanks" button, STOP, or DELETE.
  * Web: untick the box, or press "Delete my donated recordings".
  * Each of these stops collection **and deletes every donated sample** (rows and files) and sends a confirmation.
  * A withdrawal leaves a tombstone: a write that is still in progress, whose consent predates the withdrawal, is
    refused.
  * Processing done before the withdrawal stays lawful.
* **Sensitive data (s.30, s.65):**
  * s.65 counts biometric data as sensitive when it is processed for unique identification, and lists "voice
    recognition" as an example. KudiGuard never identifies or verifies speakers and will never build speaker
    identification.
  * A voice can also reveal race or ethnic origin, another s.65 category. Accent and language are exactly what we
    study, so we treat the recordings as potentially sensitive.
  * We apply the stricter regime anyway: explicit, specific consent, encryption at rest, and restricted access.
  * **DECIDE:** a lawyer should confirm the s.30 position.
* **Children (s.31):**
  * The age of majority is 18. Every route to a yes includes an 18+ confirmation, and "No" or no answer means no.
  * Age is self-declared. Verifying it by ID would collect far more data than the risk warrants. **DECIDE:**
    whether that is enough.
  * Reviewers delete (`kudiguard voice remove`) any recording that is clearly a child's voice.
* **Other people's voices:**
  * The offer asks for the donor's own voice only.
  * Meta documents `voice` (recorded in the app) on audio messages, but documents `context.forwarded` only for
    other message types. So exclusion of forwarded voice notes is **best effort**, and the texts say "we try not to
    keep forwarded voice notes".
  * Reviewers delete other people's voices.
  * **Before enabling,** confirm on the live number with one real forwarded voice note. The log line
    `voice message: recorded=… forwarded=…` shows which flags Meta sent. Record the result here.
  * The web page records live from the microphone. It has no file upload.

## 3. Necessity and proportionality
* Real WhatsApp voice notes have noise, codecs and code-switching that read-speech sets lack. Without references
  from them we cannot measure or improve accuracy, and speech errors lead to wrong scam advice.
* Alternatives were considered:
  * Team recordings and public data (FLEURS, TTS): already used, but they do not represent users.
  * Keeping only transcripts: this cannot measure ASR errors, because the reference must come from the audio.
* Proportionate scope: opt-in, adults only, own voice only, a sample transcribed rather than everything, 12
  months, no identification, per-donor daily cap (20) and a folder size cap.

## 4. Security (s.39) and access
* **Encryption at rest:** AES-256-GCM, with a random nonce per file and the sample id authenticated.
  * The key comes from `KUDIGUARD_VOICE_KEY` through HKDF-SHA256 with its own label. If that is not set, it comes
    from `KUDIGUARD_HASH_SECRET`.
  * A copy of `var/` without `.env` cannot be decrypted.
  * The key is on the same host, so this does not protect against a full compromise of the host.
* **Key rotation:**
  1. `kudiguard voice wipe --all` deletes every sample. They would be undecryptable after the rotation anyway.
  2. Change `KUDIGUARD_VOICE_KEY`, or the hash secret.
  3. Restart.
* **File system:**
  * Files are named by random ids under `KUDIGUARD_VOICE_DIR` (default `var/voice/`, git-ignored).
  * On POSIX: mode 0600, the directory is fsynced after each write.
  * On Windows the mode bits do nothing, so restrict `var\` and `.env` to the service account with `icacls`
    (docs/DEPLOY.md).
  * A `.vault` marker binds the folder to its database. Purges sweep orphan files only when the marker matches,
    and writes refuse a folder that belongs to another database. This safety check replaces a cruder "database
    empty but folder has files" rule, which would also leave genuinely erased-but-locked audio behind forever.
* **Access:** only named reviewers and transcribers under written confidentiality.
  * `kudiguard voice export` writes decrypted copies only under `var/` (next to the database) or `exports/`. It
    refuses cloud-synced folders (OneDrive, Dropbox, Google Drive, iCloud, Box) and prints a privacy reminder.
  * Transcribers work locally, with no cloud drives and no copies abroad, and delete their copies once the
    references are typed. The operator deletes the export folder after `kudiguard voice wer`.
* **Robustness:**
  * Writes run after the reply, in a background task that does not hold a turn slot. They never affect the reply.
  * The per-donor daily cap, the withdrawal tombstones and the dropped-sample counters are kept in memory. A
    restart resets them, which only loosens the daily cap. A withdrawal's deletion itself is in the database and
    is permanent. Dropped samples are counted in the log lines (`voice donation not kept: <reason> (<n> so far)`).
  * Writes are serialised with erasure.
  * Erasure deletes rows, then files.
  * SQLite runs with `secure_delete` and a WAL checkpoint after erasure.
  * A live turn and `kudiguard voice erase` run in different processes and are not serialised. Run erasure
    requests when the donor is not mid-conversation, or simply re-run them (they are idempotent).

## 5. Where the data is (ss.41–43): transfer outside Nigeria
* **During the pilot the server is the team's PC, outside Nigeria.** It will move to Nigeria or to the cloud later.
* Messages, including voice notes, pass through **WhatsApp (Meta)** and **Cloudflare**, both outside Nigeria, and
  are stored on our server.
* Every user is told this before agreeing:
  * the welcome text and web consent in four languages;
  * the donation offer;
  * /privacy, "Where your data goes".
* Each text states the transfer risk for the configured location. While the server is abroad: "Nigerian
  data-protection law may not fully protect data held abroad". In Nigeria: "data passing through Meta and
  Cloudflare may be processed abroad". Continuing (or ticking
  the box) is the user's informed consent to the transfer (s.43). No NDPC adequacy decision had been published as
  of mid-2026, and these rules are already in force under the NDPA and GAID 2025.
* **The location is a setting:** `KUDIGUARD_STORAGE_LOCATION=outside-nigeria` (default) or `nigeria`. Every text
  renders the setting, so it stays true after the move. Nothing claims "stored in Nigeria" by default.
* Moving the server abroad again requires a new `CONSENT_VERSION`. Moving it into Nigeria needs no new consent, but
  the transfer through Meta and Cloudflare remains.
* The CBN payment-data localisation circular (in force January 2027) covers payment-system participants. It does
  not apply to KudiGuard, though it is relevant to partner banks' own data.
* Transcribers must not take copies abroad or upload them anywhere.

## 6. Data-subject rights (ss.34–38)
* **Access (s.34):** `kudiguard voice access --user <number | web:<donation code>> [--out var/...]` prints the
  consent record (version, time, source) and the samples, and copies the audio for a secure hand-over.
* **Identity check:** answer only the person behind the number or code.
  * For a WhatsApp number, message that number and ask them to confirm.
  * Web donors see their donation code when they tick the box.
  * Never accept a number alone from an email.
* **Erasure and withdrawal (s.34, s.35):**
  * In chat: STOP DONATING, STOP, DELETE or "No".
  * On the web: untick the box, or press "Delete my donated recordings". This works only in the same browser;
    otherwise the donor quotes the donation code.
  * By request: `kudiguard voice erase --user …`.
  * **DELETE marker (s.34 compliance record, s.24 minimisation):** DELETE leaves only a keyed HMAC of the user's
    hash (its own label, no number and no content) with the time.
    * It is read only by the sign-up import, so an earlier form answer can never re-enrol someone who deleted.
    * It is purged within 30 days: the purge loop runs every 6 hours and drops markers one interval early.
    * Form answers older than 29 days are never imported, so a marker always outlives the answers it must block.
    * It is disclosed on /privacy under DELETE.
  * DELETE also resets the "offered" mark (N5), so a returning user may be offered once more.
* **Objection (s.36):** treated as withdrawal.
* **Automated decisions (s.37):** none are made from donations.
* **Rectification:** the only free text is the automatic transcript, which is not used about the person.

## 7. Risks
| Risk | Likelihood / impact | Mitigations | Residual |
|---|---|---|---|
| Re-identification by voice | Medium / medium | Few named listeners under confidentiality; no numbers; random ids; no speaker identification; 12 months | Low–medium |
| Third-party voices (forwarded scam calls, family members) | Medium / medium | `audio.voice` required; `forwarded` checked; own-voice wording; reviewers delete | Low–medium (forwarded flag not documented for audio) |
| Children donating | Low / high | 18+ on every route; no answer means no; reviewers delete | Low (self-declared) |
| **Spoken financial identifiers** (OTP, PIN, BVN, account numbers read aloud) | Medium / high | /privacy asks donors not to read them out; transcripts masked; transcribers type numbers as `[number]`; reviewers `remove` samples with credentials | Low–medium |
| Sensitive content (health, money troubles, ethnicity through accent) | Medium / medium | Single purpose; restricted access; withdrawal deletes | Low–medium |
| Transfer abroad (pilot server, Meta, Cloudflare) | Certain / medium | Informed consent (s.43); encryption at rest; move to Nigeria planned | Medium until the move |
| Breach (stolen disk, leaked backup, leaked export) | Low / high | Encryption; key kept out of backups; exports only to local git-ignored folders and deleted after use | Low |
| Function creep | Low / medium | Single stated purpose; a new use needs a new consent version | Low |
| Consent not demonstrable | Low / medium | Versioned record per user and per sample; source recorded; form rows validated | Low |

Overall residual risk: **medium during the pilot**, mainly because of the transfer abroad. It falls to low after
the move to Nigeria. It is not "high", so prior consultation with the NDPC (s.28) is not expected. **DECIDE:** the
lawyer confirms this.

## 8. Incident procedure (s.40)
1. **Contain:**
   * Revoke access.
   * If the key may have leaked: `kudiguard voice wipe --all`, then rotate the key (§4).
   * Stop the server if the leak is ongoing.
2. **Assess:** what leaked (audio, rows, exports), how many donors, and whether the data was encrypted.
3. **Notify the NDPC within 72 hours** of becoming aware, if the breach is likely to risk people's rights. Partial
   information may follow in phases.
4. **Tell affected people immediately**, in plain language, when the risk to them is high.
   * WhatsApp donors: through the bot, but only within 24 hours of their last message. Otherwise use another
     channel the operator holds.
   * Everyone: also post a notice on /privacy.
5. **Record** every breach, including ones not notified, in `var/incidents.md`. Keep it out of git.

## 9. Controller obligations
* **Controller:** `KUDIGUARD_OPERATOR_NAME`.
* **Privacy contact:** `KUDIGUARD_PRIVACY_CONTACT`. A DPO is mandatory only for data controllers of major
  importance (s.32).
* **Registration (s.44; GAID; NDPC Guidance Notice NDPC/HQ/GN/VOL.02/24):** the tiers count data subjects over 6
  months.

  | Tier | Data subjects in 6 months |
  |---|---|
  | OHL | more than 200 |
  | EHL | more than 1,000 |
  | UHL | more than 5,000 |

  A pilot with more than 200 users in 6 months probably reaches OHL: register, and renew every year. **DECIDE:**
  register before then.
* **Record of processing activities** (GAID semi-annual report). Entry:

  | Field | Entry |
  |---|---|
  | Activity | Voice-note donation for speech-recognition testing |
  | Data subjects | Adult users who opt in (chat, web, tester form) |
  | Categories | Voice audio, pseudonymous id, language, channel, week, redacted transcript, consent record (version, time, source, withdrawal time); DELETE markers (keyed HMAC and time, 30 days) |
  | Basis | Consent (s.25, s.30 explicit consent) |
  | Recipients | Named reviewers and transcribers under written confidentiality |
  | Transfers | Outside Nigeria during the pilot: our server, and Meta and Cloudflare in transit. Basis: informed consent (s.43) |
  | Retention | 12 months |
  | Security | §4 |

## 10. Enabling, and sign-off
To switch the feature on:
1. Sign below.
2. Have native speakers review the ha/yo/ig texts and the local command words. Until then only the English
   commands are certain to be understood (N4).
3. Confirm the forwarded-voice-note flag (§2).
4. Restrict file permissions (§4).
5. Set `KUDIGUARD_VOICE_DONATION=true` and restart.

| Role | Name | Date | Decision |
|---|---|---|---|
| Operator (controller) | | | |
| Privacy contact / DPO | | | |
| Legal review (Nigerian data-protection counsel) | | | |

Review this assessment, and bump `DONATION_VERSION`, when the purpose, the retention period, the listeners, the
server location or the offer text changes.

Sources:
* NDPA 2023, section by section: <https://www.lawglobalhub.com/?p=66857>. Definitions in s.65:
  <https://www.lawglobalhub.com/section-65-nigeria-data-protection-act-2023/>.
* FPF summary: <https://fpf.org/blog/nigerias-new-data-protection-act-explained/>.
* GAID 2025:
  * <https://www.aelex.com/the-nigeria-data-protection-act-general-application-and-implementation-directive-2025/>
  * <https://www.doa-law.com/wp-content/uploads/2025/10/Gaid-Serie-1.pdf>
* Registration tiers:
  * <https://www.michalsons.com/blog/nigeria-data-protection-act-ndpa-controller-and-processor-registration/77465>
  * <https://ndpc.gov.ng/Files/registration.pdf>
* Cross-border transfers:
  * <https://www.mondaq.com/nigeria/privacy-protection/1793876/cross-border-data-transfers-under-nigerias-ndpa-a-compliance-guide>
  * <https://www.dlapiperdataprotection.com/?t=transfer&c=NG>
* CBN circular, not applicable:
  * <https://techeconomy.ng/cbn-orders-banks-fintechs-to-store-payment-data-in-nigeria-by-january-2027/>
  * <https://digitalpolicyalert.org/event/40846-circular-introducing-systemic-oversight-measures-in-payments-system-including-data-localisation-obligations-enters-into-force>
* Meta, audio message fields (`voice`):
  <https://developers.facebook.com/docs/whatsapp/cloud-api/webhooks/payload-examples>.
