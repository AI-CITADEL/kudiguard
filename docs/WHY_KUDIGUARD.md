# Why KudiGuard

**"Not sure? Forward it to KudiGuard."** A scam verifier in your pocket: send a voice note or forward a suspicious
message on WhatsApp, in Hausa, Yoruba, Igbo or English, and get a 🔴 / 🟠 / 🟢 answer with what to do and where to
report, in seconds. Built on N-ATLAS (Awarri / NCAIR) for the NAIC 2026 build challenge.

## How it works

```
User's phone (normal WhatsApp)
   │  voice note / text / forwarded message to the KudiGuard number
   ▼
Meta WhatsApp Cloud ──webhook──► KudiGuard server
   1. verify Meta's signature; mark read + "typing…"
   2. NCAIR1 speech model → text (voice notes)
   3. N-ATLAS: translate to English, understand intent, suggest warning signs
   4. deterministic red-flag rules → verdict (never "safe")
   5. verified knowledge base → N-ATLAS writes advice in the user's language
   6. reply + buttons: What should I do? · Where to report · Warn my family · 👍/👎
```

No human operator and no autonomous "agent": a fixed, tested pipeline. The user installs nothing. One number serves
everyone in parallel. Banks, fintechs and NGOs can call the same engine through the REST API (`/v1/check`).

## What it can be used for

1. **Personal scam check**: the core use. Anyone forwards a suspicious SMS or WhatsApp message, or describes a call by
   voice, in their own language, and gets 🔴/🟠/🟢 plus what to do, before they send money or an OTP.
2. **Banks and fintechs, via the API**: screen messages customers report, power the bank's own WhatsApp or app
   "Is this real?" button, or check a transfer narration before payout. One integration protects millions of customers.
3. **Telcos (MTN, Airtel, Glo)**: flag scam SMS in bulk, and point users to the NCC 622 complaint flow.
4. **Call centres / IVR**: the same engine answering "is this call a scam?" on a phone line, for people without
   smartphones (backlog).
5. **Scam-trend intelligence**: the de-identified pattern library reveals new scams by language early (region later),
   for EFCC, CBN, NCC, banks and the media. It is also a sellable dashboard.
6. **Financial literacy at scale**: cooperatives, NGOs, schools and religious groups. "Ask anything about money" in
   Hausa, Yoruba or Igbo by voice reaches low-literacy users that apps miss.
7. **Family protection**: the "Warn my family" button spreads warnings to elderly parents, who are the most targeted.
8. **Other languages and countries**: languages are configuration, so Ghana, Kenya or Pidgin need data, not a rebuild.

## How the server decides "scam or not"

Two layers, on purpose:

1. **N-ATLAS understands the message** in any of the four languages: it translates to English, works out whether the
   user wants a check or is asking a money question, and suggests warning signs.
2. **Transparent red-flag rules decide the verdict.** Thirteen scam patterns are matched in the original language and
   the English translation, each with a weight:

| Red flag (examples) | Points |
|---|---|
| Asks for OTP / PIN / BVN / password; "install AnyDesk"; "double your money, guaranteed"; "I sent money to you by mistake, send it back" | 3 |
| "Your account will be blocked"; pay a fee to receive a loan, job or prize; "you won"; fake credit alert; asks for money; strange links | 2 |
| Claims to be bank staff / CBN / EFCC; "do it now"; "don't tell anyone" | 1 |

* **3+ points** → 🔴 *This looks like a SCAM* · **2 points**, or any sign in a check → 🟠 *Be careful* ·
  **no signs** → 🟢 *No red flags found. Still confirm with your bank.* KudiGuard never says "it's safe".
* Example: *"Bank staff called, said my account will be blocked, asked for my OTP"* → impersonation (1) + account
  threat (2) + credentials (3) = 6 → 🔴.
* Model-only warning signs (when the rules found none) are used cautiously: they are capped at 🟠 unless the user
  clearly asked for a check, and are ignored when the model's translation failed. Verified contacts are inserted by the server, never written by the
  model. If N-ATLAS is down, the rules still answer.
* New phrasings of known scam patterns, in any language, are added by editing `data/languages/<code>.yaml`
  (configuration, not code). A genuinely new *type* of red flag is a small code change (the `Flag` list and its
  weight in `safety/scam_rules.py`).

Measured on 40 blind messages (never seen during development): **95% of scams flagged red or orange, 86%
precision** with N-ATLAS; 65% with rules alone. N-ATLAS translation is what lifts Hausa and Yoruba (docs/SPIKES.md S7).

## Why Hausa, Yoruba and Igbo, and why WhatsApp rather than an app?

* **It's what the challenge scores.** NAIC is built around N-ATLAS, whose whole point is Nigerian languages and voice
  (Hausa, Yoruba, Igbo, Nigerian-accented English, plus NCAIR's speech models). Judges score "depth of N-ATLAS
  integration". An English-only scam app could be built with any generic AI and would score poorly on exactly that.
* **It's where the victims are.** The people scammers target most are often more comfortable in their own language,
  may read less easily, and prefer voice. Scam calls increasingly use local languages. An English text app misses them.
* **Reach and evidence.** NAIC wants a working product with real users (≥50 interactions):
  * **WhatsApp:** almost every Nigerian phone already has it. No install, works on cheap phones, users can start today.
  * **An app:** download, storage, Play Store review and user trust. Getting 50+ real users in 4 days would be very hard.
* **An app isn't wrong, it's the next step.** An auto-detect app (below) would use the same rules, languages and API.
  WhatsApp proves the engine with real users now, and the app expands it later.

## Roadmap: from "forward to check" to automatic protection

| Where the scam arrives | Auto-check? | How |
|---|---|---|
| SMS (fake bank alerts, "BVN blocked") | ✅ | Android app with SMS/notification access warns on arrival; iPhone SMS filter extension flags unknown-sender texts. Rules are plain config, so they can run on the phone (private, offline), with the server for harder cases |
| Phone calls (fake bank staff) | ✅ partly | Android call screening for known scam numbers; a live voice agent (call-centre backlog) |
| WhatsApp chats | ❌ directly | End-to-end encrypted; no app may read other chats, by design. Forwarding to KudiGuard is the legitimate path |
| At scale, before anyone sees it | ✅ best impact | Banks screen transfers/beneficiaries and telcos filter scam SMS through the KudiGuard API |

Today "forward it to check" → an Android/iOS auto-warning app → bank and telco integration: one engine, many channels.
