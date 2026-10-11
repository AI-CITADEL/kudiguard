# dev2 evaluation set (FROZEN 2026-10-11)

130 cases in `cases.yaml`: 45 genuine, 50 scam, 35 question.
By language: en 60, pcm 25, ha 15, yo 15, ig 15.

## Provenance

- Written on 2026-10-11 by an independent test author working **blind**.
  The author did not read `src/`, `data/packs/`, `data/languages/`, `data/eval/` (other than
  this folder), `tests/` or `docs/`. Only the case schema was given.
- **Frozen: do not tune on it.** Use it only to measure fixes. If a case is wrong, fix it in a
  separate, reviewed commit that records the reason, and do not change it to make the system pass.
- All identifiers are fictitious: phones are `0800 000 00xx`, the account is `0123456789`, and URLs
  are defanged `hxxp` look-alikes. Lender, shop, school, church and person names are invented.
  Short codes such as `*000*0#` are placeholders.

## DRAFT status of non-English text

- **pcm** (Nigerian Pidgin): written by the author. Native speakers have not reviewed it.
- **ha, yo, ig**: **DRAFT**. The author wrote these to sound natural and did not translate them
  in parallel from each other. A native speaker has **not** checked them. Expect errors in
  tone marks, spelling and idiom. Have them reviewed by native speakers before relying on
  per-language scores. Bank and telco SMS embedded in them stay in English on purpose, as they
  would in real life.
- `must_advise`, `must_not` and `note` are in English for every language.

## Field notes

- `accept`: other verdicts that are also acceptable. For example, `[genuine, caution]` on
  church, mosque and school accounts shared by a known person.
- `must_advise`: 1 to 3 specific points that a good answer should cover. They are judged by
  meaning, not by exact wording.
- `must_not`: an answer must never say any of these, in substance. Every genuine case includes
  "This is a scam".
- Loan-app debt-shaming (d2-en-032, d2-pcm-015) is labelled `scam` as a harmful practice, even
  though the debt may be real.

## Facts relied on (web-checked 2026-10-11)

- **CBN new-device limit.** A newly activated mobile banking app is limited to ₦20,000 in the
  first 24 hours. For an existing customer on a new device, the limit covers outflows only.
  Effective 1 July 2026. Sources: [BusinessDay](https://businessday.ng/news/article/cbn-caps-new-mobile-banking-transfers-at-n20000-for-24-hours/),
  [Proshare](https://proshare.co/articles/cbn-introduces-new-security-measures-and-customer-controls-for-instant-payments),
  [Leadership](https://leadership.ng/cbn-caps-mobile-banking-transactions-at-n20000-within-first-24-hours-on-new-devices/).
- **₦50 stamp duty.** Charged on electronic transfers of ₦10,000 and above. Under the Nigeria
  Tax Act 2025, the sender has paid it since 1 January 2026. Sources:
  [TechCabal](https://techcabal.com/2025/12/22/why-your-50000-transfer-will-cost-100-from-2026/),
  [Techeconomy](https://www.techeconomy.ng/stamp-duty-replaces-emtl-sender-to-bear-50-charge-on-bank-transfers),
  [Radio Nigeria](https://radionigerialagos.gov.ng/?p=35872).
- **NDIC cover.** ₦5m for deposit money banks, mobile money operator pass-through and non-interest
  banks. ₦2m for MFBs, PSBs and PMBs. Set in May 2024. Sources:
  [BusinessDay](https://businessday.ng/news/article/ndic-coverage-now-protects-99-of-nigerian-depositors/),
  [AllAfrica](https://allafrica.com/stories/202510030020.html).
- **FCCPC DEON Regulations 2025.** Digital lenders must register, and harassment or defamation of
  borrowers is prohibited. Phased enforcement began after the 5 Jan 2026 deadline. The
  regulations were briefly suspended in a legal challenge; the court then dismissed the suit
  against the FCCPC. The cases avoid relying on specific clause numbers. Sources:
  [Prime Business Africa](https://www.primebusiness.africa/?p=236419),
  [Nigeria Housing Market](https://www.nigeriahousingmarket.com/news/fccpc-loan-app-deadline-lapse-521-companies-radar-2026),
  [ITEdgeNews](https://www.itedgenews.africa/fccpc-tightens-noose-on-loan-apps-why-non-enforcement-puts-nigerian-consumers-at-serious-risk).
- **Complaints and escalation.** Complain to your own bank first and get a reference. If it is
  not resolved, escalate to the CBN Consumer Protection Department. Exact timelines vary by
  channel, so the cases do not assert a specific number of days. Sources:
  [Nairametrics](https://nairametrics.com/2022/04/25/tired-of-bank-issues-here-is-how-to-report-your-bank-to-the-cbn/),
  [Arise](https://www.arise.tv/nigerian-lawmakers-urge-cbn-to-direct-banks-on-prompt-reversal-of-failed-transactions/).
- **General knowledge, not separately web-checked in this session:**
  - Bail is free.
  - EFCC petitions are free.
  - Banks never ask for a PIN or OTP.
  - NELFUND applications are free.
  - WhatsApp re-registration and two-step verification.
  - SEC register of capital market operators.
  - Card maintenance fee and SMS cost-recovery charges under the CBN Guide to Charges.
