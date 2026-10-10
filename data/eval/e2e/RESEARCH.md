# E2E test set: scam research note

Written 2026-10-10 by an independent test author, before and without reading any KudiGuard code, rules, prompts or
existing test sets. Sources come from web searches run on 2026-10-10. Dates are publication dates where the source
showed one. "Undated" means the search result did not show a date. Several items are older than 2025 but are still
cited in 2025–2026 alerts as live patterns. The `scam_type` ids used in `cases.yaml` are in `code` style.

## Bank and identity

| scam_type | What happens | Sources |
|---|---|---|
| `bank_impersonation` | SMS or call claiming to be a bank: the account is "restricted" or "blocked", or a debit is "pending". It asks for card number, PIN, OTP or BVN, or for a call to a number in the message. Banks say they never ask for PIN, OTP, BVN, CVV or password. | Access Bank via TheCable (undated) https://www.thecable.ng/?p=463366 ; CBN BVN press statement (2015, still cited) https://www.cbn.gov.ng/Out/2015/CCD/CBN%20Press%20Statement%20on%20BVN.pdf |
| `bvn_nin_phishing` | "Your BVN has been flagged for deactivation" or "update BVN" with a link. | CBN statement on fake messages (Apr 2026) https://businessday.ng/markets/article/cbn-flags-fake-messages-warns-public-of-cyber-fraud-attempts/ ; https://isthisascam.to/blog/scams-in-nigeria |
| `nin_correction_portal` | Fake "free NIN correction or modification portal" (e.g. a gvly.xyz flyer) that harvests NIN, BVN and personal data. NIMC says to use only its self-service portal. | ThisDay (10 Mar 2025) https://thisdaylive.com/index.php/2025/03/11/nimc-modify-your-data-on-our-portal-or-risk-identity-theft ; Gazette NGR (2026) https://gazettengr.com/nimc-warns-nigerians-against-fake-nin-correction-portal/ ; Leadership https://leadership.ng/nimc-raises-the-alarm-over-fake-nin-websites-targeting-nigerians-personal-data/ |
| `safe_account_scam` | After the Zenith Bank breach (Aug 2026) exposed names, phones and emails, follow-on vishing calls use the real details: "your account is compromised, move funds to a safe account" or "give the OTP". | Nairametrics (4 Aug 2026) https://nairametrics.com/2026/08/04/zenith-bank-alerts-customers-after-cyber-incident-exposed-contact-details/ ; Techeconomy (Aug 2026) https://techeconomy.ng/zenith-bank-confirms-data-breach-as-cyberattacks-on-nigerian-banks-intensify |
| `fake_customer_care` | Fake bank "customer care" accounts on X, Facebook or Telegram reply to public complaints, move the chat to WhatsApp and ask for card details, PIN and OTP, or offer fake loans. Wema shut its Telegram presence over impersonation (Jun 2026). | Dubawa (Zenith case, 2022) https://dubawa.org/how-scammers-pose-as-zenith-banks-twitter-help-page-to-defraud-unsuspecting-customers/ ; Africa Check (GTBank fake FB pages, 2026) https://africacheck.org/fact-checks/meta-programme-fact-checks/nigerians-ignore-fake-facebook-accounts-posing-gtbank-and ; Legit (Wema, Jun 2026) https://www.legit.ng/business-economy/industry/1716886-wema-bank-suspends-social-media-account-impersonation/ |
| `remote_access_support` | Fake support asks the victim to install a screen-sharing app (AnyDesk and similar) or a fake banking app. | Wema Bank warning on fake apps (Jul 2026) https://allafrica.com/stories/202607160456.html |
| `sim_line_threat` | "Your line will be barred, dial this code" or "send NIN". This sets up SIM swap or call forwarding, after which USSD banking is drained. | NCC consumer bureau on SIM swap and USSD fraud (undated) https://thecable.ng/will-put-end-sim-swap-fraud-says-ncc ; ngCERT advisory https://csirt.ncc.gov.ng/index.php/resources/security-advisories/148-cyber-criminals-using-sim-swapping-attack-to-steal-millions-of-dollars-from-the-public |
| `card_swap_pos` | A "helper" at an ATM or POS stand watches the PIN and swaps the card. | General pattern. No 2025 official alert found, so it is kept as a minor type. |

## Payments between people

| scam_type | What happens | Sources |
|---|---|---|
| `fake_credit_alert` | A buyer shows a fabricated SMS or screenshot from "flash fund" apps, and the trader releases goods. Police cases include N150k of rice in Edo and clothes in Osun (2025–2026). | Dubawa https://dubawa.org/how-to-identify-fake-bank-alerts-generated-by-flash-fund-apps/ ; Gazette NGR (Osun, 2026) https://gazettengr.com/osun-police-nab-four-suspects-using-fake-alerts-to-buy-goods/ ; BusinessDay (Edo) https://businessday.ng/news/article/edo-police-arrest-suspected-fake-transfer-fraudster-in-benin/ ; NIBSS 2025 fraud figures via https://abokiforex.app/show-news/1723846-fake-bank-alerts-how-nigerians-detect-avoid-payment-scams-2026 |
| `wrong_transfer_refund` | "I sent money to you by mistake, please return it to this other account." The credit is fake, stolen or a loan-app disbursement. | Dubawa, four credit-alert schemes https://dubawa.org/four-bank-credit-alert-schemes-scammers-have-used-to-defraud-victims/ |
| `whatsapp_account_takeover` | "I sent a 6-digit code to you by mistake, forward it." The calls pose as a yoga or chess club. The hijacked account then begs the victim's contacts for money. NPF-NCCC broke up a syndicate (complaint 14 Apr 2025). NITDA advisory. | NPF-NCCC https://nccc.npf.gov.ng/news/press-release-police-uncovers- ; Dubawa https://dubawa.org/how-whatsapp-scammers-circumvent-security-features-to-swindle-nigerians/ ; VON/NITDA https://von.gov.ng/?p=323484 |
| `hijacked_or_new_number_money_request` | A "friend", relative or pastor messages from a hijacked or "new" number: "send 50k urgently, will refund, don't call". | Same sources as `whatsapp_account_takeover` |
| `emergency_relative_impersonation` | "Mummy, police arrested me" or "your son had an accident, send money now". | General pattern, covered by the impersonation alerts below |
| `payment_diversion` | "Our account details have changed": school, landlord or supplier fees are redirected to a personal account. | General BEC and fees pattern. No Nigerian 2025 alert found. Included as a hard case with no keywords. |
| `online_vendor_advance_payment` | Instagram or Facebook vendor demands full or part payment first, with no pay-on-delivery, into a personal account. | General pattern. Fake ads and pages are flagged in the Africa Check and Opay items. |

## Investment, grants and prizes

| scam_type | What happens | Sources |
|---|---|---|
| `ponzi_investment` | "Double your money" or a guaranteed % per week. CBEX collapsed on 14 Apr 2025 with 600k or more Nigerian investors. Ajo-style pyramids. | TechCabal (15 Apr 2025) https://techcabal.com/2025/04/15/cbex-ponzi-collapse-nigeria/ ; Prime Business https://www.primebusiness.africa/?p=221428 |
| `investment_withdrawal_fee` | The collapsed scheme "resumes" and demands a withdrawal or audit fee to unlock frozen balances (CBEX as ST Technologies, Jun 2025). SEC says to check registration at sec.gov.ng/cmos. | Nairametrics (11 Jun 2025) https://nairametrics.com/2025/06/11/sec-issues-fresh-warning-as-cbex-resumes-operations-in-nigeria-two-months-after-crash/ ; The Condia https://thecondia.com/sec-warning-cbex-ponzi-scheme-nigeria/ ; NAN https://nannews.ng/cbex-illegal-unlicensed-digital-assets-exchange-sec/ |
| `romance_investment` | Romance or "pig-butchering": a relationship, then a crypto or forex platform. The EFCC's Lagos raid (Dec 2024) arrested 792. Prosecutions and deportations ran through 2025–2026. | Al Jazeera (22 Aug 2025) https://www.aljazeera.com/news/2025/8/22/nigeria-deports-60-chinese-39-filipino-convicted-in-crypto-romance-scams ; AllAfrica (Jan 2026) https://allafrica.com/stories/202601150010.html |
| `crypto_giveaway`, `fixed_match_betting` | "Send 1, get 2 back" giveaways, and "100% sure" fixed-match tickets. | General guaranteed-returns pattern (SEC warnings above) |
| `fake_grant` | "FG / Tinubu / CBN N50,000 grant": apply through a link, enter BVN and account, "share to 5 groups". The CBN says it does not award contracts, loans or grants through unsolicited SMS or WhatsApp (12 May 2025). | Nairametrics (12 May 2025) https://nairametrics.com/2025/05/12/cbn-warns-nigerians-against-fake-contracts-grants-loans/ ; Africa Check https://africacheck.org/fact-checks/meta-programme-fact-checks/no-nigerian-government-not-giving-out-n50000-grants-ignore ; Dubawa https://dubawa.org/false-claims-circulate-about-federal-government-₦50000-grant-scheme-portal-for-youth/ ; CERRT.ng https://www.itedgenews.africa/cerrtng-to-nigerians-president-tinubu-n50-000-cash-grant-is-fake/ |
| `fake_lottery_promo` | "Your line won N2.5m in the MTN/Glo promo, pay a processing fee or send account details". | General pattern. Similar telco-branded giveaways appear in the Opay fake-giveaway report below. |
| `inheritance_advance_fee` | Classic 419: inheritance, a "box of dollars" or a diplomat, with fees for clearance. | Background pattern (EFCC prosecutions) |
| `recovery_scam` | "NDIC/EFCC victims' compensation: pay a fee to receive your refund." It targets people who were already scammed. | Pattern documented at https://scamnet.wa.gov.au/scamnet/Scam_types-Unexpected_money-Nigerian_scams-Nigerian_scam_refund.htm and https://scamwarners.com/forum/viewtopic.php?p=304275 . No 2025 NDIC alert found. |
| `fake_shutdown_rumour` | "Opay is shutting down 1 Sept, move your money now". Opay denied it in 2026, and the panic makes people easy targets for fraudsters. Earlier fake Opay N100k giveaways and loan pages. | Leadership https://leadership.ng/?p=688651 ; AbokiFX https://abokiforex.app/show-news/1728142-opay-breaks-silence-nigerians-identify-alleged-sources-viral-shut-rumour |

## Loans, jobs, travel, exams

| scam_type | What happens | Sources |
|---|---|---|
| `fake_loan_upfront_fee` | "Instant loan, no BVN, pay insurance/processing fee first". Fake bank loan pages. | Africa Check (GTBank fake loans, 2026) above ; https://nairacompare.ng/blogs/full-list-illegal-loan-apps-delisted-by-the-fg-updated-2026 |
| `illegal_loan_app_harassment` | Unregistered or illegal digital lenders threaten to defame the borrower to their contacts, or "credit" unsolicited loans. FCCPC DEON Regulations 2025 (21 Jul 2025). Registration deadline 5 Jan 2026, with more than 100 apps on the enforcement radar. | BusinessDay https://businessday.ng/technology/article/fccpc-approves-154-loan-apps-takes-action-against-unethical-practices/ ; Guardian https://guardian.ng/?p=2770103 ; ICIR https://www.icirnigeria.org/hardship-more-nigerians-opt-for-loan-sharks-amid-harassment-defamation/ |
| `fake_job_recruitment` | Fake recruitment by NNPC, EFCC, NIS, NCC or census staff, with a "documentation fee". EFCC denied 2026 recruitment (Jan 2026). NIS (Feb 2025) and NPC (Jan 2025) issued warnings. | AllAfrica (EFCC, 29 Jan 2026) https://allafrica.com/stories/202601290488.html ; Leadership https://leadership.ng/efcc-denies-viral-claims-of-secret-recruitment-warns-nigerians-against-scam/ ; NCC disclaimer https://ncc.gov.ng/node/2589 ; Nairametrics https://nairametrics.com/?p=508829 |
| `task_job_scam` | "Like videos and earn N30k daily". Small early payouts, then a "negative balance" or "combination task" that needs a deposit to withdraw. Losses up 485% YoY in 2025 (global). | Bitdefender https://www.bitdefender.com/en-us/blog/hotforsecurity/easy-tasks-real-money-the-hidden-danger-behind-task-scams ; CPA Practice Advisor (Jun 2026) https://www.cpapracticeadvisor.com/2026/06/08/scam-watch-job-hunters-lured-by-task-based-schemes/184759/ |
| `japa_visa_scam` | Guaranteed UK, Canada or other visas with job sponsorship for large fees. Fake CoS and employment letters. Police warning (Mar 2026). NIS on fake migration schemes. | Gazette NGR (8 Mar 2026) https://gazettengr.com/police-warn-nigerians-planning-to-japa-about-visa-scams/ ; Arise https://www.arise.tv/nis-raises-alarm-over-rising-fake-migration-schemes-human-trafficking-networks/ ; ICIR https://www.icirnigeria.org/japa-efcc-nabs-serial-visa-scammer-in-abuja-luxury-hotel/ ; Prime Business (IOM) https://www.primebusiness.africa/?p=110441 |
| `fake_flight_ticket` | Discount airline tickets sold through links or WhatsApp agents. The EFCC reported more than 700 victims who lost N652m. | Leadership https://leadership.ng/efcc-over-700-people-lost-n652m-to-airline-ticket-discount-scam/ |
| `fake_exam_body` | Fake JAMB UTME "rescheduling" for N15,700 paid to a personal Sterling account, using a spoofed 55019/8863 sender (2025). Fake admission and scholarship offers. | NAN (2025) https://nannews.ng/?p=279756 ; 21st Century Chronicle https://21stcenturychronicle.com/jamb-disowns-site-requesting-payment-from-candidates-who-missed-2025-utme/ ; Africa Check (UniAbuja) https://africacheck.org/fact-checks/meta-programme-fact-checks/nigerias-university-abuja-communicates-official-channels-be |
| `fake_scholarship` | Government or agency "scholarship" with an application fee. | NIMASA warning (older) https://shippingposition.com.ng/nimasa-raises-alarm-over-fake-online-scholarship-syndicate/ |

## Impersonating authorities and utilities

| scam_type | What happens | Sources |
|---|---|---|
| `law_enforcement_impersonation` | Fake EFCC or police officers: "your account is linked to fraud, pay bail or settlement" or street extortion. EFCC alarm (21 Apr 2026). EFCC also warns about fake help-desk alerts and fake social accounts. | Channels TV (21 Apr 2026) https://www.channelstv.com/2026/04/21/efcc-raises-alarm-over-surge-in-impersonation-scams-fake-officers/ ; AllAfrica https://allafrica.com/stories/202604220072.html ; Ripples https://www.ripplesnigeria.com/beware-of-scam-alert-coming-from-our-help-desk-efcc-warns/ |
| `parcel_customs_fee` | "Your parcel is held at customs, pay clearance through this link". No Nigerian 2025 alert was found, but the pattern is widespread. | SA Post Office https://www.gov.za/nr/news/media-statements/post-office-warns-parcel-scam-22-mar-2022 ; Monzo https://monzo.com/ie/help/custom-parcel-scams/help-customs-text |
| `utility_disconnection_threat` | "DisCo/DStv: you will be disconnected today, pay the field officer or update your card through this link". | General pattern (impersonation and urgency) |
| `fake_rental_listing` | Inspection fee or "commitment fee" before viewing a flat that does not exist. | General Lagos pattern |
| `charity_or_pastor_impersonation` | A pastor's or imam's "new number" asks for a seed or donation. | General pattern (new-number impersonation) |

## Gaps

* No 2025 NDIC-specific or Nigeria Customs-specific alert was found. Those types rely on the general pattern.
* No official task-scam alert from the NPF or NCCC was found. The task-scam sources are global.
* GTBank, Moniepoint and PalmPay 2025 advisories did not come up in searches. The Access, Zenith and Wema advisories and
  the CBN rule ("never asks for PIN, OTP, BVN") stand in for them.
