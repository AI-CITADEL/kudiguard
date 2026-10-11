# Sustainability

## Cost per check (estimates; to be confirmed with pilot data)
| Item | Pilot (this PC) | Cloud |
|---|---|---|
| N-ATLAS on one GPU (~10–12 checks/min: ~5 s per check, one at a time, on the RTX 4070 laptop; English text p50 is 3.4 s, Yoruba/Igbo and voice take longer) | electricity only | assumed ~US$0.6–1/h for a 24 GB GPU VM (not a quote) ⇒ ~US$0.001–0.002 per check at full load |
| Speech (CPU), search, storage | included | included in the same VM |
| WhatsApp replies | free (user-initiated service conversations, while KudiGuard never sends templates) | free |
| Tunnel / HTTPS | free (Cloudflare) | free / a few dollars a month |

Fixed text (greetings, commands, buttons) costs no model time; the rules answer even when the model is down.

## Licence path
N-ATLAS is used under its licence (≤ 1,000 active end-users per rolling 30 days across WhatsApp, the website and the
partner API, enforced in code; attribution to the Federal Ministry of Communications, Innovation and Digital Economy
and Awarri shown to users). The partner API is **off in the pilot**. A partner that wants it needs either its own
N-ATLaS licence cover (its own deployment) or to name each end user with a pseudonymous id, so that every one of them
counts toward our cap. Beyond the cap: a commercial licence from Awarri, or partner deployments under each partner's
own licence.

## Revenue (post-competition)
* **API for banks and fintechs** (off in the pilot; needs licence cover beyond the cap): per-check pricing for "Is
  this real?" in their apps and support channels.
* **White-label WhatsApp** for banks and telcos: their brand, our engine, their verified contacts as a knowledge pack.
* **Partner knowledge packs**: institutions plug in their own FAQs and contacts (lock-in through integration).
* **Scam-trend reports** from the de-identified pattern library: early warnings by language (region later).
* **Free for citizens**: the public WhatsApp number stays free, funded by the above and by grants/CSR.

## Partners to approach
Banks' fraud and customer-service teams, telcos (NCC 622 flow), EFCC/CBN consumer protection, cooperatives and NGOs
for outreach in Hausa, Yoruba and Igbo communities.

## Operations
One container (Dockerfile) + one GPU host; languages, scam phrasings and contacts are configuration, so new languages
and partners need no code changes (a new *type* of red flag does). Next scale step: shared database and several instances (docs/DEPLOY.md).
