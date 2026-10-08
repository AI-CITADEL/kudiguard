# Sustainability

## Cost per check (estimates; to be confirmed with pilot data)
| Item | Pilot (this PC) | Cloud |
|---|---|---|
| N-ATLAS on one GPU (~10–12 checks/min) | electricity only | ~US$0.6–1/h for a 24 GB GPU VM ⇒ ~US$0.001–0.002 per check at full load |
| Speech (CPU), search, storage | included | included in the same VM |
| WhatsApp replies | free (user-initiated service conversations) | free |
| Tunnel / HTTPS | free (Cloudflare) | free / a few dollars a month |

Fixed text (greetings, commands, buttons) costs no model time; the rules answer even when the model is down.

## Licence path
N-ATLAS is used under its licence (≤ 1,000 active end-users per rolling 30 days, enforced in code; attribution to the
Federal Ministry of Communications, Innovation and Digital Economy and Awarri shown to users). Beyond the cap: a
commercial licence from Awarri, or partner deployments under each partner's own licence.

## Revenue (post-competition)
* **API for banks and fintechs**: per-check pricing for "Is this real?" in their apps and support channels.
* **White-label WhatsApp** for banks and telcos: their brand, our engine, their verified contacts as a knowledge pack.
* **Partner knowledge packs**: institutions plug in their own FAQs and contacts (lock-in through integration).
* **Scam-trend reports** from the de-identified pattern library: early warnings by language (region later).
* **Free for citizens**: the public WhatsApp number stays free, funded by the above and by grants/CSR.

## Partners to approach
Banks' fraud and customer-service teams, telcos (NCC 622 flow), EFCC/CBN consumer protection, cooperatives and NGOs
for outreach in Hausa, Yoruba and Igbo communities.

## Operations
One container (Dockerfile) + one GPU host; languages, scam phrasings and contacts are configuration, so new languages
and partners need no code changes. Next scale step: shared database and several instances (docs/DEPLOY.md).
