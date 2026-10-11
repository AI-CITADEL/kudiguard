# Knowledge packs — bring your own knowledge

KudiGuard answers from **knowledge packs**. Our core pack carries scam-safety guidance; a bank, fintech, cooperative
or NGO plugs in its own FAQ, safety messages and contact channels, with no code changes. Answers are composed in the
user's language (Hausa, Yoruba, Igbo, English…) from your content, so you write it once, in English. Quality differs
by language: on our dev set the answers cover about 0.64 of the key points in English, 0.46 in Hausa, 0.58 in Yoruba
and 0.33 in Igbo (docs/CLAIMS.md C6), and no Hausa/Yoruba/Igbo reply has been native-reviewed yet.

**Shipped packs:** `kudiguard-core` (scam red flags, basic money literacy, verified report contacts) and
`money-skills` (planning irregular income, emergency money, savings goals, ajo/esusu/adashe/isusu groups,
cooperatives, deposit insurance, debt, credit history, investing, crypto/forex warnings, insurance, micro pension,
small-business records and pricing, family support). `money-skills` is a partner-kind pack below core priority with no
red-flag tags, so it never takes a red-flag slot from core safety guidance; it was added as a pack; the only code change was the optional `cite` manifest field.
Partner packs that tag entries with red flags share the red-flag slots with core (round-robin).

## Option A — files (no engineering needed)

```
data/packs/<your-pack-id>/
├── pack.yaml       # required: who you are, priority
├── faq.csv         # question,answer[,id][,keywords]   ← an export of your existing FAQ works
├── topics.yaml     # optional: curated entries, can be tied to scam red flags
├── contacts.yaml   # optional: your fraud desk, WhatsApp, USSD block code, website
└── examples.yaml   # optional: entry id -> a model answer (or `example:` inline in topics.yaml)
```

A model answer is shown to the model for the top entry only, as something to adapt, never copy (a reply
that copies it is written again without it; docs/ARCHITECTURE.md "Advice quality").

See `data/packs/demo-bank/` (fictional sample) for a complete example.

**pack.yaml**
```yaml
id: acme-bank            # must equal the folder name
name: Acme Bank
owner: Acme Bank Plc
kind: partner            # 'core' is reserved for KudiGuard's safety pack
priority: 50             # higher = your answers and contacts come first
content_language: en
cite: true               # your facts reach the model as "(source: Acme Bank)"; only KudiGuard's own packs may set false
source: {type: files}
```

**faq.csv** — UTF-8 (Excel's "CSV UTF-8" is fine). `keywords` are optional, separated by `;`.
```csv
id,question,answer,keywords
block-card,How do I block my card?,Open the app and tap Cards then Freeze card.,block card;lost card
```

**topics.yaml / contacts.yaml** — see the sample pack. Red flags you can tag:
`asks_credentials, remote_access, guaranteed_returns, wrong_transfer, account_threat, upfront_fee, prize_lottery,
fake_payment, money_request, suspicious_link, impersonation, urgency, secrecy`.
Tagging a contact with flags (e.g. your fraud desk with `asks_credentials`) makes it the first contact shown in
matching scam replies.

## Option B — your API (contract v1)

```yaml
source:
  type: http
  base_url: https://kb.acme.example/kudiguard
  mode: sync              # sync: we pull /export every refresh_s | live: we call /search per question (off by default)
  timeout_s: 1.5
  refresh_s: 3600
  auth: {env: ACME_KB_TOKEN}   # sent as "Authorization: Bearer <token>"
```

| Endpoint | Request | Response |
|---|---|---|
| `GET /export` (sync) | — | `{"entries": [...], "contacts": [...]}` |
| `POST /search` (live) | `{"query": "<English text>", "flags": [...], "language": "ha", "limit": 3}` | same shape, best first |

Entries and contacts use the same fields as the files. A runnable reference server is in
`src/kudiguard/knowledge/reference_server.py` (`python scripts/mock_partner_api.py`).

**What is sent and checked:** HTTPS + per-partner token. Sync mode sends nothing about users. Live mode sends the
**English translation of the user's message** (most numbers and emails masked; names are not removed, and some
phone formats may pass), the red flags and a language code (never the phone number or audio); it is therefore off
unless the operator sets `KUDIGUARD_ALLOW_LIVE_PACKS=true`, which also discloses it on /privacy. Responses are
size-capped (1 MB default) and schema-validated.
**Resilience:** in sync mode the last good copy keeps serving if your API is down; every call has a timeout; three
consecutive failures open a circuit breaker for 60 s; live results are cached for `cache_ttl_s`.

## How answers are assembled

1. The user's message is transcribed (voice) and translated to English by N-ATLAS; deterministic rules decide the
   scam verdict.
2. All packs are searched **in parallel** within a 1.5 s budget (a slow or failing pack is skipped, never fatal):
   red-flag entries (core safety first) + hybrid text search (BM25 keywords fused with semantic embeddings) +
   contacts (flag-relevant first).
3. N-ATLAS restates **only those facts** in the user's language. Partner content can add facts and contacts but can
   never change the scam verdict or remove core safety guidance.

## Quality (measured, `tests/integration/test_retrieval_quality.py`)

**Held-out set** (`data/eval/retrieval_blind.yaml`): 24 English questions written by an independent author who never
saw the pack content. Production path (scam rules + hybrid search, production packs): **top-1 62%, top-3 96%**
(core pack alone: 67% / 96%; BM25 only: 67% / 75%). n=24, so one question moves the rate by 4 points, and real
queries arrive as N-ATLAS translations, which this set does not test.

**Dev sets** (`data/eval/retrieval.yaml`, 54 questions on the core pack, Pidgin-influenced phrasing included): the
pack content was tuned on them, so these are **not** held-out numbers:

| Setup (dev, tuned on) | Top-1 | Top-3 | p50 latency |
|---|---|---|---|
| BM25 only | 81% | 93% | ~0.1 ms |
| Hybrid (BM25 + bge-small-en-v1.5, CPU) | 89% | 94% | ~7-10 ms |
| Production path: scam rules + hybrid knowledge base | 89% | 96% | ~7-10 ms |

The query is embedded once per request, off the event loop, and cached; sources are searched in parallel.

## Operational rules

* **Scope**: each channel / API key can choose which partner packs apply (`Query.packs`); the core pack always applies.
* **Samples**: packs marked `sample: true` (e.g. the fictional Demo Bank) are refused unless explicitly allowed.
* **Merging**: each pack keeps its own ranking; relevance decides which pack's next answer comes first, and weak
  partner matches are dropped so partner content never crowds out a better core answer.
* **Contacts**: "your own bank" is pinned first, then contacts relevant to the detected red flags, then general ones.
* **Validation**: contact values are checked by kind (https links only, phone digits, USSD codes, email format);
  long FAQ answers are split into sentence-sized facts; files over 5 MB are rejected.
* **Partner APIs**: https required (plain http only on localhost); stale synced copies are served instantly while
  re-syncing in the background; hanging partners time out and trip the circuit breaker.
