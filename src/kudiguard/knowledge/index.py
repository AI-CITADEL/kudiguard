"""In-memory index over one source's entries: red-flag lookup + hybrid text search.

Text search = BM25 (exact words) fused with optional dense embeddings (meaning) by Reciprocal Rank Fusion.
Small by design (hundreds to a few thousand entries per pack), so dense search is a brute-force dot product.
Queries are English pivot text. The query embedding is computed once per request by the KnowledgeBase (off the
event loop) and passed in ``Query.vector``; the index never embeds queries itself, so it never blocks. Without a
usable vector it degrades to BM25 only.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from kudiguard.knowledge.embeddings import Embedder
from kudiguard.knowledge.models import Contact, Entry, Hit, Query, SourceResult
from kudiguard.safety.scam_rules import WEIGHTS
from kudiguard.text import tokens

log = logging.getLogger(__name__)

STOPWORDS = frozenset(
    """a an and are as at be been but by can could do does doing did for from get got has have how i if in into
    is it its me my no not of on or our please should so that the their them then there they this to too us was
    we were what when where which who why will with would you your yes ok okay abeg tell say said know want need
    like make let also just some any much many about thing things good best wise""".split()  # noqa: SIM905
)
K1 = 1.5
B = 0.75
MIN_SCORE = 0.5  # BM25 absolute floor: below this a "hit" is noise
RELATIVE_CUTOFF = 0.4  # drop BM25 hits scoring under 40% of the best hit
FIELD_BOOST = 2  # title and keywords count double
DENSE_MIN = 0.62  # cosine floor for bge-small-en-v1.5 (unrelated text sits around 0.45-0.55)
RRF_K = 60
CANDIDATES_PER_RESULT = 3  # each ranking contributes at most limit * 3 candidates to fusion
FLAG_HIT_WEIGHT = 0.5  # scam-warning entries rank below literacy topics in free-text results (measured)
BM25_SQUASH = 3.0  # relevance = score / (score + 3) when there are no embeddings

Ranked = list[tuple[int, float]]  # (entry position, score), best first


def _stem(token: str) -> str:
    return token[:-1] if len(token) > 3 and token.endswith("s") and not token.endswith("ss") else token


def _terms(text: str) -> list[str]:
    return [_stem(t) for t in tokens(text, STOPWORDS)]


def _boosted(entry: Entry) -> str:
    return " ".join((entry.title, *entry.keywords))


def _body(entry: Entry) -> str:
    return " ".join((*entry.facts, *entry.actions))


def document_text(entry: Entry) -> str:
    """The text embedded for an entry (also used to build indexes off the event loop)."""
    return f"{_boosted(entry)}. {_body(entry)}"


class EntryIndex:
    def __init__(
        self,
        entries: Sequence[Entry],
        contacts: Sequence[Contact],
        *,
        embedder: Embedder | None = None,
        dense_min: float = DENSE_MIN,
    ) -> None:
        self._entries = tuple(entries)
        self._contacts = tuple(contacts)
        self._docs = [Counter(_terms(_boosted(e)) * FIELD_BOOST + _terms(_body(e))) for e in self._entries]
        self._lengths = [sum(d.values()) for d in self._docs]
        self._avg_len = (sum(self._lengths) / len(self._lengths)) if self._docs else 0.0
        df: Counter[str] = Counter(term for doc in self._docs for term in doc)
        n = len(self._docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self._flagged = {i: FLAG_HIT_WEIGHT for i, e in enumerate(self._entries) if e.flags}

        self._dense_min = dense_min
        self._matrix = (
            embedder.embed_documents([document_text(e) for e in self._entries])
            if embedder is not None and self._entries
            else None
        )

    def __len__(self) -> int:
        return len(self._entries)

    def search(self, query: Query) -> SourceResult:
        wanted = set(query.flags)
        flagged = [
            (-max(WEIGHTS[f] for f in e.flags if f in wanted), i)
            for i, e in enumerate(self._entries)
            if wanted.intersection(e.flags)
        ]
        exclude = {i for _, i in flagged}
        return SourceResult(
            flag_entries=tuple(self._entries[i] for _, i in sorted(flagged)),
            hits=self._rank(query, exclude),
            contacts=tuple(c for c in self._contacts if wanted.intersection(c.flags))
            + tuple(c for c in self._contacts if not c.flags),
        )

    def _rank(self, query: Query, exclude: set[int]) -> tuple[Hit, ...]:
        if query.limit <= 0 or not query.text.strip():
            return ()
        candidates = query.limit * CANDIDATES_PER_RESULT
        sims = self._similarities(query)
        lexical = self._bm25(query.text, exclude)[:candidates]
        dense = self._dense(sims, exclude)[:candidates]
        order = fuse(lexical, dense, weights=self._flagged)[: query.limit]

        bm25 = dict(lexical)
        return tuple(
            Hit(self._entries[i], float(sims[i]) if sims is not None else bm25[i] / (bm25[i] + BM25_SQUASH))
            for i in order
        )

    def _similarities(self, query: Query) -> NDArray[np.float32] | None:
        if self._matrix is None or query.vector is None:
            return None
        if query.vector.shape != (self._matrix.shape[1],):
            log.warning(
                "query vector shape %s does not match index dim %d; using BM25 only",
                query.vector.shape,
                self._matrix.shape[1],
            )
            return None
        return self._matrix @ query.vector

    def _bm25(self, text: str, exclude: set[int]) -> Ranked:
        terms = set(_terms(text)) & self._idf.keys()
        if not terms:
            return []
        scored = sorted(
            ((i, self._score(i, terms)) for i in range(len(self._docs)) if i not in exclude),
            key=lambda pair: (-pair[1], pair[0]),
        )
        if not scored or scored[0][1] < MIN_SCORE:
            return []
        floor = max(MIN_SCORE, scored[0][1] * RELATIVE_CUTOFF)
        return [(i, score) for i, score in scored if score >= floor]

    def _score(self, i: int, terms: set[str]) -> float:
        doc, norm = self._docs[i], K1 * (1 - B + B * self._lengths[i] / self._avg_len)
        return sum(self._idf[t] * doc[t] * (K1 + 1) / (doc[t] + norm) for t in terms if t in doc)

    def _dense(self, sims: NDArray[np.float32] | None, exclude: set[int]) -> Ranked:
        if sims is None:
            return []
        order = np.argsort(-sims, kind="stable")
        return [
            (int(i), float(sims[i])) for i in order if int(i) not in exclude and sims[i] >= self._dense_min
        ]


def fuse(*rankings: Ranked, weights: Mapping[int, float] | None = None) -> list[int]:
    """Reciprocal Rank Fusion of ranked ``(doc, score)`` lists (each best-first).

    ``weights`` scale a doc's fused score (e.g. down-rank scam warnings in general search). Ties are broken by
    confidence: the sum of each doc's score relative to that method's best, so a decisive preference beats a
    marginal one.
    """
    weights = weights or {}
    fused: dict[int, float] = {}
    confidence: dict[int, float] = {}
    for ranking in rankings:
        if not ranking:
            continue
        top = ranking[0][1] or 1.0
        for rank, (i, score) in enumerate(ranking):
            fused[i] = fused.get(i, 0.0) + 1 / (RRF_K + rank + 1)
            confidence[i] = confidence.get(i, 0.0) + score / top
    return sorted(fused, key=lambda i: (-round(fused[i] * weights.get(i, 1.0), 12), -confidence[i], i))
