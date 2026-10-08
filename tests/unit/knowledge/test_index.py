from collections.abc import Sequence

import numpy as np
import pytest

from kudiguard.domain import Flag
from kudiguard.knowledge.embeddings import l2_normalize
from kudiguard.knowledge.index import EntryIndex, fuse
from kudiguard.knowledge.models import Contact, ContactKind, Entry, Query


def entry(
    id_: str, title: str, facts: str = "x", flags: tuple[Flag, ...] = (), kw: tuple[str, ...] = ()
) -> Entry:
    return Entry(id=id_, title=title, facts=(facts,), flags=flags, keywords=kw)


ENTRIES = [
    entry("otp", "Never share your OTP", "Banks never ask for your OTP or PIN.", (Flag.ASKS_CREDENTIALS,)),
    entry(
        "block", "Account blocked messages", "Scammers say your account is blocked.", (Flag.ACCOUNT_THREAT,)
    ),
    entry("both", "OTP plus blocked account", "Combined trick.", (Flag.URGENCY, Flag.ASKS_CREDENTIALS)),
    entry("saving", "How to start saving", "Save a fixed amount every payday.", kw=("savings", "save money")),
    entry(
        "loan", "Loan app interest", "Check the total cost of a loan app before borrowing.", kw=("loan apps",)
    ),
    entry("transfer", "Failed transfer", "Most failed transfers reverse automatically; contact your bank."),
]
CONTACTS = [
    Contact(label="Bank fraud desk", kind=ContactKind.PHONE, value="111", flags=(Flag.ASKS_CREDENTIALS,)),
    Contact(label="General help", kind=ContactKind.WEB, value="https://example.org"),
    Contact(
        label="Investment regulator",
        kind=ContactKind.WEB,
        value="https://sec.example",
        flags=(Flag.GUARANTEED_RETURNS,),
    ),
]


@pytest.fixture
def index() -> EntryIndex:
    return EntryIndex(ENTRIES, CONTACTS)


def test_flag_entries_ordered_by_strongest_flag_then_file_order(index: EntryIndex) -> None:
    result = index.search(Query(text="", flags=(Flag.URGENCY, Flag.ACCOUNT_THREAT, Flag.ASKS_CREDENTIALS)))
    assert [e.id for e in result.flag_entries] == ["otp", "both", "block"]


def test_no_flags_means_no_flag_entries(index: EntryIndex) -> None:
    assert index.search(Query(text="how do I save money")).flag_entries == ()


@pytest.mark.parametrize(
    ("text", "best"),
    [
        ("How can I start saving money every month?", "saving"),
        ("Is it wise to borrow from loan apps?", "loan"),
        ("My transfer failed but I was debited", "transfer"),
    ],
)
def test_text_search_ranks_relevant_entry_first(index: EntryIndex, text: str, best: str) -> None:
    assert index.search(Query(text=text)).hits[0].entry.id == best


def test_irrelevant_or_stopword_only_query_has_no_hits(index: EntryIndex) -> None:
    assert index.search(Query(text="what is the weather in Kano")).hits == ()
    assert index.search(Query(text="how do I do it?")).hits == ()
    assert index.search(Query(text="")).hits == ()


def test_hits_exclude_flag_entries_and_respect_limit(index: EntryIndex) -> None:
    result = index.search(
        Query(text="otp pin blocked account saving loan transfer", flags=(Flag.ASKS_CREDENTIALS,), limit=2)
    )
    assert {e.id for e in result.flag_entries}.isdisjoint(h.entry.id for h in result.hits)
    assert len(result.hits) <= 2


def test_weak_trailing_hits_are_dropped(index: EntryIndex) -> None:
    hits = index.search(Query(text="save money savings every payday transfer", limit=5)).hits
    assert hits[0].entry.id == "saving"
    assert len(hits) < 5


def test_contacts_relevant_first_then_general(index: EntryIndex) -> None:
    result = index.search(Query(text="", flags=(Flag.ASKS_CREDENTIALS,)))
    assert [c.value for c in result.contacts] == ["111", "https://example.org"]


def test_general_query_gets_only_general_contacts(index: EntryIndex) -> None:
    assert [c.value for c in index.search(Query(text="save")).contacts] == ["https://example.org"]


def test_empty_index() -> None:
    result = EntryIndex([], []).search(Query(text="anything", flags=(Flag.URGENCY,)))
    assert result.flag_entries == result.hits == result.contacts == ()


# --- hybrid (BM25 + dense) ----------------------------------------------------------------------------

CONCEPTS = {
    "transfer": {"transfer", "debited", "reach", "reversal", "reverse", "failed"},
    "saving": {"saving", "save", "aside", "payday"},
    "loan": {"loan", "borrow", "lender"},
}


class FakeEmbedder:
    """Concept-bag vectors: texts sharing a concept are similar, others orthogonal."""

    def _vec(self, text: str) -> np.ndarray:
        words = set(text.lower().replace(";", " ").replace(".", " ").split())
        v = np.array([len(words & c) for c in CONCEPTS.values()], dtype=np.float32)
        return l2_normalize(v)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, 0), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


@pytest.fixture
def hybrid() -> EntryIndex:
    return EntryIndex(ENTRIES, CONTACTS, embedder=FakeEmbedder())


def vq(text: str, **kw: object) -> Query:
    """A query carrying its embedding, as the KnowledgeBase provides it."""
    return Query(text=text, vector=FakeEmbedder().embed_query(text), **kw)  # type: ignore[arg-type]


def test_dense_finds_paraphrase_with_no_shared_words(index: EntryIndex, hybrid: EntryIndex) -> None:
    query = vq("my cash did not reach him")
    assert index.search(query).hits == ()  # no embeddings in this index
    assert hybrid.search(query).hits[0].entry.id == "transfer"


def test_dense_below_threshold_is_dropped(hybrid: EntryIndex) -> None:
    assert hybrid.search(vq(text="what is the weather in Kano")).hits == ()


def test_fusion_prefers_entries_found_by_both(hybrid: EntryIndex) -> None:
    # BM25 ranks loan first, transfer second; dense finds only transfer -> fusion puts transfer first.
    hits = hybrid.search(vq(text="loan apps borrow, and failed reversal", limit=3)).hits
    assert [h.entry.id for h in hits[:2]] == ["transfer", "loan"]


def test_dense_respects_exclusions_and_limit(hybrid: EntryIndex) -> None:
    result = hybrid.search(
        Query(text="save aside payday reversal debited", flags=(Flag.ASKS_CREDENTIALS,), limit=1)
    )
    assert len(result.hits) == 1
    assert result.hits[0].entry.id not in {e.id for e in result.flag_entries}


def test_hybrid_on_empty_index() -> None:
    assert EntryIndex([], [], embedder=FakeEmbedder()).search(vq("save")).hits == ()


def test_fuse_single_ranking_keeps_order() -> None:
    assert fuse([(2, 9.0), (0, 1.0)], []) == [2, 0]
    assert fuse([], []) == []


def test_fuse_rewards_agreement() -> None:
    assert fuse([(0, 5.0), (1, 4.0)], [(1, 0.9), (2, 0.8)]) == [1, 0, 2]


@pytest.mark.parametrize(
    ("lexical", "dense", "winner"),
    [
        # BM25 decisive (6.2 vs 3.4), dense marginal (0.708 vs 0.705): BM25's pick wins the RRF tie
        ([(0, 6.2), (1, 3.4)], [(1, 0.708), (0, 0.705)], 0),
        # dense decisive, BM25 marginal: dense's pick wins
        ([(0, 5.0), (1, 4.9)], [(1, 0.85), (0, 0.63)], 1),
    ],
)
def test_fuse_breaks_ties_by_confidence(
    lexical: list[tuple[int, float]], dense: list[tuple[int, float]], winner: int
) -> None:
    assert fuse(lexical, dense)[0] == winner


def test_index_never_embeds_queries_itself(hybrid: EntryIndex) -> None:
    # Without a vector, dense search is skipped (BM25 only) instead of blocking on an embedding call.
    assert hybrid.search(Query(text="my cash did not reach him")).hits == ()


def test_mismatched_vector_falls_back_to_bm25(hybrid: EntryIndex) -> None:
    wrong = Query(text="save money payday", vector=np.ones(7, np.float32))
    assert hybrid.search(wrong).hits[0].entry.id == "saving"


def test_hits_carry_relevance(index: EntryIndex, hybrid: EntryIndex) -> None:
    bm25_hit = index.search(Query(text="how do I save money")).hits[0]
    assert bm25_hit.relevance is not None and 0 < bm25_hit.relevance < 1
    dense_hit = hybrid.search(vq("my cash did not reach him")).hits[0]
    assert dense_hit.relevance == pytest.approx(1.0)


def test_scam_entries_down_weighted_in_free_text_search() -> None:
    entries = [
        entry("scam", "Upfront fee scam", "Pay a fee to get a loan.", (Flag.UPFRONT_FEE,)),
        entry("loan", "Loan basics", "How a loan fee works."),
    ]
    hits = EntryIndex(entries, []).search(Query(text="loan fee")).hits
    assert hits[0].entry.id == "loan"


def test_fuse_weights_scale_scores() -> None:
    assert fuse([(0, 5.0), (1, 4.5)]) == [0, 1]
    assert fuse([(0, 5.0), (1, 4.5)], weights={0: 0.5}) == [1, 0]


def test_len_counts_entries_not_contacts() -> None:
    assert len(EntryIndex(ENTRIES, CONTACTS)) == len(ENTRIES)
    assert len(EntryIndex([], CONTACTS)) == 0
