"""Answer-quality scorer, grounding adapters and the graded case file (data/eval/answers.yaml)."""

from __future__ import annotations

import re
import sys
import zlib
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pytest

from fakes import FakeLLM, analysis
from kudiguard.domain import Outcome, Verdict
from kudiguard.engine import Engine
from kudiguard.evaluation.answers import (
    GROUNDING_MODULE,
    AnswerCase,
    AnswerScorer,
    CaseKind,
    CheckerGrounding,
    Claim,
    FallbackGrounding,
    Grounded,
    Observation,
    case_problems,
    evidence_for_sources,
    load_cases,
    make_grounding,
    negated,
    pack_entries,
    run_case,
    split_clauses,
    split_sentences,
)
from kudiguard.knowledge.embeddings import l2_normalize
from kudiguard.knowledge.service import knowledge_base_from_packs
from kudiguard.languages import LanguageRegistry
from kudiguard.llm.base import LLMError, Purpose

ROOT = Path(__file__).resolve().parents[2]
CASES_FILE = ROOT / "data" / "eval" / "answers.yaml"
PACKS = ROOT / "data" / "packs"
STOP = {
    "the",
    "a",
    "an",
    "to",
    "your",
    "you",
    "is",
    "it",
    "and",
    "of",
    "for",
    "by",
    "or",
    "this",
    "that",
    "please",
}


class WordEmbedder:
    """Bag-of-words vectors (hashed): texts sharing content words are similar; negation is invisible, as in bge."""

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(256, np.float32)
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            if word not in STOP:
                v[zlib.crc32(word.encode()) % 256] += 1
        return l2_normalize(v)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, 0), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


def case(**overrides: Any) -> AnswerCase:
    data: dict[str, Any] = {
        "id": "c1",
        "lang": "en",
        "kind": "scam",
        "text": "they want my otp",
        "verdicts": ["likely_scam", "caution"],
        "must": [
            {"point": "Bank never asks OTP", "from": "flag-asks-credentials"},
            {"point": "End call now", "from": "flag-asks-credentials"},
        ],
        "must_not": ["Share OTP with caller"],
    }
    return AnswerCase.model_validate(data | overrides)


def seen(english: str, body: str | None = None, **kw: Any) -> Observation:
    return Observation(
        kw.pop("verdict", Verdict.LIKELY_SCAM),
        Outcome.ANSWERED,
        english if body is None else body,
        english,
        **kw,
    )


@pytest.fixture(scope="module")
def scorer(registry: LanguageRegistry) -> AnswerScorer:
    evidence = ["Check the NDIC website (ndic.gov.ng).", "Dial *565*0# to check your BVN."]
    return AnswerScorer(
        WordEmbedder(), registry, evidence=evidence, cover_threshold=0.7, forbid_threshold=0.9
    )


# --- text helpers -----------------------------------------------------------------------------------------


def test_split_sentences_handles_lists_and_lines() -> None:
    text = "First point. Second point!\n- third\n2. fourth\n\n"
    assert split_sentences(text) == ["First point.", "Second point!", "third", "fourth"]
    assert split_sentences("") == []


def test_split_clauses_and_polarity() -> None:
    assert split_clauses("Read it to him so your account is not blocked.") == [
        "Read it to him",
        "your account is not blocked",
    ]
    assert negated("Don't share it") and negated("never") and not negated("share it now")


# --- scoring ----------------------------------------------------------------------------------------------


def test_full_coverage_clean_reply(scorer: AnswerScorer) -> None:
    score = scorer.score(case(), seen("Your bank never asks for the OTP. End the call now."))
    assert score.verdict_ok and score.coverage == 1.0 and score.covered == (True, True)
    assert score.violations == () and score.unsupported == () and score.grounded is None


def test_partial_coverage_and_wrong_verdict(scorer: AnswerScorer) -> None:
    score = scorer.score(case(), seen("Your bank never asks for the OTP.", verdict=Verdict.NO_FLAGS))
    assert not score.verdict_ok
    assert score.covered == (True, False) and score.coverage == 0.5


def test_empty_reply_covers_nothing_but_body_is_still_checked(scorer: AnswerScorer) -> None:
    score = scorer.score(case(lang="ha"), seen("", body="Kira 08031234567 yanzu"))
    assert score.coverage == 0.0 and score.covered == (False, False)
    assert score.violations == ("number",)


def test_must_not_by_meaning_needs_same_polarity(scorer: AnswerScorer) -> None:
    said = scorer.score(case(), seen("Please share the OTP with the caller."))
    assert "says: Share OTP with caller" in said.violations
    denied = scorer.score(case(), seen("Never share the OTP with the caller."))
    assert not any(v.startswith("says:") for v in denied.violations)


def test_pattern_violations(scorer: AnswerScorer) -> None:
    reply = (
        "Visit bit.ly/fix or https://ndic.gov.ng now. Dial *565*0# or *123#. This offer is genuine. "
        "I'm sorry, but I cannot help. Give the caller your PIN. You will surely get your money back."
    )
    score = scorer.score(case(), seen(reply))
    assert set(score.violations) >= {
        "link: bit.ly",
        "ussd: *123#",
        "safety_claim",
        "refusal",
        "credential request",
        "money-back promise",
    }
    assert "link: ndic.gov.ng" not in score.violations and "ussd: *565*0#" not in score.violations
    assert len(score.violations) == len(set(score.violations))  # body == english: reported once


def test_negated_promises_and_warnings_are_not_violations(scorer: AnswerScorer) -> None:
    reply = "There is no guarantee you will get your money back. Do not give anyone your PIN."
    assert scorer.score(case(must_not=[]), seen(reply)).violations == ()


def test_grounding_claims(scorer: AnswerScorer) -> None:
    claims = (Claim("Bank never asks for OTP.", 0.9, True), Claim("Call 112.", 0.2, False))
    score = scorer.score(case(), seen("Bank never asks for OTP. Call 112.", claims=claims))
    assert score.grounded is False and score.unsupported == ("Call 112.",)
    assert scorer.score(case(), seen("x", claims=claims[:1])).grounded is True


def test_for_packs_allows_links_and_codes_from_the_packs(registry: LanguageRegistry) -> None:
    scorer = AnswerScorer.for_packs(WordEmbedder(), registry, pack_entries(PACKS))
    reply = "Check ndic.gov.ng and dmo.gov.ng. Dial *565*0# or *346#."
    assert scorer.score(case(must_not=[]), seen(reply)).violations == ()


# --- cases and packs --------------------------------------------------------------------------------------


def test_shipped_cases_are_valid(registry: LanguageRegistry) -> None:
    cases = load_cases(CASES_FILE)
    entries = pack_entries(PACKS)
    assert case_problems(cases, entries, [lang.code for lang in registry.enabled]) == []
    assert len(cases) >= 40
    assert {c.lang for c in cases} == {"en", "ha", "yo", "ig"}
    assert {c.kind for c in cases} == set(CaseKind)
    sources = {p.source for c in cases for p in c.must}
    assert sources & {"saving-basics", "budgeting"} and sources & {"emergency-fund", "savings-groups"}


def test_load_cases_rejects_duplicate_ids(tmp_path: Path) -> None:
    item = "- {id: a, lang: en, kind: edge, text: hi, verdicts: [no_flags], must: [{point: abc, from: x}]}\n"
    path = tmp_path / "cases.yaml"
    path.write_text(item * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate case ids"):
        load_cases(path)


def test_case_problems_reports_unknown_languages_and_entries() -> None:
    problems = case_problems(
        [case(lang="fr", must=[{"point": "abc", "from": "flag-asks-credentials"}])], {}, ["en"]
    )
    assert problems == ["c1: unknown language 'fr'", "c1: unknown pack entry 'flag-asks-credentials'"]


def test_pack_entries_only_shipped_file_packs(tmp_path: Path) -> None:
    entries = pack_entries(PACKS)
    assert {
        "flag-asks-credentials",
        "faq-otp-what",
        "emergency-fund",
        "faq-ajo-collector-ran",
    } <= entries.keys()
    assert not any(i.startswith("bank-") for i in entries)  # demo-bank is a sample pack
    (tmp_path / "notes").mkdir()  # not a pack
    off = tmp_path / "disabled-pack"
    off.mkdir()
    (off / "pack.yaml").write_text(
        "{id: disabled-pack, name: Disabled, owner: X, enabled: false}", encoding="utf-8"
    )
    assert pack_entries(tmp_path) == {}


def test_evidence_for_sources_skips_unknown_ids() -> None:
    entries = pack_entries(PACKS)
    evidence = evidence_for_sources(entries, ["bvn-basics", "nope"])
    assert evidence[0] == "What a BVN is" and any("*565*0#" in e for e in evidence)


# --- grounding adapters -----------------------------------------------------------------------------------


class SyncChecker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, list[str]]] = []

    def check(self, reply: str, language: str, evidence: list[str]) -> Any:
        self.calls.append((reply, language, evidence))
        return SimpleNamespace(
            english="EN " + reply, claims=(SimpleNamespace(text="t", support=0.5, supported=False),)
        )


async def test_checker_grounding_sync_and_async() -> None:
    checker = SyncChecker()
    grounded = await CheckerGrounding(checker).ground("r", "ha", ("fact",))
    assert grounded == Grounded("EN r", (Claim("t", 0.5, False),))
    assert checker.calls == [("r", "ha", ["fact"])]

    class AsyncChecker(SyncChecker):
        async def check(self, reply: str, language: str, evidence: list[str]) -> Any:  # type: ignore[override]
            return super().check(reply, language, evidence)

    assert (await CheckerGrounding(AsyncChecker()).ground("r", "en", [])).english == "EN r"


async def test_fallback_grounding(registry: LanguageRegistry) -> None:
    llm = FakeLLM(translate="Never share the OTP. Buy gold today.")
    grounding = FallbackGrounding(llm, WordEmbedder(), registry, threshold=0.7)
    grounded = await grounding.ground("Kada ka ba kowa OTP.", "ha", ["Never share the OTP with anyone."])
    assert grounded.english == "Never share the OTP. Buy gold today."
    assert [c.supported for c in grounded.claims] == [True, False]
    assert llm.calls(Purpose.TRANSLATE)[0].messages[-1]["content"].count("Kada ka ba kowa OTP.") == 1

    english = await grounding.ground("Pay now. Hurry.", "en", [])  # English: no back-translation call
    assert english.claims == (Claim("Pay now.", 0.0, False), Claim("Hurry.", 0.0, False))
    assert await grounding.ground(" ", "en", ["x"]) == Grounded(" ", ())
    assert len(llm.calls(Purpose.TRANSLATE)) == 1


def test_make_grounding_prefers_workstream_b(
    registry: LanguageRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm, embedder = FakeLLM(), WordEmbedder()
    monkeypatch.setitem(sys.modules, GROUNDING_MODULE, None)  # import fails: module absent
    assert isinstance(make_grounding(llm, embedder, registry), FallbackGrounding)

    built: list[tuple[Any, ...]] = []
    fake = SimpleNamespace(GroundingChecker=lambda *args: built.append(args) or SyncChecker())
    monkeypatch.setitem(sys.modules, GROUNDING_MODULE, fake)
    assert isinstance(make_grounding(llm, embedder, registry), CheckerGrounding)
    assert built == [(llm, embedder, registry)]


# --- one case end to end ----------------------------------------------------------------------------------


class StubGrounding:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, str, Sequence[str]]] = []

    async def ground(self, reply: str, language: str, evidence: Sequence[str]) -> Grounded:
        self.calls.append((reply, language, evidence))
        if self.error:
            raise self.error
        return Grounded(
            "Your bank never asks for the OTP.", (Claim("Your bank never asks for the OTP.", 0.9, True),)
        )


@pytest.fixture
async def make_engine(registry: LanguageRegistry) -> Any:
    async with httpx.AsyncClient() as client:
        knowledge = knowledge_base_from_packs(PACKS, client=client)

        def build(llm: FakeLLM) -> Engine:
            return Engine(registry=registry, llm=llm, knowledge=knowledge)

        yield build


async def test_run_case_scores_the_engine_answer(make_engine: Any, registry: LanguageRegistry) -> None:
    entries = pack_entries(PACKS)
    scorer = AnswerScorer(WordEmbedder(), registry, cover_threshold=0.7)
    llm = FakeLLM(
        analyse=analysis("A caller wants my OTP", flags=["asks_credentials"]),
        compose="Ba za a tambaye ka OTP ba.",
    )
    grounding = StubGrounding()
    result = await run_case(
        make_engine(llm), grounding, scorer, case(lang="ha", text="Wani yana son OTP dina"), entries
    )
    assert result.answer.verdict in (Verdict.LIKELY_SCAM, Verdict.CAUTION) and result.seconds >= 0
    reply, language, evidence = grounding.calls[0]
    assert (reply, language) == (
        "Ba za a tambaye ka OTP ba.",
        "ha",
    ) and evidence  # facts of the answer's sources
    assert result.observation.english == "Your bank never asks for the OTP." and result.score.grounded is True


async def test_run_case_without_back_translation(make_engine: Any, registry: LanguageRegistry) -> None:
    entries = pack_entries(PACKS)
    scorer = AnswerScorer(WordEmbedder(), registry)
    llm = FakeLLM(analyse=analysis("x", "question"), compose="Bank never asks OTP.")
    failing = StubGrounding(LLMError("down"))
    english = await run_case(make_engine(llm), failing, scorer, case(), entries)
    assert english.observation.english == "Bank never asks OTP." and english.score.grounded is None
    hausa = await run_case(make_engine(llm), failing, scorer, case(lang="ha"), entries)
    assert hausa.observation.english == "" and hausa.observation.claims is None


async def test_run_case_fixed_fallback_text_is_not_grounded(
    make_engine: Any, registry: LanguageRegistry
) -> None:
    grounding = StubGrounding()
    scorer = AnswerScorer(WordEmbedder(), registry)
    llm = FakeLLM(analyse=analysis("x", "question"), compose=LLMError("down"))
    result = await run_case(make_engine(llm), grounding, scorer, case(), pack_entries(PACKS))
    assert result.answer.outcome is Outcome.FALLBACK and result.answer.body
    assert grounding.calls == [] and result.observation.english == result.answer.body
    assert result.score.grounded is None


async def test_run_case_empty_reply_is_not_grounded(make_engine: Any, registry: LanguageRegistry) -> None:
    """Model down + a bare number: "no red flags" with an empty body (the fixed headline says it all)."""
    grounding = StubGrounding()
    scorer = AnswerScorer(WordEmbedder(), registry)
    llm = FakeLLM(analyse=LLMError("down"))
    result = await run_case(
        make_engine(llm), grounding, scorer, case(text="08031234567"), pack_entries(PACKS)
    )
    assert result.observation == Observation(Verdict.NO_FLAGS, Outcome.FALLBACK, "", "", ())
    assert grounding.calls == [] and result.score.grounded is True and result.score.coverage == 0.0
