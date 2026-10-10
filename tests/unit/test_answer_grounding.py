"""Answer-eval grounding (workstream B adapter and stand-in) and the end-to-end case runner."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from answer_helpers import WordEmbedder, case
from fakes import FakeLLM, analysis
from kudiguard.domain import Flag, Outcome, Verdict
from kudiguard.engine import Engine
from kudiguard.evaluation.answers import AnswerScorer, names_in, numbers_in
from kudiguard.evaluation.grounding import (
    GROUNDING_MODULE,
    VERDICT_LINE,
    CheckerGrounding,
    Claim,
    FallbackGrounding,
    GroundingError,
    GroundingReport,
    evidence_for,
    make_grounding,
    split_claims,
    text_specifics,
)
from kudiguard.evaluation.runner import TURN_DEADLINE_S, run_case, turn_deadline
from kudiguard.knowledge.models import Entry, KnowledgeResult, PackKind, SourcedEntry
from kudiguard.knowledge.service import KnowledgeBase, knowledge_base_from_packs
from kudiguard.languages import LanguageRegistry
from kudiguard.llm.base import LLMError, Purpose
from kudiguard.pipeline import REPLY_RESERVE_S

PACKS = Path(__file__).resolve().parents[2] / "data" / "packs"


def knowledge_result() -> KnowledgeResult:
    core = Entry(id="a", title="Title A", facts=("Fact one.", "Fact two."), actions=("Act one.",))
    partner = Entry(id="b", title="Title B", facts=("Partner fact.",), actions=("Act one.", "Act two."))
    return KnowledgeResult(
        entries=(
            SourcedEntry(core, "kudiguard-core", "Core"),
            SourcedEntry(partner, "bank", "Demo Bank", kind=PackKind.PARTNER),
        )
    )


# --- evidence and claims ----------------------------------------------------------------------------------


def test_evidence_is_what_the_model_saw() -> None:
    evidence = evidence_for(
        knowledge_result(), verdict=Verdict.LIKELY_SCAM, flags=(Flag.URGENCY,), message="  They want my OTP. "
    )
    assert evidence == [
        "Fact one.",
        "Fact two.",
        "Partner fact. (source: Demo Bank)",  # facts as the prompt shows them (no titles)
        "Act one.",
        "Act two.",
        VERDICT_LINE[Verdict.LIKELY_SCAM],
        "The message pressures you to act fast.",
        "They want my OTP.",
    ]
    assert evidence_for(KnowledgeResult(), verdict=Verdict.NOT_A_CHECK, message=" ") == []


def test_split_claims_and_report() -> None:
    assert split_claims("Stay calm. What to do:\n1. Call your bank now.\n2. Do not pay them.") == [
        "Call your bank now.",
        "Do not pay them.",
    ]
    report = GroundingReport("x", (Claim("a", 0.9, True), Claim("b", 0.1, False)))
    assert not report.grounded and report.unsupported == ("b",)
    assert GroundingReport("x", ()).grounded


# --- stand-in grounding -----------------------------------------------------------------------------------


async def test_fallback_grounding_back_translates_and_scores(registry: LanguageRegistry) -> None:
    llm = FakeLLM(translate="Never share the OTP code. Buy gold from the EFCC today.")
    grounding = FallbackGrounding(llm, WordEmbedder(), registry, threshold=0.7)
    evidence = grounding.evidence(knowledge_result(), verdict=Verdict.CAUTION, flags=(), message="")
    assert evidence[-1] == VERDICT_LINE[Verdict.CAUTION]
    report = await grounding.ground("Kada ka ba kowa OTP.", "ha", ["Never share the OTP code with anyone."])
    assert report.english == "Never share the OTP code. Buy gold from the EFCC today."
    assert [c.supported for c in report.claims] == [True, False]
    assert report.claims[1].unmatched == ("efcc",)
    assert "Kada ka ba kowa OTP." in llm.calls(Purpose.TRANSLATE)[0].messages[-1]["content"]


async def test_fallback_grounding_english_and_edge_cases(registry: LanguageRegistry) -> None:
    llm = FakeLLM(translate=["", LLMError("down")])
    grounding = FallbackGrounding(llm, WordEmbedder(), registry)
    report = await grounding.ground("Pay the fee now. Hurry up please.", "en", [])
    assert report.claims == (Claim("Pay the fee now.", 0.0, False), Claim("Hurry up please.", 0.0, False))
    assert (await grounding.ground("Ok.", "en", ["x"])).claims == ()
    assert llm.calls(Purpose.TRANSLATE) == []
    with pytest.raises(GroundingError, match="empty"):
        await grounding.ground("x", "ha", [])
    with pytest.raises(GroundingError, match="failed"):
        await grounding.ground("x", "ha", [])


# --- workstream B adapter ---------------------------------------------------------------------------------


class BError(RuntimeError):
    pass


class FakeChecker:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, str, list[str]]] = []

    async def check(self, reply: str, language: str, evidence: list[str]) -> Any:
        self.calls.append((reply, language, evidence))
        if self.error:
            raise self.error
        claim = SimpleNamespace(text="t", support=0.5, supported=False, unmatched=("efcc",))
        return SimpleNamespace(english="EN " + reply, claims=(claim,), ms=12)


def fake_b(checker: FakeChecker, built: list[tuple[Any, ...]] | None = None) -> SimpleNamespace:
    def make(*args: Any) -> FakeChecker:
        (built if built is not None else []).append(args)
        return checker

    return SimpleNamespace(
        GroundingChecker=make,
        GroundingError=BError,
        evidence_for=lambda knowledge, **kw: ["b-evidence", kw["verdict"], kw["message"]],
        names_in=lambda text: {"b-names"},
        numbers_in=lambda text: {"b-numbers"},
    )


async def test_checker_grounding_adapts_workstream_b() -> None:
    checker = FakeChecker()
    adapter = CheckerGrounding(fake_b(checker), checker)  # type: ignore[arg-type]
    report = await adapter.ground("r", "ha", ("fact",))
    assert report == GroundingReport("EN r", (Claim("t", 0.5, False, ("efcc",)),), 12)
    assert checker.calls == [("r", "ha", ["fact"])]
    evidence = adapter.evidence(KnowledgeResult(), verdict=Verdict.CAUTION, flags=(), message="m")
    assert evidence == ["b-evidence", Verdict.CAUTION, "m"]
    failing = FakeChecker(BError("no translation"))
    with pytest.raises(GroundingError, match="no translation"):
        await CheckerGrounding(fake_b(failing), failing).ground("r", "ha", ())  # type: ignore[arg-type]


def test_make_grounding_and_specifics_prefer_workstream_b(
    registry: LanguageRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm, embedder = FakeLLM(), WordEmbedder()
    monkeypatch.setitem(sys.modules, GROUNDING_MODULE, None)  # import fails: B not merged
    assert isinstance(make_grounding(llm, embedder, registry), FallbackGrounding)
    assert text_specifics() == (names_in, numbers_in)

    built: list[tuple[Any, ...]] = []
    monkeypatch.setitem(sys.modules, GROUNDING_MODULE, fake_b(FakeChecker(), built))
    assert isinstance(make_grounding(llm, embedder, registry), CheckerGrounding)
    assert built == [(llm, embedder, registry)]
    names, numbers = text_specifics()
    assert names("x") == {"b-names"} and numbers("x") == {"b-numbers"}


# --- one case end to end ----------------------------------------------------------------------------------


class StubGrounding:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, str, Sequence[str]]] = []
        self.knowledge: list[KnowledgeResult] = []

    def evidence(
        self, knowledge: KnowledgeResult, *, verdict: Verdict, flags: Sequence[Flag], message: str
    ) -> list[str]:
        self.knowledge.append(knowledge)
        return evidence_for(knowledge, verdict=verdict, flags=flags, message=message)

    async def ground(self, reply: str, language: str, evidence: Sequence[str]) -> GroundingReport:
        self.calls.append((reply, language, evidence))
        if self.error:
            raise self.error
        claim = Claim("Your bank never asks for the OTP.", 0.9, True)
        return GroundingReport("Your bank never asks for the OTP.", (claim,), 7)


@pytest.fixture
async def knowledge() -> Any:
    async with httpx.AsyncClient() as client:
        yield knowledge_base_from_packs(PACKS, client=client)


def engine(registry: LanguageRegistry, knowledge: KnowledgeBase, llm: FakeLLM) -> Engine:
    return Engine(registry=registry, llm=llm, knowledge=knowledge)


def test_turn_deadline_mirrors_the_pipeline() -> None:
    assert turn_deadline(100.0) == 100.0 + TURN_DEADLINE_S - REPLY_RESERVE_S
    assert turn_deadline(0.0, turn_s=4.0) == 2.0  # never less than half the turn


async def test_run_case_grounds_on_what_the_model_saw(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    llm = FakeLLM(
        analyse=analysis("A caller wants my OTP", flags=["asks_credentials"]), compose="Kar ka ba su OTP."
    )
    grounding = StubGrounding()
    scorer = AnswerScorer(WordEmbedder(), registry, cover_threshold=0.6)
    result = await run_case(engine(registry, knowledge, llm), knowledge, grounding, scorer, case(lang="ha"))
    reply, language, evidence = grounding.calls[0]
    assert (reply, language) == ("Kar ka ba su OTP.", "ha")
    assert (
        evidence[-1] == "A caller wants my OTP"
        and "The message asks for a code, PIN, password or BVN." in evidence
    )
    assert [e.entry.id for e in grounding.knowledge[0].entries][:1] == ["flag-asks-credentials"]
    assert result.observation.english == "Your bank never asks for the OTP." and result.observation.evidence
    assert result.score.grounded is True and result.grounding_ms == 7 and result.seconds >= 0
    assert result.score.covered[0] is True


async def test_run_case_without_back_translation(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    llm = FakeLLM(analyse=analysis("x", "question"), compose="Bank never asks OTP.")
    scorer = AnswerScorer(WordEmbedder(), registry)
    failing = StubGrounding(GroundingError("no translation"))
    english = await run_case(engine(registry, knowledge, llm), knowledge, failing, scorer, case())
    assert english.observation.english == "Bank never asks OTP." and english.score.grounded is None
    assert english.score.coverage is not None
    hausa = await run_case(engine(registry, knowledge, llm), knowledge, failing, scorer, case(lang="ha"))
    assert hausa.observation.english is None and hausa.score.coverage is None and hausa.score.grounded is None


async def test_run_case_fixed_text_is_not_grounded(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    scorer = AnswerScorer(WordEmbedder(), registry)
    grounding = StubGrounding()
    fallback = FakeLLM(analyse=analysis("x", "question"), compose=LLMError("down"))
    result = await run_case(engine(registry, knowledge, fallback), knowledge, grounding, scorer, case())
    assert result.answer.outcome is Outcome.FALLBACK and result.answer.body
    assert result.score.grounded is None and result.score.coverage == 0.0
    # Model down + a bare number: "no red flags" with an empty body.
    silent = FakeLLM(analyse=LLMError("down"))
    empty = await run_case(
        engine(registry, knowledge, silent), knowledge, grounding, scorer, case(text="08031234567")
    )
    assert empty.answer.body == "" and empty.observation.verdict is Verdict.NO_FLAGS
    assert empty.score.grounded is None and empty.score.coverage == 0.0
    assert grounding.calls == []
