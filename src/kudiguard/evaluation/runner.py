"""One graded case end to end: the production engine answers, the reply is grounded and scored."""

from __future__ import annotations

import time
from dataclasses import dataclass

from kudiguard.domain import Outcome
from kudiguard.engine import Answer, Engine
from kudiguard.evaluation.answers import AnswerCase, AnswerScorer, Observation, Score
from kudiguard.evaluation.grounding import Grounding, GroundingError
from kudiguard.knowledge.models import Query
from kudiguard.knowledge.service import KnowledgeBase
from kudiguard.languages import PIVOT
from kudiguard.pipeline import REPLY_RESERVE_S, Limits
from kudiguard.prompts import MAX_ENTRIES

TURN_DEADLINE_S = Limits().turn_deadline_s  # production default (Settings.turn_deadline_s)


def turn_deadline(now: float, turn_s: float = TURN_DEADLINE_S) -> float:
    """The deadline the pipeline gives ``Engine.check`` (``time.monotonic()`` seconds)."""
    return now + max(turn_s - REPLY_RESERVE_S, turn_s / 2)


@dataclass(frozen=True, slots=True)
class CaseResult:
    answer: Answer
    observation: Observation
    score: Score
    seconds: float  # engine time only (excludes back-translation and scoring)
    grounding_ms: int | None = None
    # The rebuilt search returned other entries than the answer used (evidence may not be what the model saw).
    evidence_mismatch: bool | None = None  # None: not rebuilt (fixed text)


async def run_case(
    engine: Engine,
    knowledge: KnowledgeBase,
    grounding: Grounding,
    scorer: AnswerScorer,
    case: AnswerCase,
    *,
    turn_s: float = TURN_DEADLINE_S,
) -> CaseResult:
    """Only model-composed replies are grounded; the evidence is rebuilt the way the engine searched for it."""
    started = time.perf_counter()
    answer = await engine.check(case.text, case.lang, deadline=turn_deadline(time.monotonic(), turn_s))
    seconds = time.perf_counter() - started
    english: str | None = ""
    unsupported: tuple[str, ...] | None = None
    evidence: tuple[str, ...] = ()
    grounding_ms: int | None = None
    mismatch: bool | None = None
    if answer.outcome is Outcome.ANSWERED and answer.body.strip():
        found = await knowledge.search(Query(text=answer.english, flags=answer.flags, language=case.lang))
        mismatch = tuple(item.entry.id for item in found.entries[:MAX_ENTRIES]) != answer.sources
        evidence = tuple(grounding.evidence(found, verdict=answer.verdict, flags=answer.flags))
        try:
            report = await grounding.ground(answer.body, case.lang, evidence, message=answer.english)
        except GroundingError:
            english = answer.body if case.lang == PIVOT else None  # unscored: no back-translation
        else:
            english, unsupported, grounding_ms = report.english, report.unsupported, report.ms
    observation = Observation(answer.verdict, answer.outcome, answer.body, english, evidence, unsupported)
    score = scorer.score(case, observation)
    return CaseResult(answer, observation, score, seconds, grounding_ms, mismatch)
