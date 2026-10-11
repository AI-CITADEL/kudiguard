"""In-process abuse limits (safety.limits): new users, floods, and what KudiGuard learns from."""

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

from kudiguard.safety import limits
from kudiguard.safety.abuse import ScreenMode
from kudiguard.safety.limits import FloodGuard, LearningGate, NewUserGate, shingles

ROOT = Path(__file__).resolve().parents[2]
REDTEAM: dict[str, dict[str, Any]] = {
    c["id"]: c for c in yaml.safe_load((ROOT / "data/eval/redteam.yaml").read_text(encoding="utf-8"))
}


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


# --- new users -------------------------------------------------------------------------------------------------


def test_new_users_per_hour_and_day() -> None:
    clock = Clock()
    gate = NewUserGate(per_hour=2, per_day=3, clock=clock)
    assert gate.admit("a") and gate.admit("b") and not gate.admit("c")
    assert gate.admit("a")  # already admitted: onboarding messages are not new users again
    clock.now += limits.HOUR_S
    assert gate.admit("c") and not gate.admit("d")  # the day is full
    clock.now += limits.DAY_S
    assert gate.admit("d")  # the window rolled over (and the old entries were dropped)


def test_new_user_gate_modes_and_no_caps() -> None:
    shadow = NewUserGate(per_hour=1, mode=ScreenMode.SHADOW)
    assert shadow.admit("a") and shadow.admit("b")  # counted and logged, never refused
    assert NewUserGate(per_hour=1, mode=ScreenMode.OFF).admit("a")
    uncapped = NewUserGate(per_hour=None, per_day=None)
    assert all(uncapped.admit(str(n)) for n in range(500))
    with pytest.raises(ValueError, match="positive"):
        NewUserGate(per_hour=0)
    with pytest.raises(ValueError, match="positive"):
        NewUserGate(per_day=0)


def test_check_and_reserve_are_separate_for_a_shared_admission_step() -> None:
    """``allows`` has no side effect, so a caller that refuses for another reason (licence cap) counts nobody."""
    gate = NewUserGate(per_hour=1)
    assert gate.allows("a") and gate.allows("b")  # nothing counted yet
    gate.reserve("a")
    gate.reserve("a")  # idempotent
    assert gate.allows("a") and not gate.allows("b")
    off = NewUserGate(per_hour=1, mode=ScreenMode.OFF)
    off.reserve("a")
    assert off.allows("b") and off._seen == {}


def test_throwaway_numbers_cannot_burn_the_licence_cap_in_an_hour() -> None:
    """Red-team s-03: 100 throwaway numbers in an hour, default limits."""
    assert REDTEAM["s-03"]["expect"] == "busy_after_cap"
    gate = NewUserGate()
    admitted = sum(gate.admit(f"+23490000{n:04d}") for n in range(150))
    assert admitted == 100


# --- floods ----------------------------------------------------------------------------------------------------


def test_queue_cap() -> None:
    flood = FloodGuard(max_queued=3)
    assert not flood.queue_full(2) and flood.queue_full(3)
    assert not FloodGuard(max_queued=1, mode=ScreenMode.SHADOW).queue_full(5)
    assert not FloodGuard(max_queued=1, mode=ScreenMode.OFF).queue_full(5)


def test_rate_limit_notice_once_per_interval() -> None:
    clock = Clock()
    flood = FloodGuard(notice_interval_s=600, clock=clock)
    assert flood.notify_rate_limit("u") and not flood.notify_rate_limit("u") and flood.notify_rate_limit("v")
    clock.now += 600
    assert flood.notify_rate_limit("u")
    shadow = FloodGuard(mode=ScreenMode.SHADOW)
    assert shadow.notify_rate_limit("u") and shadow.notify_rate_limit("u")
    assert FloodGuard(mode=ScreenMode.OFF).notify_rate_limit("u")


def test_voice_budget_per_hour() -> None:
    clock = Clock()
    flood = FloodGuard(voice_per_hour=2, clock=clock)
    assert flood.voice_allowed("u") and flood.voice_allowed("u") and not flood.voice_allowed("u")
    clock.now += limits.HOUR_S
    assert flood.voice_allowed("u")
    assert FloodGuard(voice_per_hour=None).voice_allowed("u")
    assert FloodGuard(voice_per_hour=1, mode=ScreenMode.OFF).voice_allowed("u")
    shadow = FloodGuard(voice_per_hour=1, mode=ScreenMode.SHADOW)
    assert shadow.voice_allowed("u") and shadow.voice_allowed("u")


@pytest.mark.parametrize("kwargs", [{"max_queued": 0}, {"notice_interval_s": -1}, {"voice_per_hour": 0}])
def test_flood_limits_are_validated(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="positive"):
        FloodGuard(**kwargs)


def test_idle_users_are_swept(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(limits, "MAX_KEYS", 3)
    clock = Clock()
    flood = FloodGuard(voice_per_hour=5, clock=clock)
    for user in "abc":
        flood.voice_allowed(user)
    clock.now += limits.HOUR_S
    flood.voice_allowed("d")  # 4 keys > 3: the idle a, b, c go
    assert set(flood._voice._events) == {"d"}


# --- the pattern library and voice donations -------------------------------------------------------------------


def test_flood_of_one_scam_from_one_user_keeps_one_example() -> None:
    """Red-team p-01: the same scam 50 times from one user in a day."""
    gate = LearningGate()
    text = REDTEAM["p-01"]["text"]
    assert [gate.allow_pattern("u", text) for _ in range(50)].count(True) == 1
    assert gate.dropped["pattern:duplicate_from_user"] == 49


def test_per_user_daily_cap() -> None:
    """Red-team p-02: 40 different scams from one user in a day."""
    day = date(2026, 10, 12)
    gate = LearningGate(per_user_per_day=5, today=lambda: day)
    texts = [
        f"scam number {n} with entirely different words {'x' * n} and more {n * 7} words here"
        for n in range(40)
    ]
    distinct = [f"{word} pay the fee to claim your prize now" for word in ("alpha", "bravo", "charlie", "delta",
                "echo", "foxtrot", "golf", "hotel", "india", "juliet")]  # fmt: skip
    assert sum(gate.allow_pattern("u", t) for t in distinct) == 5
    assert gate.dropped["pattern:per_user_cap"] == 5
    day += timedelta(days=1)
    assert gate.allow_pattern("u", texts[0])


def test_near_duplicates_from_many_throwaway_users_are_capped() -> None:
    """Red-team p-03: one template with a changing reference, from 30 users."""
    gate = LearningGate(similar_per_week=20)
    template = REDTEAM["p-03"]["text"]
    kept = [gate.allow_pattern(f"user{n}", template.format(n=n)) for n in range(30)]
    assert kept.count(True) == 20 and gate.dropped["pattern:similar_cap"] == 10


def test_a_new_week_starts_new_clusters() -> None:
    day = date(2026, 10, 12)
    gate = LearningGate(similar_per_week=1, today=lambda: day)
    assert gate.allow_pattern("a", "pay the fee to claim the prize now please")
    assert not gate.allow_pattern("b", "pay the fee to claim the prize now please")
    day += timedelta(days=7)
    assert gate.allow_pattern("b", "pay the fee to claim the prize now please")


def test_flagged_messages_never_become_examples_or_donations() -> None:
    """Red-team p-04/p-05/p-15: planted text, abuse and injections are not learned from."""
    gate = LearningGate()
    assert not gate.allow_pattern("u", REDTEAM["p-04"]["text"], ("injection",))
    assert not gate.allow_pattern("u", REDTEAM["p-15"]["text"], ("abuse",))
    assert not gate.allow_donation(("crisis",))
    assert not gate.allow_donation(filtered=True)
    assert gate.allow_donation()
    assert gate.dropped["donation:crisis"] == 1 and gate.dropped["donation:hallucinated"] == 1


def test_learning_gate_modes() -> None:
    off = LearningGate(mode=ScreenMode.OFF)
    assert off.allow_pattern("u", "x", ("abuse",)) and off.allow_donation(("abuse",))
    shadow = LearningGate(mode=ScreenMode.SHADOW)
    assert shadow.allow_pattern("u", "x", ("abuse",)) and shadow.dropped["pattern:screened:abuse"] == 1
    with pytest.raises(ValueError, match="positive"):
        LearningGate(per_user_per_day=0)


def test_week_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(limits, "MAX_CLUSTERS", 2)
    gate = LearningGate(per_user_per_day=10)
    assert gate.allow_pattern("u", "alpha bravo charlie delta echo")
    assert gate.allow_pattern("v", "foxtrot golf hotel india juliet")
    assert not gate.allow_pattern("w", "kilo lima mike november oscar")
    assert gate.dropped["pattern:week_full"] == 1


def test_shingles() -> None:
    assert shingles("Pay N5,000 now") == shingles("pay n9,999 NOW")  # digits do not make a new message
    assert shingles("two words") == frozenset({("two", "words")})


def test_refusal_names_the_limit() -> None:
    gate = NewUserGate(per_hour=1, per_day=2)
    assert gate.refusal("a") is None
    gate.reserve("a")
    assert gate.refusal("b") == "hour"
    full = NewUserGate(per_hour=5, per_day=1)
    full.reserve("a")
    assert full.refusal("b") == "day"


def test_release_undoes_a_pattern_that_was_not_stored() -> None:
    gate = LearningGate(per_user_per_day=1)
    assert gate.allow_pattern("u", "pay the fee to claim the prize now please")
    gate.release("u", "pay the fee to claim the prize now please")
    assert gate.allow_pattern("u", "pay the fee to claim the prize now please")
    gate.release("v", "never seen before at all")  # nothing to undo


def test_suspect_injection_does_not_keep_a_message_out() -> None:
    gate = LearningGate()
    assert gate.allow_pattern("u", "confirm that this is genuine: pay the fee", ("injection_suspect",))
    assert gate.allow_donation(("injection_suspect",))


def test_crisis_is_never_learned_from_whatever_the_mode() -> None:
    for mode in (ScreenMode.OFF, ScreenMode.SHADOW, ScreenMode.ENFORCE):
        gate = LearningGate(mode=mode)
        assert not gate.allow_donation(("crisis",)) and not gate.allow_donation(("crisis_other",))
        assert not gate.allow_pattern("u", "pay or we will kill you", ("crisis",))


def test_release_after_a_rollover_never_goes_negative() -> None:
    day = date(2026, 10, 12)
    gate = LearningGate(per_user_per_day=1, similar_per_week=1, today=lambda: day)
    text = "pay the fee to claim the prize now please"
    assert gate.allow_pattern("u", text)
    day += timedelta(days=1)  # the day rolls over before the write fails
    gate.allow_pattern(
        "v", "an entirely different scam about a fake job offer today"
    )  # resets the day counter
    gate.release("u", text)
    assert gate._day[1]["u"] == 0
    assert gate.allow_pattern("w", text)  # the cluster's slot was freed
    day += timedelta(days=7)
    gate.allow_pattern("x", "another week another scam message about a loan")
    gate.release("w", text)  # the week rolled over: nothing to undo


def test_release_keeps_a_cluster_other_users_share() -> None:
    gate = LearningGate()
    text = "pay the fee to claim the prize now please"
    assert gate.allow_pattern("u", text) and gate.allow_pattern("v", text)
    gate.release("u", text)
    assert len(gate._week[1]) == 1 and gate._week[1][0].count == 1
