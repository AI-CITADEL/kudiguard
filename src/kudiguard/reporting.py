"""Validation statistics from the privacy-safe evidence export (``Store.export_interactions``). Pure functions.

Web "users" are browser sessions (a private window is a new session), so report them separately from WhatsApp users,
whose ids are phone-number hashes. These are the numbers for docs/VALIDATION.md: real usage by unique (pseudonymous) users, language and verdict mix,
helpfulness from the 👍/👎 buttons, and latency percentiles per message kind. Rows from the team's own testing or the
B2B API can be excluded by channel or by user hash.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

USER_CHANNELS = frozenset(
    {"whatsapp", "web"}
)  # end-user channels (the API's callers are businesses, not people)


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile (``q`` in 0..100); None for no data."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


@dataclass(frozen=True, slots=True)
class Report:
    interactions: int
    unique_users: int
    users_by_channel: Mapping[str, int]
    by_language: Mapping[str, int]
    by_verdict: Mapping[str, int]
    by_outcome: Mapping[str, int]
    by_kind: Mapping[str, int]
    feedback_given: int
    helpful: int
    latency_ms: Mapping[str, Mapping[str, float | None]]  # kind -> {p50, p95, n}
    per_day: Mapping[str, int] = field(default_factory=dict)

    @property
    def helpful_rate(self) -> float | None:
        return self.helpful / self.feedback_given if self.feedback_given else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "interactions": self.interactions,
            "unique_users": self.unique_users,
            "users_by_channel": dict(self.users_by_channel),
            "by_language": dict(self.by_language),
            "by_verdict": dict(self.by_verdict),
            "by_outcome": dict(self.by_outcome),
            "by_kind": dict(self.by_kind),
            "feedback_given": self.feedback_given,
            "helpful": self.helpful,
            "helpful_rate": self.helpful_rate,
            "latency_ms": {k: dict(v) for k, v in self.latency_ms.items()},
            "per_day": dict(self.per_day),
        }


def build_report(
    rows: Iterable[Mapping[str, Any]],
    *,
    channels: frozenset[str] = USER_CHANNELS,
    exclude_users: frozenset[str] = frozenset(),
) -> Report:
    """Aggregate export rows. Only ``channels`` count; ``exclude_users`` drops team testers (by user hash)."""
    kept = [r for r in rows if r.get("channel") in channels and r.get("user") not in exclude_users]
    users_by_channel: dict[str, set[str]] = {}
    latencies: dict[str, list[float]] = {}
    feedback = [r["helpful"] for r in kept if r.get("helpful") is not None]
    for row in kept:
        users_by_channel.setdefault(str(row["channel"]), set()).add(str(row["user"]))
        if isinstance(row.get("total_ms"), int | float):
            latencies.setdefault(str(row.get("kind")), []).append(float(row["total_ms"]))
    return Report(
        interactions=len(kept),
        unique_users=len({r["user"] for r in kept}),
        users_by_channel={k: len(v) for k, v in sorted(users_by_channel.items())},
        by_language=dict(Counter(str(r.get("language")) for r in kept).most_common()),
        by_verdict=dict(Counter(str(r["verdict"]) for r in kept if r.get("verdict")).most_common()),
        by_outcome=dict(Counter(str(r.get("outcome")) for r in kept).most_common()),
        by_kind=dict(Counter(str(r.get("kind")) for r in kept).most_common()),
        feedback_given=len(feedback),
        helpful=sum(1 for v in feedback if v),
        latency_ms={
            kind: {"n": len(v), "p50": percentile(v, 50), "p95": percentile(v, 95)}
            for kind, v in sorted(latencies.items())
        },
        per_day=dict(sorted(Counter(str(r.get("created_at", ""))[:10] for r in kept).items())),
    )


def format_report(report: Report) -> str:
    """Human-readable summary (CLI)."""
    rate = f"{report.helpful_rate:.0%}" if report.helpful_rate is not None else "n/a"
    lines = [
        f"Interactions: {report.interactions}   unique users: {report.unique_users} "
        f"(whatsapp users: {report.users_by_channel.get('whatsapp', 0)}, "
        f"web sessions: {report.users_by_channel.get('web', 0)})",
        f"Languages: {dict(report.by_language)}",
        f"Verdicts: {dict(report.by_verdict)}",
        f"Outcomes: {dict(report.by_outcome)}",
        f"Helpful: {report.helpful}/{report.feedback_given} ({rate})",
    ]
    for kind, stats in report.latency_ms.items():
        p50, p95 = stats["p50"], stats["p95"]
        lines.append(f"Latency {kind}: n={stats['n']:.0f} p50={p50 or 0:.0f} ms p95={p95 or 0:.0f} ms")
    lines.append(f"Per day: {dict(report.per_day)}")
    return "\n".join(lines)
