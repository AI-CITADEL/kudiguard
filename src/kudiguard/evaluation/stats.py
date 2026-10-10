"""Small statistics shared by the evaluation scripts."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from statistics import fmean


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (index ``int(n * q)``, capped); 0.0 for no values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * q), len(ordered) - 1)]


def mean_or_none(values: Iterable[float]) -> float | None:
    """Mean, or None when there is nothing to average (a metric that was not measured)."""
    items = list(values)
    return fmean(items) if items else None
