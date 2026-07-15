"""Inspect drawdown episodes in an equity curve."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real
from typing import Any


def _episode(
    values: Sequence[float],
    peak_index: int,
    trough_index: int,
    recovery_index: int | None,
) -> dict[str, float | int | bool | None]:
    endpoint = recovery_index if recovery_index is not None else len(values)
    return {
        "peak_index": peak_index,
        "trough_index": trough_index,
        "recovery_index": recovery_index,
        "peak_equity": values[peak_index],
        "trough_equity": values[trough_index],
        "depth": values[trough_index] / values[peak_index] - 1.0,
        "underwater_observations": endpoint - peak_index - 1,
        "recovered": recovery_index is not None,
    }


def analyze_drawdowns(equity: Sequence[float]) -> dict[str, Any]:
    """Return every peak-to-recovery episode and key drawdown summaries."""

    if not equity:
        raise ValueError("equity must contain at least one observation")
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0
        for value in equity
    ):
        raise ValueError("equity values must be finite positive numbers")

    values = [float(value) for value in equity]
    peak_index = 0
    trough_index = 0
    in_drawdown = False
    episodes = []

    for index in range(1, len(values)):
        value = values[index]
        if not in_drawdown:
            if value >= values[peak_index]:
                peak_index = index
            else:
                in_drawdown = True
                trough_index = index
            continue

        if value < values[trough_index]:
            trough_index = index
        if value >= values[peak_index]:
            episodes.append(_episode(values, peak_index, trough_index, index))
            peak_index = index
            in_drawdown = False

    if in_drawdown:
        episodes.append(_episode(values, peak_index, trough_index, None))

    maximum = min(episodes, key=lambda item: item["depth"], default=None)
    longest = max(
        episodes,
        key=lambda item: (item["underwater_observations"], -item["peak_index"]),
        default=None,
    )
    running_peak = max(values)
    return {
        "observations": len(values),
        "current_drawdown": values[-1] / running_peak - 1.0,
        "episode_count": len(episodes),
        "maximum_drawdown": maximum,
        "longest_underwater": longest,
        "episodes": episodes,
    }
