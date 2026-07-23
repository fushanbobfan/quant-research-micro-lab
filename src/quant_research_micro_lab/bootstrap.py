"""Estimate path-metric uncertainty with a moving-block bootstrap."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from numbers import Real
from typing import Any

from .backtest import maximum_drawdown


def _annualized_volatility(returns: Sequence[float], periods_per_year: int) -> float:
    mean_return = sum(returns) / len(returns)
    variance = sum((value - mean_return) ** 2 for value in returns) / len(returns)
    return math.sqrt(variance * periods_per_year)


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


def _path_metrics(returns: Sequence[float], periods_per_year: int) -> dict[str, float]:
    equity = [1.0]
    for value in returns:
        equity.append(equity[-1] * (1.0 + value))
    return {
        "total_return": equity[-1] - 1.0,
        "annualized_volatility": _annualized_volatility(
            returns, periods_per_year
        ),
        "maximum_drawdown": maximum_drawdown(equity),
    }


def bootstrap_equity_performance(
    equity: Sequence[float],
    *,
    block_size: int = 5,
    samples: int = 2_000,
    confidence: float = 0.95,
    seed: int = 0,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """Return moving-block bootstrap intervals for three path metrics."""

    for name, value in (
        ("block_size", block_size),
        ("samples", samples),
        ("periods_per_year", periods_per_year),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0.0 < confidence < 1.0
    ):
        raise ValueError("confidence must be between 0 and 1")
    if isinstance(equity, (str, bytes)) or not isinstance(equity, Sequence):
        raise ValueError("equity must be a sequence")
    if len(equity) < 2:
        raise ValueError("equity must contain at least two observations")
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0
        for value in equity
    ):
        raise ValueError("equity values must be finite positive numbers")

    values = [float(value) for value in equity]
    returns = [
        values[index] / values[index - 1] - 1.0
        for index in range(1, len(values))
    ]
    if block_size > len(returns):
        raise ValueError("block_size must not exceed the return count")

    observed = {
        "total_return": values[-1] / values[0] - 1.0,
        "annualized_volatility": _annualized_volatility(
            returns, periods_per_year
        ),
        "maximum_drawdown": maximum_drawdown(values),
    }
    block_start_count = len(returns) - block_size + 1
    generator = random.Random(seed)
    bootstrap_metrics = []
    for _ in range(samples):
        sampled_returns: list[float] = []
        while len(sampled_returns) < len(returns):
            start = generator.randrange(block_start_count)
            sampled_returns.extend(returns[start : start + block_size])
        bootstrap_metrics.append(
            _path_metrics(sampled_returns[: len(returns)], periods_per_year)
        )

    alpha = (1.0 - float(confidence)) / 2.0
    metric_names = (
        "total_return",
        "annualized_volatility",
        "maximum_drawdown",
    )
    distributions = {
        metric: [sample[metric] for sample in bootstrap_metrics]
        for metric in metric_names
    }
    return {
        "observations": len(values),
        "return_observations": len(returns),
        "block_size": block_size,
        "block_start_count": block_start_count,
        "samples": samples,
        "confidence": float(confidence),
        "seed": seed,
        "periods_per_year": periods_per_year,
        "observed": observed,
        "bootstrap_mean": {
            metric: sum(distributions[metric]) / samples
            for metric in metric_names
        },
        "intervals": {
            metric: {
                "lower": _percentile(distributions[metric], alpha),
                "upper": _percentile(distributions[metric], 1.0 - alpha),
            }
            for metric in metric_names
        },
        "negative_total_return_rate": sum(
            value < 0.0 for value in distributions["total_return"]
        )
        / samples,
    }
