"""Compare an equity curve with an aligned benchmark series."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from numbers import Real
from pathlib import Path
from typing import Any

from .cli import load_price_csv
from .risk import load_equity_csv


def _validate_values(name: str, values: Sequence[float]) -> list[float]:
    validated = []
    for index, value in enumerate(values):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(
                f"{name} value {index} must be a finite positive number"
            )
        validated.append(float(value))
    return validated


def _returns(values: Sequence[float]) -> list[float]:
    return [
        values[index] / values[index - 1] - 1.0
        for index in range(1, len(values))
    ]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _variance(values: Sequence[float]) -> float:
    mean = _mean(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def compare_to_benchmark(
    strategy_equity: Sequence[float],
    benchmark_values: Sequence[float],
    *,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """Return relative performance diagnostics for two aligned value series."""

    if (
        isinstance(periods_per_year, bool)
        or not isinstance(periods_per_year, int)
        or periods_per_year <= 0
    ):
        raise ValueError("periods_per_year must be a positive integer")
    strategy = _validate_values("strategy", strategy_equity)
    benchmark = _validate_values("benchmark", benchmark_values)
    if len(strategy) != len(benchmark):
        raise ValueError("strategy and benchmark must have the same number of values")
    if len(strategy) < 2:
        raise ValueError("strategy and benchmark must contain at least two values")

    strategy_returns = _returns(strategy)
    benchmark_returns = _returns(benchmark)
    active_returns = [
        strategy_return - benchmark_return
        for strategy_return, benchmark_return in zip(
            strategy_returns, benchmark_returns
        )
    ]
    strategy_mean = _mean(strategy_returns)
    benchmark_mean = _mean(benchmark_returns)
    active_mean = _mean(active_returns)
    strategy_variance = _variance(strategy_returns)
    benchmark_variance = _variance(benchmark_returns)
    active_variance = _variance(active_returns)
    covariance = sum(
        (strategy_return - strategy_mean)
        * (benchmark_return - benchmark_mean)
        for strategy_return, benchmark_return in zip(
            strategy_returns, benchmark_returns
        )
    ) / len(strategy_returns)

    annualization = math.sqrt(periods_per_year)
    strategy_volatility = math.sqrt(strategy_variance) * annualization
    benchmark_volatility = math.sqrt(benchmark_variance) * annualization
    tracking_error = math.sqrt(active_variance) * annualization
    information_ratio = (
        active_mean * periods_per_year / tracking_error
        if tracking_error > 0
        else None
    )
    beta = covariance / benchmark_variance if benchmark_variance > 0 else None
    correlation = (
        covariance / math.sqrt(strategy_variance * benchmark_variance)
        if strategy_variance > 0 and benchmark_variance > 0
        else None
    )

    strategy_growth = strategy[-1] / strategy[0]
    benchmark_growth = benchmark[-1] / benchmark[0]
    return {
        "observations": len(strategy),
        "return_observations": len(strategy_returns),
        "periods_per_year": periods_per_year,
        "strategy_total_return": strategy_growth - 1.0,
        "benchmark_total_return": benchmark_growth - 1.0,
        "relative_total_return": strategy_growth / benchmark_growth - 1.0,
        "strategy_annualized_volatility": strategy_volatility,
        "benchmark_annualized_volatility": benchmark_volatility,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "beta": beta,
        "correlation": correlation,
        "active_return_hit_rate": sum(
            value > 0 for value in active_returns
        )
        / len(active_returns),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("strategy", type=Path)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument(
        "--strategy-column",
        choices=("equity", "gross_equity"),
        default="equity",
    )
    parser.add_argument("--periods-per-year", type=int, default=252)
    args = parser.parse_args(argv)

    try:
        strategy_dates, strategy_equity = load_equity_csv(
            args.strategy, args.strategy_column
        )
        benchmark_dates, benchmark_values = load_price_csv(args.benchmark)
        if strategy_dates != benchmark_dates:
            raise ValueError("strategy and benchmark dates must match exactly")
        result = compare_to_benchmark(
            strategy_equity,
            benchmark_values,
            periods_per_year=args.periods_per_year,
        )
    except (OSError, UnicodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "strategy_column": args.strategy_column,
                "start_date": strategy_dates[0],
                "end_date": strategy_dates[-1],
                **result,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
