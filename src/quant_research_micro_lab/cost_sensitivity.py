"""Measure how tested transaction-cost assumptions change a crossover backtest."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .backtest import backtest_crossover
from .cli import load_price_csv


def _validate_costs(values: Sequence[float]) -> list[float]:
    if not values:
        raise ValueError("at least one transaction cost is required")
    validated = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value < 10_000
        ):
            raise ValueError(
                "transaction cost values must be finite and between 0 and 10,000"
            )
        converted = float(value)
        if converted in validated:
            raise ValueError("transaction cost values must be unique")
        validated.append(converted)
    if 0.0 not in validated:
        raise ValueError("transaction cost values must include 0 as the baseline")
    return sorted(validated)


def _validate_maximum(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


def _validate_finite(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def analyze_cost_sensitivity(
    prices: Sequence[float],
    *,
    short_window: int,
    long_window: int,
    transaction_costs_bps: Sequence[float],
    max_return_degradation: float | None = None,
    min_total_return_at_highest_cost: float | None = None,
) -> dict[str, Any]:
    """Run one lagged crossover strategy across an ascending cost grid."""

    costs = _validate_costs(transaction_costs_bps)
    thresholds = {
        "return_degradation": _validate_maximum(
            "max_return_degradation", max_return_degradation
        ),
        "total_return_at_highest_cost": _validate_finite(
            "min_total_return_at_highest_cost", min_total_return_at_highest_cost
        ),
    }

    scenarios = []
    for cost_bps in costs:
        result = backtest_crossover(
            prices,
            short_window=short_window,
            long_window=long_window,
            transaction_cost_bps=cost_bps,
        )
        scenarios.append(
            {
                "transaction_cost_bps": cost_bps,
                "total_return": result["total_return"],
                "gross_total_return": result["gross_total_return"],
                "cost_drag": result["cost_drag"],
                "total_turnover": result["total_turnover"],
                "annualized_volatility": result["annualized_volatility"],
                "maximum_drawdown": result["maximum_drawdown"],
            }
        )

    zero_return = scenarios[0]["total_return"]
    for scenario in scenarios:
        scenario["return_change_from_zero_cost"] = (
            scenario["total_return"] - zero_return
        )

    highest = scenarios[-1]
    return_degradation = zero_return - highest["total_return"]
    monotonic_nonincreasing = all(
        current["total_return"] <= previous["total_return"] + 1e-15
        for previous, current in zip(scenarios, scenarios[1:])
    )
    first_nonpositive = next(
        (
            scenario["transaction_cost_bps"]
            for scenario in scenarios
            if scenario["total_return"] <= 0
        ),
        None,
    )
    break_even_bracket = next(
        (
            {
                "lower_tested_cost_bps": previous["transaction_cost_bps"],
                "upper_tested_cost_bps": current["transaction_cost_bps"],
            }
            for previous, current in zip(scenarios, scenarios[1:])
            if previous["total_return"] > 0 >= current["total_return"]
        ),
        None,
    )

    failures: list[dict[str, Any]] = []
    maximum_degradation = thresholds["return_degradation"]
    if maximum_degradation is not None and return_degradation > maximum_degradation:
        failures.append(
            {
                "metric": "return_degradation",
                "actual": return_degradation,
                "maximum": maximum_degradation,
                "excess": return_degradation - maximum_degradation,
            }
        )
    minimum_return = thresholds["total_return_at_highest_cost"]
    if minimum_return is not None and highest["total_return"] < minimum_return:
        failures.append(
            {
                "metric": "total_return_at_highest_cost",
                "actual": highest["total_return"],
                "minimum": minimum_return,
                "shortfall": minimum_return - highest["total_return"],
            }
        )

    return {
        "passed": not failures,
        "metrics": {
            "scenario_count": len(scenarios),
            "zero_cost_total_return": zero_return,
            "highest_tested_cost_bps": highest["transaction_cost_bps"],
            "total_return_at_highest_cost": highest["total_return"],
            "return_degradation": return_degradation,
            "monotonic_nonincreasing": monotonic_nonincreasing,
            "first_nonpositive_tested_cost_bps": first_nonpositive,
            "positive_to_nonpositive_bracket": break_even_bracket,
        },
        "thresholds": thresholds,
        "failures": failures,
        "scenarios": scenarios,
        "settings": {
            "short_window": short_window,
            "long_window": long_window,
            "transaction_costs_bps": costs,
        },
    }


def _paths_alias(source: Path, output: Path) -> bool:
    if source.resolve() == output.resolve():
        return True
    try:
        return source.samefile(output)
    except (FileNotFoundError, OSError):
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--short-window", type=int, default=5)
    parser.add_argument("--long-window", type=int, default=20)
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        action="append",
        required=True,
        dest="transaction_costs_bps",
        help="tested one-way turnover cost; repeat and include zero",
    )
    parser.add_argument("--max-return-degradation", type=float)
    parser.add_argument("--min-total-return-at-highest-cost", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        if args.output is not None and _paths_alias(args.dataset, args.output):
            raise ValueError("output must not alias the source dataset")
        dates, prices = load_price_csv(args.dataset)
        result = analyze_cost_sensitivity(
            prices,
            short_window=args.short_window,
            long_window=args.long_window,
            transaction_costs_bps=args.transaction_costs_bps,
            max_return_degradation=args.max_return_degradation,
            min_total_return_at_highest_cost=(
                args.min_total_return_at_highest_cost
            ),
        )
        report = {
            "observations": len(dates),
            "start_date": dates[0],
            "end_date": dates[-1],
            **result,
        }
        rendered = json.dumps(report, indent=2) + "\n"
        if args.output is None:
            print(rendered, end="")
        else:
            args.output.write_text(rendered, encoding="utf-8")
    except (OSError, UnicodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
