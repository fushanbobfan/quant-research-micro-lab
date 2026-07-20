"""Measure the empirical lower tail of equity-curve returns."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from numbers import Real
from pathlib import Path
from typing import Any

from .risk import load_equity_csv


def analyze_return_tail(
    equity: Sequence[float],
    *,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Return fixed-count lower-tail and downside metrics for an equity curve."""

    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, Real)
        or not math.isfinite(confidence)
        or not 0.0 < confidence < 1.0
    ):
        raise ValueError("confidence must be a finite number between 0 and 1")
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
    periods = [
        {
            "start_index": index - 1,
            "end_index": index,
            "return": values[index] / values[index - 1] - 1.0,
        }
        for index in range(1, len(values))
    ]
    ordered_periods = sorted(
        periods,
        key=lambda item: (item["return"], item["end_index"]),
    )
    tail_count = max(1, math.ceil((1.0 - float(confidence)) * len(periods)))
    tail_periods = ordered_periods[:tail_count]
    returns = [item["return"] for item in periods]
    loss_count = sum(value < 0.0 for value in returns)

    return {
        "observations": len(values),
        "return_observations": len(returns),
        "confidence": float(confidence),
        "tail_fraction": 1.0 - float(confidence),
        "tail_observation_count": tail_count,
        "tail_cutoff_return": tail_periods[-1]["return"],
        "mean_tail_return": sum(item["return"] for item in tail_periods) / tail_count,
        "worst_return": min(returns),
        "best_return": max(returns),
        "loss_period_count": loss_count,
        "loss_period_rate": loss_count / len(returns),
        "downside_deviation": math.sqrt(
            sum(min(value, 0.0) ** 2 for value in returns) / len(returns)
        ),
        "tail_periods": tail_periods,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--column",
        choices=("equity", "gross_equity"),
        default="equity",
        help="curve to analyze from a quant-backtest equity export",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args(argv)

    try:
        dates, equity = load_equity_csv(args.dataset, args.column)
        report = analyze_return_tail(equity, confidence=args.confidence)
    except (OSError, UnicodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    report = {
        **report,
        "column": args.column,
        "start_date": dates[0],
        "end_date": dates[-1],
        "tail_periods": [
            {
                **period,
                "start_date": dates[period["start_index"]],
                "end_date": dates[period["end_index"]],
            }
            for period in report["tail_periods"]
        ],
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
