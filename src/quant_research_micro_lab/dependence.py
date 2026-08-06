"""Measure serial dependence in a dated equity curve's simple returns."""

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


def analyze_return_dependence(
    returns: Sequence[float],
    *,
    max_lag: int = 5,
    max_abs_autocorrelation: float | None = None,
) -> dict[str, Any]:
    """Return deterministic autocorrelation and portmanteau diagnostics."""

    if len(returns) < 2:
        raise ValueError("returns must contain at least two observations")
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= -1.0
        for value in returns
    ):
        raise ValueError("returns must be finite numbers greater than -1")
    if (
        isinstance(max_lag, bool)
        or not isinstance(max_lag, int)
        or max_lag <= 0
        or max_lag >= len(returns)
    ):
        raise ValueError("max_lag must be positive and less than the return count")
    if max_abs_autocorrelation is not None and (
        isinstance(max_abs_autocorrelation, bool)
        or not isinstance(max_abs_autocorrelation, Real)
        or not math.isfinite(max_abs_autocorrelation)
        or not 0.0 <= max_abs_autocorrelation <= 1.0
    ):
        raise ValueError("max_abs_autocorrelation must be between 0 and 1")

    values = [float(value) for value in returns]
    count = len(values)
    if all(value == values[0] for value in values[1:]):
        raise ValueError("returns must have non-zero variance")
    mean_return = sum(values) / count
    centered = [value - mean_return for value in values]
    centered_sum_squares = sum(value * value for value in centered)
    if centered_sum_squares == 0.0:
        raise ValueError("returns must have non-zero variance")

    lag_reports = []
    for lag in range(1, max_lag + 1):
        numerator = sum(
            centered[index] * centered[index - lag]
            for index in range(lag, count)
        )
        autocorrelation = numerator / centered_sum_squares
        lag_reports.append(
            {
                "lag": lag,
                "paired_observations": count - lag,
                "autocorrelation": autocorrelation,
                "absolute_autocorrelation": abs(autocorrelation),
            }
        )

    maximum = min(
        lag_reports,
        key=lambda item: (-item["absolute_autocorrelation"], item["lag"]),
    )
    ljung_box_statistic = count * (count + 2.0) * sum(
        item["autocorrelation"] ** 2 / (count - item["lag"])
        for item in lag_reports
    )
    threshold = (
        float(max_abs_autocorrelation)
        if max_abs_autocorrelation is not None
        else None
    )
    failures = []
    if threshold is not None and maximum["absolute_autocorrelation"] > threshold:
        failures.append(
            {
                "metric": "maximum_absolute_autocorrelation",
                "lag": maximum["lag"],
                "actual": maximum["absolute_autocorrelation"],
                "maximum": threshold,
                "excess": maximum["absolute_autocorrelation"] - threshold,
            }
        )

    return {
        "passed": not failures,
        "return_count": count,
        "summary": {
            "mean_return": mean_return,
            "standard_deviation": math.sqrt(centered_sum_squares / count),
            "positive_return_rate": sum(value > 0.0 for value in values) / count,
            "negative_return_rate": sum(value < 0.0 for value in values) / count,
            "zero_return_count": sum(value == 0.0 for value in values),
            "maximum_absolute_autocorrelation": {
                "lag": maximum["lag"],
                "value": maximum["absolute_autocorrelation"],
                "signed_value": maximum["autocorrelation"],
            },
            "ljung_box_statistic": ljung_box_statistic,
            "ljung_box_lags": max_lag,
        },
        "autocorrelations": lag_reports,
        "thresholds": {"max_abs_autocorrelation": threshold},
        "failures": failures,
        "settings": {
            "max_lag": max_lag,
            "autocorrelation_denominator": "full_centered_sum_squares",
            "ljung_box_p_value_reported": False,
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
    parser.add_argument(
        "--column",
        choices=("equity", "gross_equity"),
        default="equity",
        help="curve to analyze from a quant-backtest equity export",
    )
    parser.add_argument("--max-lag", type=int, default=5)
    parser.add_argument("--max-abs-autocorrelation", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        if args.output is not None and _paths_alias(args.dataset, args.output):
            raise ValueError("output must not alias the source dataset")
        dates, equity = load_equity_csv(args.dataset, args.column)
        returns = [
            equity[index] / equity[index - 1] - 1.0
            for index in range(1, len(equity))
        ]
        report = analyze_return_dependence(
            returns,
            max_lag=args.max_lag,
            max_abs_autocorrelation=args.max_abs_autocorrelation,
        )
        report = {
            **report,
            "column": args.column,
            "equity_observations": len(equity),
            "start_date": dates[0],
            "end_date": dates[-1],
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
