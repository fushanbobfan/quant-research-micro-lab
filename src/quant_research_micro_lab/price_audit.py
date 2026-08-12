"""Audit dated closing-price series for research data-quality warnings."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from datetime import date
from numbers import Real
from pathlib import Path
from typing import Any

from .cli import load_price_csv


def _validate_optional_count(
    name: str, value: int | None, *, minimum: int = 0
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def _validate_optional_nonnegative(
    name: str, value: float | None
) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value < 0.0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


def _unchanged_runs(
    dates: Sequence[str], prices: Sequence[float]
) -> list[dict[str, Any]]:
    runs = []
    start_index: int | None = None
    for index in range(1, len(prices)):
        if prices[index] == prices[index - 1]:
            if start_index is None:
                start_index = index - 1
            continue
        if start_index is not None:
            runs.append(
                {
                    "start_date": dates[start_index],
                    "end_date": dates[index - 1],
                    "unchanged_transitions": index - 1 - start_index,
                    "observations": index - start_index,
                    "close": prices[start_index],
                }
            )
            start_index = None
    if start_index is not None:
        runs.append(
            {
                "start_date": dates[start_index],
                "end_date": dates[-1],
                "unchanged_transitions": len(prices) - 1 - start_index,
                "observations": len(prices) - start_index,
                "close": prices[start_index],
            }
        )
    return runs


def audit_price_series(
    dates: Sequence[str],
    prices: Sequence[float],
    *,
    max_calendar_gap_days: int | None = None,
    max_unchanged_run: int | None = None,
    max_abs_return: float | None = None,
    max_details: int = 10,
) -> dict[str, Any]:
    """Return gap, stale-price, and extreme-return diagnostics for a price series."""

    if len(dates) != len(prices):
        raise ValueError("dates and prices must have the same length")
    if len(dates) < 2:
        raise ValueError("at least two dated prices are required")
    if isinstance(max_details, bool) or not isinstance(max_details, int) or max_details < 0:
        raise ValueError("max_details must be a non-negative integer")

    maximum_gap = _validate_optional_count(
        "max_calendar_gap_days", max_calendar_gap_days, minimum=1
    )
    maximum_unchanged = _validate_optional_count(
        "max_unchanged_run", max_unchanged_run
    )
    maximum_return = _validate_optional_nonnegative(
        "max_abs_return", max_abs_return
    )

    parsed_dates = []
    previous_date: date | None = None
    for index, value in enumerate(dates):
        if not isinstance(value, str):
            raise ValueError(f"date {index} must be an ISO date string")
        try:
            parsed = date.fromisoformat(value)
        except ValueError as error:
            raise ValueError(f"date {index} must be an ISO date string") from error
        if previous_date is not None and parsed <= previous_date:
            raise ValueError("dates must be strictly increasing")
        parsed_dates.append(parsed)
        previous_date = parsed

    values = []
    for index, value in enumerate(prices):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or value <= 0.0
        ):
            raise ValueError(f"price {index} must be a finite positive number")
        values.append(float(value))

    intervals = [
        {
            "start_date": dates[index - 1],
            "end_date": dates[index],
            "calendar_gap_days": (parsed_dates[index] - parsed_dates[index - 1]).days,
        }
        for index in range(1, len(dates))
    ]
    return_periods = [
        {
            "start_date": dates[index - 1],
            "end_date": dates[index],
            "return": values[index] / values[index - 1] - 1.0,
        }
        for index in range(1, len(values))
    ]
    for period in return_periods:
        period["absolute_return"] = abs(period["return"])

    unchanged_runs = _unchanged_runs(dates, values)
    longest_unchanged = max(
        unchanged_runs,
        key=lambda item: item["unchanged_transitions"],
        default=None,
    )
    largest_gap = max(
        intervals,
        key=lambda item: item["calendar_gap_days"],
    )
    largest_return = max(
        return_periods,
        key=lambda item: item["absolute_return"],
    )
    longest_unchanged_count = (
        longest_unchanged["unchanged_transitions"]
        if longest_unchanged is not None
        else 0
    )

    failures = []
    if maximum_gap is not None and largest_gap["calendar_gap_days"] > maximum_gap:
        failures.append(
            {
                "metric": "maximum_calendar_gap_days",
                "actual": largest_gap["calendar_gap_days"],
                "maximum": maximum_gap,
                "excess": largest_gap["calendar_gap_days"] - maximum_gap,
            }
        )
    if maximum_unchanged is not None and longest_unchanged_count > maximum_unchanged:
        failures.append(
            {
                "metric": "longest_unchanged_run",
                "actual": longest_unchanged_count,
                "maximum": maximum_unchanged,
                "excess": longest_unchanged_count - maximum_unchanged,
            }
        )
    if maximum_return is not None and largest_return["absolute_return"] > maximum_return:
        failures.append(
            {
                "metric": "maximum_absolute_return",
                "actual": largest_return["absolute_return"],
                "maximum": maximum_return,
                "excess": largest_return["absolute_return"] - maximum_return,
            }
        )

    return {
        "passed": not failures,
        "metrics": {
            "observations": len(values),
            "return_observations": len(return_periods),
            "start_date": dates[0],
            "end_date": dates[-1],
            "mean_calendar_gap_days": sum(
                item["calendar_gap_days"] for item in intervals
            )
            / len(intervals),
            "calendar_gaps_over_one_day": sum(
                item["calendar_gap_days"] > 1 for item in intervals
            ),
            "maximum_calendar_gap": largest_gap,
            "unchanged_return_count": sum(
                period["return"] == 0.0 for period in return_periods
            ),
            "unchanged_run_count": len(unchanged_runs),
            "longest_unchanged_run": longest_unchanged,
            "maximum_absolute_return": largest_return,
        },
        "thresholds": {
            "max_calendar_gap_days": maximum_gap,
            "max_unchanged_run": maximum_unchanged,
            "max_abs_return": maximum_return,
        },
        "failures": failures,
        "largest_calendar_gaps": sorted(
            intervals,
            key=lambda item: (-item["calendar_gap_days"], item["start_date"]),
        )[:max_details],
        "unchanged_runs": sorted(
            unchanged_runs,
            key=lambda item: (-item["unchanged_transitions"], item["start_date"]),
        )[:max_details],
        "largest_absolute_returns": sorted(
            return_periods,
            key=lambda item: (-item["absolute_return"], item["end_date"]),
        )[:max_details],
        "details_truncated": {
            "calendar_gaps": len(intervals) > max_details,
            "unchanged_runs": len(unchanged_runs) > max_details,
            "absolute_returns": len(return_periods) > max_details,
        },
        "settings": {"max_details": max_details},
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
    parser.add_argument("--max-calendar-gap-days", type=int)
    parser.add_argument("--max-unchanged-run", type=int)
    parser.add_argument("--max-abs-return", type=float)
    parser.add_argument("--max-details", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        if args.output is not None and _paths_alias(args.dataset, args.output):
            raise ValueError("output must not alias the source dataset")
        dates, prices = load_price_csv(args.dataset)
        report = audit_price_series(
            dates,
            prices,
            max_calendar_gap_days=args.max_calendar_gap_days,
            max_unchanged_run=args.max_unchanged_run,
            max_abs_return=args.max_abs_return,
            max_details=args.max_details,
        )
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
