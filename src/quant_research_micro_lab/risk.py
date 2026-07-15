"""Inspect drawdown episodes in an equity curve."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Sequence
from datetime import date
from numbers import Real
from pathlib import Path
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


def load_equity_csv(path: Path, column: str = "equity") -> tuple[list[str], list[float]]:
    """Load a net or gross curve from the backtest equity export format."""

    if column not in {"equity", "gross_equity"}:
        raise ValueError("column must be equity or gross_equity")

    dates: list[str] = []
    values: list[float] = []
    previous_date: date | None = None
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["date", "equity", "gross_equity"]:
            raise ValueError("CSV header must be exactly: date,equity,gross_equity")

        for row_number, row in enumerate(reader, start=2):
            try:
                parsed_date = date.fromisoformat(row.get("date") or "")
            except ValueError as error:
                raise ValueError(f"row {row_number} has an invalid ISO date") from error
            if previous_date is not None and parsed_date <= previous_date:
                raise ValueError(f"row {row_number} date must be strictly increasing")
            try:
                value = float(row.get(column) or "")
            except ValueError as error:
                raise ValueError(f"row {row_number} has an invalid {column}") from error
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"row {row_number} {column} must be finite and positive"
                )

            dates.append(parsed_date.isoformat())
            values.append(value)
            previous_date = parsed_date

    if not dates:
        raise ValueError("CSV must contain at least one equity row")
    return dates, values


def _add_dates(episode: dict[str, Any] | None, dates: Sequence[str]) -> dict[str, Any] | None:
    if episode is None:
        return None
    recovery_index = episode["recovery_index"]
    return {
        **episode,
        "peak_date": dates[episode["peak_index"]],
        "trough_date": dates[episode["trough_index"]],
        "recovery_date": dates[recovery_index] if recovery_index is not None else None,
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
    args = parser.parse_args(argv)

    try:
        dates, values = load_equity_csv(args.dataset, args.column)
        report = analyze_drawdowns(values)
    except (OSError, UnicodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    report = {
        **report,
        "column": args.column,
        "start_date": dates[0],
        "end_date": dates[-1],
        "maximum_drawdown": _add_dates(report["maximum_drawdown"], dates),
        "longest_underwater": _add_dates(report["longest_underwater"], dates),
        "episodes": [_add_dates(episode, dates) for episode in report["episodes"]],
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
