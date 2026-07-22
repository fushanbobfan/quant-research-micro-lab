"""Measure rolling performance windows on a dated equity curve."""

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


def _maximum_drawdown(values: Sequence[float]) -> float:
    peak = values[0]
    maximum = 0.0
    for value in values:
        peak = max(peak, value)
        maximum = min(maximum, value / peak - 1.0)
    return maximum


def analyze_rolling_performance(
    equity: Sequence[float],
    *,
    window: int,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """Return overlapping fixed-return-count performance windows."""

    for name, value in (("window", window), ("periods_per_year", periods_per_year)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if len(equity) < window + 1:
        raise ValueError(
            "equity must contain at least window + 1 observations"
        )
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0
        for value in equity
    ):
        raise ValueError("equity values must be finite positive numbers")

    values = [float(value) for value in equity]
    windows = []
    for end_index in range(window, len(values)):
        start_index = end_index - window
        window_values = values[start_index : end_index + 1]
        returns = [
            window_values[index] / window_values[index - 1] - 1.0
            for index in range(1, len(window_values))
        ]
        mean_return = sum(returns) / window
        variance = sum((value - mean_return) ** 2 for value in returns) / window
        annualized_volatility = math.sqrt(variance * periods_per_year)
        total_return = window_values[-1] / window_values[0] - 1.0
        windows.append(
            {
                "start_index": start_index,
                "end_index": end_index,
                "total_return": total_return,
                "annualized_return": (1.0 + total_return)
                ** (periods_per_year / window)
                - 1.0,
                "annualized_volatility": annualized_volatility,
                "annualized_sharpe": (
                    mean_return * periods_per_year / annualized_volatility
                    if annualized_volatility > 0
                    else None
                ),
                "annualized_downside_deviation": math.sqrt(
                    sum(min(value, 0.0) ** 2 for value in returns)
                    / window
                    * periods_per_year
                ),
                "maximum_drawdown": _maximum_drawdown(window_values),
                "positive_return_rate": sum(value > 0 for value in returns)
                / window,
                "worst_period_return": min(returns),
            }
        )

    worst_total_return = min(
        windows,
        key=lambda item: (item["total_return"], item["end_index"]),
    )
    worst_drawdown = min(
        windows,
        key=lambda item: (item["maximum_drawdown"], item["end_index"]),
    )
    highest_volatility = max(
        windows,
        key=lambda item: (item["annualized_volatility"], -item["end_index"]),
    )
    return {
        "observations": len(values),
        "window_return_periods": window,
        "window_count": len(windows),
        "periods_per_year": periods_per_year,
        "latest_window": windows[-1],
        "worst_total_return_window": worst_total_return,
        "worst_drawdown_window": worst_drawdown,
        "highest_volatility_window": highest_volatility,
        "windows": windows,
    }


def _add_dates(window: dict[str, Any], dates: Sequence[str]) -> dict[str, Any]:
    return {
        **window,
        "start_date": dates[window["start_index"]],
        "end_date": dates[window["end_index"]],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--periods-per-year", type=int, default=252)
    parser.add_argument(
        "--column",
        choices=("equity", "gross_equity"),
        default="equity",
        help="curve to analyze from a quant-backtest equity export",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        dates, equity = load_equity_csv(args.dataset, args.column)
        report = analyze_rolling_performance(
            equity,
            window=args.window,
            periods_per_year=args.periods_per_year,
        )
        dated_windows = [_add_dates(item, dates) for item in report["windows"]]
        dated_report = {
            **report,
            "column": args.column,
            "start_date": dates[0],
            "end_date": dates[-1],
            "latest_window": _add_dates(report["latest_window"], dates),
            "worst_total_return_window": _add_dates(
                report["worst_total_return_window"], dates
            ),
            "worst_drawdown_window": _add_dates(
                report["worst_drawdown_window"], dates
            ),
            "highest_volatility_window": _add_dates(
                report["highest_volatility_window"], dates
            ),
            "windows": dated_windows,
        }
        rendered = json.dumps(dated_report, indent=2) + "\n"
        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
    except (OSError, UnicodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
