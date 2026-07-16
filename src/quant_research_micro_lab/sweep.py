"""Evaluate a deterministic grid of crossover parameter candidates."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .backtest import backtest_crossover
from .cli import load_price_csv


_RANK_DIRECTIONS = {
    "total_return": "maximize",
    "maximum_drawdown": "maximize",
    "annualized_volatility": "minimize",
    "total_turnover": "minimize",
}


def _validate_windows(name: str, values: Sequence[int]) -> list[int]:
    if not values:
        raise ValueError(f"at least one {name} is required")
    validated = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} values must be positive integers")
        if value in validated:
            raise ValueError(f"{name} values must be unique")
        validated.append(value)
    return validated


def sweep_crossover(
    prices: Sequence[float],
    *,
    short_windows: Sequence[int],
    long_windows: Sequence[int],
    transaction_cost_bps: float = 0.0,
    rank_by: str = "total_return",
) -> dict[str, Any]:
    """Backtest every valid short/long pair and rank the compact results."""

    shorts = _validate_windows("short_window", short_windows)
    longs = _validate_windows("long_window", long_windows)
    if rank_by not in _RANK_DIRECTIONS:
        choices = ", ".join(_RANK_DIRECTIONS)
        raise ValueError(f"rank_by must be one of: {choices}")

    results = []
    skipped_pairs = []
    for short_window in shorts:
        for long_window in longs:
            if short_window >= long_window:
                skipped_pairs.append(
                    {
                        "short_window": short_window,
                        "long_window": long_window,
                        "reason": "short_window must be smaller than long_window",
                    }
                )
                continue
            backtest = backtest_crossover(
                prices,
                short_window=short_window,
                long_window=long_window,
                transaction_cost_bps=transaction_cost_bps,
            )
            results.append(
                {
                    "short_window": short_window,
                    "long_window": long_window,
                    "total_return": backtest["total_return"],
                    "gross_total_return": backtest["gross_total_return"],
                    "cost_drag": backtest["cost_drag"],
                    "total_turnover": backtest["total_turnover"],
                    "annualized_volatility": backtest["annualized_volatility"],
                    "maximum_drawdown": backtest["maximum_drawdown"],
                }
            )

    if not results:
        raise ValueError("parameter grid does not contain a valid window pair")

    direction = _RANK_DIRECTIONS[rank_by]
    multiplier = -1 if direction == "maximize" else 1
    results.sort(
        key=lambda result: (
            multiplier * result[rank_by],
            result["short_window"],
            result["long_window"],
        )
    )
    for rank, result in enumerate(results, start=1):
        result["rank"] = rank

    return {
        "rank_by": rank_by,
        "direction": direction,
        "transaction_cost_bps": transaction_cost_bps,
        "candidate_count": len(results),
        "skipped_pairs": skipped_pairs,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--short-window",
        type=int,
        action="append",
        required=True,
        dest="short_windows",
        help="short lookback; repeat to build the grid",
    )
    parser.add_argument(
        "--long-window",
        type=int,
        action="append",
        required=True,
        dest="long_windows",
        help="long lookback; repeat to build the grid",
    )
    parser.add_argument("--transaction-cost-bps", type=float, default=0.0)
    parser.add_argument("--rank-by", choices=tuple(_RANK_DIRECTIONS), default="total_return")
    args = parser.parse_args(argv)

    try:
        dates, prices = load_price_csv(args.dataset)
        result = sweep_crossover(
            prices,
            short_windows=args.short_windows,
            long_windows=args.long_windows,
            transaction_cost_bps=args.transaction_cost_bps,
            rank_by=args.rank_by,
        )
    except (OSError, UnicodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    report = {
        "observations": len(dates),
        "start_date": dates[0],
        "end_date": dates[-1],
        **result,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
