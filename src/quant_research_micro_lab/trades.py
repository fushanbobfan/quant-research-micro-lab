"""Build an auditable trade ledger from the crossover strategy."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from datetime import date
from numbers import Real
from pathlib import Path
from typing import Any, cast

from .backtest import _moving_average, backtest_crossover
from .cli import load_price_csv


def _validate_observations(
    dates: Sequence[str], prices: Sequence[float]
) -> tuple[list[str], list[float]]:
    if len(dates) != len(prices):
        raise ValueError("dates and prices must have the same length")

    validated_dates = []
    validated_prices = []
    previous_date: date | None = None
    for index, (raw_date, raw_price) in enumerate(zip(dates, prices)):
        if not isinstance(raw_date, str):
            raise ValueError(f"date {index} must be an ISO date string")
        try:
            parsed_date = date.fromisoformat(raw_date)
        except ValueError as error:
            raise ValueError(f"date {index} must be a valid ISO date") from error
        if previous_date is not None and parsed_date <= previous_date:
            raise ValueError("dates must be strictly increasing")
        if (
            isinstance(raw_price, bool)
            or not isinstance(raw_price, Real)
            or not math.isfinite(raw_price)
            or raw_price <= 0
        ):
            raise ValueError(f"price {index} must be finite and positive")
        validated_dates.append(parsed_date.isoformat())
        validated_prices.append(float(raw_price))
        previous_date = parsed_date
    return validated_dates, validated_prices


def build_trade_ledger(
    dates: Sequence[str],
    prices: Sequence[float],
    *,
    short_window: int = 5,
    long_window: int = 20,
    transaction_cost_bps: float = 0.0,
) -> dict[str, Any]:
    """Return completed and open crossover trades aligned to execution dates."""

    validated_dates, validated_prices = _validate_observations(dates, prices)
    result = backtest_crossover(
        validated_prices,
        short_window=short_window,
        long_window=long_window,
        transaction_cost_bps=transaction_cost_bps,
    )
    equity = cast(list[float], result["equity"])

    short_average = _moving_average(validated_prices, short_window)
    long_average = _moving_average(validated_prices, long_window)
    signals = [
        1.0 if short is not None and long is not None and short > long else 0.0
        for short, long in zip(short_average, long_average)
    ]

    trades = []
    closed_net_returns = []
    active: dict[str, Any] | None = None
    previous_position = 0.0
    for price_index in range(1, len(validated_prices)):
        position = signals[price_index - 1]
        execution_index = price_index - 1
        if position > previous_position:
            active = {
                "entry_index": execution_index,
                "entry_date": validated_dates[execution_index],
                "entry_price": validated_prices[execution_index],
                "entry_equity": equity[price_index - 1],
            }
        elif position < previous_position:
            assert active is not None
            gross_return = (
                validated_prices[execution_index] / active["entry_price"] - 1.0
            )
            net_return = equity[price_index] / active["entry_equity"] - 1.0
            closed_net_returns.append(net_return)
            trades.append(
                {
                    "trade_id": len(trades) + 1,
                    "status": "closed",
                    "entry_date": active["entry_date"],
                    "entry_price": active["entry_price"],
                    "exit_date": validated_dates[execution_index],
                    "exit_price": validated_prices[execution_index],
                    "holding_observations": execution_index
                    - active["entry_index"],
                    "gross_return": gross_return,
                    "net_return": net_return,
                    "cost_drag": gross_return - net_return,
                }
            )
            active = None
        previous_position = position

    if active is not None:
        gross_return = validated_prices[-1] / active["entry_price"] - 1.0
        net_return = equity[-1] / active["entry_equity"] - 1.0
        trades.append(
            {
                "trade_id": len(trades) + 1,
                "status": "open",
                "entry_date": active["entry_date"],
                "entry_price": active["entry_price"],
                "exit_date": None,
                "exit_price": None,
                "mark_date": validated_dates[-1],
                "mark_price": validated_prices[-1],
                "holding_observations": len(validated_prices)
                - 1
                - active["entry_index"],
                "gross_return": gross_return,
                "net_return": net_return,
                "cost_drag": gross_return - net_return,
            }
        )

    winning_closed_trades = sum(value > 0 for value in closed_net_returns)
    return {
        "observations": len(validated_prices),
        "start_date": validated_dates[0],
        "end_date": validated_dates[-1],
        "strategy": {
            "short_window": short_window,
            "long_window": long_window,
            "transaction_cost_bps": transaction_cost_bps,
            "execution_lag_observations": 1,
        },
        "summary": {
            "trade_count": len(trades),
            "closed_trade_count": len(closed_net_returns),
            "open_trade_count": sum(trade["status"] == "open" for trade in trades),
            "winning_closed_trades": winning_closed_trades,
            "closed_win_rate": (
                winning_closed_trades / len(closed_net_returns)
                if closed_net_returns
                else None
            ),
            "average_closed_net_return": (
                sum(closed_net_returns) / len(closed_net_returns)
                if closed_net_returns
                else None
            ),
            "best_closed_net_return": (
                max(closed_net_returns) if closed_net_returns else None
            ),
            "worst_closed_net_return": (
                min(closed_net_returns) if closed_net_returns else None
            ),
            "total_return": result["total_return"],
            "gross_total_return": result["gross_total_return"],
            "cost_drag": result["cost_drag"],
            "total_turnover": result["total_turnover"],
        },
        "trades": trades,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--short-window", type=int, default=5)
    parser.add_argument("--long-window", type=int, default=20)
    parser.add_argument("--transaction-cost-bps", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        dates, prices = load_price_csv(args.dataset)
        report = build_trade_ledger(
            dates,
            prices,
            short_window=args.short_window,
            long_window=args.long_window,
            transaction_cost_bps=args.transaction_cost_bps,
        )
        rendered = json.dumps(report, indent=2) + "\n"
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
