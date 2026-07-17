"""Select crossover parameters on rolling training windows and test them forward."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .backtest import backtest_crossover, maximum_drawdown
from .sweep import sweep_crossover


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _returns(curve: Sequence[float], start_index: int) -> list[float]:
    return [
        curve[index] / curve[index - 1] - 1.0
        for index in range(start_index + 1, len(curve))
    ]


def _equity(returns: Sequence[float]) -> list[float]:
    values = [1.0]
    for value in returns:
        values.append(values[-1] * (1.0 + value))
    return values


def _annualized_volatility(returns: Sequence[float]) -> float:
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / len(returns)
    return math.sqrt(variance) * math.sqrt(252)


def walk_forward_crossover(
    prices: Sequence[float],
    *,
    short_windows: Sequence[int],
    long_windows: Sequence[int],
    train_size: int,
    test_size: int,
    transaction_cost_bps: float = 0.0,
    rank_by: str = "total_return",
) -> dict[str, Any]:
    """Select on each rolling train window and aggregate the following test returns."""

    train_size = _positive_integer("train_size", train_size)
    test_size = _positive_integer("test_size", test_size)
    validated_prices = []
    for index, price in enumerate(prices):
        if (
            isinstance(price, bool)
            or not isinstance(price, (int, float))
            or not math.isfinite(price)
            or price <= 0
        ):
            raise ValueError(f"price {index} must be a finite positive number")
        validated_prices.append(float(price))
    if len(validated_prices) < train_size + test_size:
        raise ValueError("prices must include at least train_size + test_size observations")

    fold_count = (len(validated_prices) - train_size) // test_size
    folds = []
    out_of_sample_returns: list[float] = []
    gross_out_of_sample_returns: list[float] = []
    selection_counts: dict[tuple[int, int], int] = {}
    for fold_index in range(fold_count):
        train_start = fold_index * test_size
        train_end = train_start + train_size
        test_end = train_end + test_size
        training_prices = validated_prices[train_start:train_end]
        sweep = sweep_crossover(
            training_prices,
            short_windows=short_windows,
            long_windows=long_windows,
            transaction_cost_bps=transaction_cost_bps,
            rank_by=rank_by,
        )
        selected = sweep["results"][0]
        short_window = selected["short_window"]
        long_window = selected["long_window"]
        selection_key = (short_window, long_window)
        selection_counts[selection_key] = selection_counts.get(selection_key, 0) + 1

        evaluation = backtest_crossover(
            validated_prices[train_start:test_end],
            short_window=short_window,
            long_window=long_window,
            transaction_cost_bps=transaction_cost_bps,
        )
        boundary = train_size - 1
        test_returns = _returns(evaluation["equity"], boundary)
        gross_test_returns = _returns(evaluation["gross_equity"], boundary)
        test_equity = _equity(test_returns)
        gross_test_equity = _equity(gross_test_returns)
        out_of_sample_returns.extend(test_returns)
        gross_out_of_sample_returns.extend(gross_test_returns)
        folds.append(
            {
                "fold": fold_index + 1,
                "train_start_index": train_start,
                "train_end_index": train_end - 1,
                "test_start_index": train_end,
                "test_end_index": test_end - 1,
                "selected_short_window": short_window,
                "selected_long_window": long_window,
                "training_score": selected[rank_by],
                "test_total_return": test_equity[-1] - 1.0,
                "test_gross_total_return": gross_test_equity[-1] - 1.0,
                "test_cost_drag": gross_test_equity[-1] - test_equity[-1],
                "test_annualized_volatility": _annualized_volatility(test_returns),
                "test_maximum_drawdown": maximum_drawdown(test_equity),
            }
        )

    out_of_sample_equity = _equity(out_of_sample_returns)
    gross_out_of_sample_equity = _equity(gross_out_of_sample_returns)
    selections = [
        {
            "short_window": short_window,
            "long_window": long_window,
            "fold_count": count,
        }
        for (short_window, long_window), count in sorted(selection_counts.items())
    ]
    return {
        "train_size": train_size,
        "test_size": test_size,
        "rank_by": rank_by,
        "transaction_cost_bps": transaction_cost_bps,
        "fold_count": fold_count,
        "evaluated_observations": len(out_of_sample_returns),
        "unused_trailing_observations": len(validated_prices)
        - (train_size + fold_count * test_size),
        "selection_counts": selections,
        "out_of_sample": {
            "total_return": out_of_sample_equity[-1] - 1.0,
            "gross_total_return": gross_out_of_sample_equity[-1] - 1.0,
            "cost_drag": gross_out_of_sample_equity[-1] - out_of_sample_equity[-1],
            "annualized_volatility": _annualized_volatility(out_of_sample_returns),
            "maximum_drawdown": maximum_drawdown(out_of_sample_equity),
        },
        "folds": folds,
    }
