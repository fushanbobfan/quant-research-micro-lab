"""A minimal moving-average backtest with explicit anti-look-ahead logic."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _moving_average(values: Sequence[float], window: int) -> list[float | None]:
    if window <= 0:
        raise ValueError("window must be positive")
    result: list[float | None] = [None] * len(values)
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= window:
            running -= values[index - window]
        if index >= window - 1:
            result[index] = running / window
    return result


def maximum_drawdown(equity: Sequence[float]) -> float:
    if not equity:
        return 0.0
    peak = equity[0]
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        worst = min(worst, value / peak - 1.0)
    return worst


def backtest_crossover(
    prices: Sequence[float],
    short_window: int = 5,
    long_window: int = 20,
    transaction_cost_bps: float = 0.0,
) -> dict[str, float | list[float]]:
    if len(prices) < long_window + 1:
        raise ValueError("prices must include at least long_window + 1 observations")
    if short_window >= long_window:
        raise ValueError("short_window must be smaller than long_window")
    if any(price <= 0 for price in prices):
        raise ValueError("prices must be positive")
    if not 0 <= transaction_cost_bps < 10_000:
        raise ValueError("transaction_cost_bps must be between 0 and 10,000")

    short_average = _moving_average(prices, short_window)
    long_average = _moving_average(prices, long_window)
    signals = [
        1.0 if short is not None and long is not None and short > long else 0.0
        for short, long in zip(short_average, long_average)
    ]

    equity = [1.0]
    gross_equity = [1.0]
    strategy_returns: list[float] = []
    total_turnover = 0.0
    previous_position = 0.0
    cost_rate = transaction_cost_bps / 10_000
    for index in range(1, len(prices)):
        asset_return = prices[index] / prices[index - 1] - 1.0
        position = signals[index - 1]
        turnover = abs(position - previous_position)
        gross_return = position * asset_return
        strategy_return = (1.0 + gross_return) * (1.0 - turnover * cost_rate) - 1.0
        strategy_returns.append(strategy_return)
        equity.append(equity[-1] * (1.0 + strategy_return))
        gross_equity.append(gross_equity[-1] * (1.0 + gross_return))
        total_turnover += turnover
        previous_position = position

    mean_return = sum(strategy_returns) / len(strategy_returns)
    variance = sum((value - mean_return) ** 2 for value in strategy_returns) / len(strategy_returns)
    return {
        "total_return": equity[-1] - 1.0,
        "gross_total_return": gross_equity[-1] - 1.0,
        "cost_drag": gross_equity[-1] - equity[-1],
        "total_turnover": total_turnover,
        "transaction_cost_bps": transaction_cost_bps,
        "annualized_volatility": math.sqrt(variance) * math.sqrt(252),
        "maximum_drawdown": maximum_drawdown(equity),
        "equity": equity,
        "gross_equity": gross_equity,
    }
