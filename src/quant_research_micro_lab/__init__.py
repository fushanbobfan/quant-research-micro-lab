"""Small quantitative research primitives."""

from typing import Any

from .backtest import backtest_crossover, maximum_drawdown

__all__ = ["backtest_crossover", "load_price_csv", "maximum_drawdown"]


def __getattr__(name: str) -> Any:
    if name == "load_price_csv":
        from .cli import load_price_csv

        return load_price_csv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
