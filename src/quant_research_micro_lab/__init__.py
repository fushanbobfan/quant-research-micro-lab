"""Small quantitative research primitives."""

from typing import Any

from .backtest import backtest_crossover, maximum_drawdown

__all__ = [
    "analyze_drawdowns",
    "backtest_crossover",
    "load_equity_csv",
    "load_price_csv",
    "maximum_drawdown",
]


def __getattr__(name: str) -> Any:
    if name in {"analyze_drawdowns", "load_equity_csv"}:
        from .risk import analyze_drawdowns, load_equity_csv

        return {
            "analyze_drawdowns": analyze_drawdowns,
            "load_equity_csv": load_equity_csv,
        }[name]
    if name == "load_price_csv":
        from .cli import load_price_csv

        return load_price_csv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
