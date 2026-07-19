"""Small quantitative research primitives."""

from typing import Any

from .backtest import backtest_crossover, maximum_drawdown

__all__ = [
    "analyze_drawdowns",
    "backtest_crossover",
    "build_trade_ledger",
    "compare_to_benchmark",
    "load_equity_csv",
    "load_price_csv",
    "maximum_drawdown",
    "sweep_crossover",
    "walk_forward_crossover",
]


def __getattr__(name: str) -> Any:
    if name in {"analyze_drawdowns", "load_equity_csv"}:
        from .risk import analyze_drawdowns, load_equity_csv

        return {
            "analyze_drawdowns": analyze_drawdowns,
            "load_equity_csv": load_equity_csv,
        }[name]
    if name == "build_trade_ledger":
        from .trades import build_trade_ledger

        return build_trade_ledger
    if name == "load_price_csv":
        from .cli import load_price_csv

        return load_price_csv
    if name == "compare_to_benchmark":
        from .benchmark import compare_to_benchmark

        return compare_to_benchmark
    if name == "sweep_crossover":
        from .sweep import sweep_crossover

        return sweep_crossover
    if name == "walk_forward_crossover":
        from .walk_forward import walk_forward_crossover

        return walk_forward_crossover
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
