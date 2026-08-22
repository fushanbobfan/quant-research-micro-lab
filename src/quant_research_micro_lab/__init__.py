"""Small quantitative research primitives."""

from typing import Any

from .backtest import backtest_crossover, maximum_drawdown

__all__ = [
    "analyze_drawdowns",
    "analyze_return_tail",
    "analyze_return_dependence",
    "analyze_correlation_drift",
    "analyze_cost_sensitivity",
    "analyze_rolling_performance",
    "analyze_risk_contributions",
    "audit_allocation_drift",
    "audit_group_exposure",
    "audit_portfolio_exposure",
    "audit_portfolio_turnover",
    "audit_panel_coverage",
    "audit_price_series",
    "backtest_crossover",
    "backtest_var_forecasts",
    "bootstrap_equity_performance",
    "build_trade_ledger",
    "compare_to_benchmark",
    "evaluate_portfolio_stress",
    "load_equity_csv",
    "load_group_exposure_csv",
    "load_portfolio_csv",
    "load_price_csv",
    "load_returns_csv",
    "maximum_drawdown",
    "sweep_crossover",
    "walk_forward_crossover",
]


def __getattr__(name: str) -> Any:
    if name == "audit_allocation_drift":
        from .allocation_drift import audit_allocation_drift

        return audit_allocation_drift
    if name in {"audit_group_exposure", "load_group_exposure_csv"}:
        from .group_exposure import audit_group_exposure, load_group_exposure_csv

        return {
            "audit_group_exposure": audit_group_exposure,
            "load_group_exposure_csv": load_group_exposure_csv,
        }[name]
    if name == "analyze_correlation_drift":
        from .correlation_drift import analyze_correlation_drift

        return analyze_correlation_drift
    if name == "analyze_cost_sensitivity":
        from .cost_sensitivity import analyze_cost_sensitivity

        return analyze_cost_sensitivity
    if name == "backtest_var_forecasts":
        from .var_backtest import backtest_var_forecasts

        return backtest_var_forecasts
    if name == "audit_portfolio_turnover":
        from .turnover import audit_portfolio_turnover

        return audit_portfolio_turnover
    if name == "audit_panel_coverage":
        from .panel_coverage import audit_panel_coverage

        return audit_panel_coverage
    if name in {"audit_portfolio_exposure", "load_portfolio_csv"}:
        from .exposure import audit_portfolio_exposure, load_portfolio_csv

        return {
            "audit_portfolio_exposure": audit_portfolio_exposure,
            "load_portfolio_csv": load_portfolio_csv,
        }[name]
    if name == "bootstrap_equity_performance":
        from .bootstrap import bootstrap_equity_performance

        return bootstrap_equity_performance
    if name == "audit_price_series":
        from .price_audit import audit_price_series

        return audit_price_series
    if name in {"analyze_drawdowns", "load_equity_csv"}:
        from .risk import analyze_drawdowns, load_equity_csv

        return {
            "analyze_drawdowns": analyze_drawdowns,
            "load_equity_csv": load_equity_csv,
        }[name]
    if name == "analyze_return_tail":
        from .tail_risk import analyze_return_tail

        return analyze_return_tail
    if name == "analyze_return_dependence":
        from .dependence import analyze_return_dependence

        return analyze_return_dependence
    if name in {"analyze_risk_contributions", "load_returns_csv"}:
        from .risk_contribution import analyze_risk_contributions, load_returns_csv

        return {
            "analyze_risk_contributions": analyze_risk_contributions,
            "load_returns_csv": load_returns_csv,
        }[name]
    if name == "analyze_rolling_performance":
        from .rolling import analyze_rolling_performance

        return analyze_rolling_performance
    if name == "build_trade_ledger":
        from .trades import build_trade_ledger

        return build_trade_ledger
    if name == "load_price_csv":
        from .cli import load_price_csv

        return load_price_csv
    if name == "compare_to_benchmark":
        from .benchmark import compare_to_benchmark

        return compare_to_benchmark
    if name == "evaluate_portfolio_stress":
        from .stress import evaluate_portfolio_stress

        return evaluate_portfolio_stress
    if name == "sweep_crossover":
        from .sweep import sweep_crossover

        return sweep_crossover
    if name == "walk_forward_crossover":
        from .walk_forward import walk_forward_crossover

        return walk_forward_crossover
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
