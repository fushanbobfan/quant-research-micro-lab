"""Compare gross and net results on a small synthetic price series."""

from quant_research_micro_lab import backtest_crossover


prices = [100, 100, 101, 103, 105, 104, 102, 101, 103, 106]

for cost in (0, 10, 50):
    result = backtest_crossover(
        prices,
        short_window=2,
        long_window=3,
        transaction_cost_bps=cost,
    )
    print(
        f"{cost:>2} bps | gross={result['gross_total_return']:.2%} "
        f"net={result['total_return']:.2%} turnover={result['total_turnover']:.1f}"
    )
