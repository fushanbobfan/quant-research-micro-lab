# Quant Research Micro Lab

Transparent, dependency-free building blocks for learning how backtests work.

The backtest module implements a moving-average crossover strategy with a one-period execution lag to avoid look-ahead bias. It reports total return, annualized volatility, and maximum drawdown.

Optional transaction costs are expressed in basis points and charged when the executed position changes. Results include gross and net equity, compounded cost drag, and total one-way turnover. The final open position is not forcibly liquidated, so its closing cost is not included.

```powershell
python -m unittest discover -s tests
python examples/transaction_cost_demo.py
```

## CSV command line workflow

The `quant-backtest` command runs the same audited crossover logic on a `date,close` CSV file. Dates must be unique, strictly increasing ISO dates, and closes must be finite positive numbers.

```powershell
python -m quant_research_micro_lab.cli examples/synthetic_prices.csv `
  --short-window 2 --long-window 3 --transaction-cost-bps 10 `
  --equity-output equity.csv
```

The JSON report records the observation range and all performance fields. The optional equity export aligns each input date with net and gross equity, which makes downstream checking straightforward. Malformed input and invalid backtest parameters return exit code `2` without writing a result.

The example uses synthetic prices to show how greater costs reduce the reported net result. This project is educational software, not investment advice. It does not execute trades or make return guarantees; examples should use synthetic or properly licensed data.
