# Quant Research Micro Lab

Transparent, dependency-free building blocks for learning how backtests work.

The backtest module implements a moving-average crossover strategy with a one-period execution lag to avoid look-ahead bias. It reports total return, annualized volatility, and maximum drawdown.

Optional transaction costs are expressed in basis points and charged when the executed position changes. Results include gross and net equity, compounded cost drag, and total one-way turnover. The final open position is not forcibly liquidated, so its closing cost is not included.

```powershell
python -m unittest discover -s tests
python examples/transaction_cost_demo.py
```

The example uses synthetic prices to show how greater costs reduce the reported net result. This project is educational software, not investment advice. It does not execute trades or make return guarantees; examples should use synthetic or properly licensed data.
