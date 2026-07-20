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

## Trade ledger

Turn the same lagged crossover run into a dated, reviewable list of entries and exits:

```powershell
python -m quant_research_micro_lab.trades examples/synthetic_prices.csv `
  --short-window 2 --long-window 3 `
  --transaction-cost-bps 10 --output trades.json
```

Each closed trade records its execution dates and prices, holding observations, gross return, net return, and compounded cost drag. Open positions are marked to the final observation with a `null` exit and do not assume a future liquidation cost. The summary includes closed-trade win rate and return statistics alongside the full backtest return, cost drag, and turnover, so the ledger can be reconciled with `quant-backtest`.

Signals use the existing one-observation execution lag. An entry cost is included when a position opens, an exit cost is included only when it actually closes, and irregular dates are counted as observations rather than invented calendar durations. The ledger describes a historical rule on supplied data; it does not represent executable orders or expected future returns.

## Drawdown diagnostics

The `quant-risk` command consumes the equity CSV written by `quant-backtest` and reconstructs each peak-to-trough-to-recovery episode:

```powershell
python -m quant_research_micro_lab.cli examples/synthetic_prices.csv `
  --short-window 2 --long-window 3 --transaction-cost-bps 10 `
  --equity-output equity.csv
python -m quant_research_micro_lab.risk equity.csv
```

Use `--column gross_equity` to inspect the pre-cost curve. The deterministic JSON report includes the current drawdown, maximum drawdown episode, longest underwater episode, and every episode in chronological order. Each episode records peak, trough, and optional recovery dates plus the number of underwater observations. Observation counts are intentionally distinct from calendar-day duration so irregular market calendars are not misrepresented.

## Empirical return-tail diagnostics

Drawdown describes the path of an equity curve, while a return tail shows the worst individual periods in the supplied sample. The `quant-tail-risk` command reads the same strict equity export and reports the worst and best period returns, loss-period rate, zero-target downside deviation, and a dated lower-tail subset:

```powershell
python -m quant_research_micro_lab.tail_risk examples/tail-risk-equity.csv `
  --confidence 0.95
```

For `n` period returns, confidence `c` selects exactly `ceil((1 - c) * n)` of the lowest returns, with a minimum of one. The report includes that fixed sample count, the least-severe selected return as `tail_cutoff_return`, the mean selected return, and every selected start/end date. Equal returns at the boundary are resolved by their end-date order rather than expanding the tail unpredictably. Use `--column gross_equity` to inspect the pre-cost curve.

Downside deviation is the square root of the mean squared negative return using a zero target and every supplied period; it is not annualized. These are descriptive historical sample statistics, not forecasts, confidence intervals, or claims about future loss probabilities. Small datasets can make a high-confidence tail depend on a single observation.

## Benchmark diagnostics

Compare a backtest equity export with a benchmark `date,close` series on the exact same dates:

```powershell
python -m quant_research_micro_lab.benchmark `
  examples/benchmark-strategy.csv examples/benchmark-prices.csv `
  --periods-per-year 252
```

The report includes each total return, the strategy growth multiple relative to the benchmark, annualized volatility, tracking error, information ratio, beta, correlation, and the share of periods with positive active return. Use `--strategy-column gross_equity` to inspect the pre-cost curve. Dates must be unique, increasing, and identical across both files so accidental row offsets cannot become performance results.

Volatility, tracking error, covariance, and variance use population moments over the supplied return observations. Information ratio annualizes the arithmetic mean active return; beta, correlation, or information ratio is `null` when its denominator is zero. Benchmark selection and sampling frequency materially affect every diagnostic, and historical comparisons do not imply future performance.

## Parameter grid evaluation

Evaluate several crossover settings against the same validated price history without writing custom loops. Repeat each window option to define the grid:

```powershell
python -m quant_research_micro_lab.sweep examples/synthetic_prices.csv `
  --short-window 2 --short-window 3 `
  --long-window 4 --long-window 5 `
  --transaction-cost-bps 10 --rank-by total_return
```

The compact JSON report ranks every valid short/long pair and preserves invalid pairs in `skipped_pairs` instead of silently losing them. Rankings are deterministic, including window-based tie breaking. `total_return` and `maximum_drawdown` are maximized; `annualized_volatility` and `total_turnover` are minimized. Full equity curves remain available through `quant-backtest` for any candidate that merits closer inspection.

Parameter rankings are in-sample comparisons, not evidence of future performance. Use held-out data and account for multiple testing before drawing research conclusions.

## Rolling walk-forward evaluation

Use consecutive training and test windows to check how parameter selection behaves outside the data that ranked it. Each fold chooses the top crossover pair only from its rolling training window, then measures that pair over the immediately following, non-overlapping test window:

```powershell
python -m quant_research_micro_lab.walk_forward examples/walk_forward_prices.csv `
  --short-window 2 --short-window 3 `
  --long-window 4 --long-window 5 `
  --train-size 10 --test-size 5 `
  --transaction-cost-bps 10 --rank-by total_return
```

The JSON report records dated fold boundaries, the selected parameters and training score, per-fold test metrics, parameter selection counts, and a compounded out-of-sample summary. Training windows advance by `test-size`; only complete test windows are evaluated, and any remaining tail is reported as `unused_trailing_observations`. The selected strategy is run through its training window before the test boundary so lagged signals and transaction costs carry into the first test return without reading future prices.

Walk-forward results reduce one obvious source of in-sample bias but do not eliminate selection bias, regime risk, data quality problems, or trading frictions. Repeatedly changing the grid after seeing test results also contaminates the holdout.

The example uses synthetic prices to show how greater costs reduce the reported net result. This project is educational software, not investment advice. It does not execute trades or make return guarantees; examples should use synthetic or properly licensed data.
