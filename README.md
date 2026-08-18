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

## Transaction-cost sensitivity audit

A single cost assumption can hide how quickly a backtest degrades. Run the same lagged crossover strategy across an explicit grid that includes a zero-cost baseline:

```powershell
python -m quant_research_micro_lab.cost_sensitivity `
  examples/cost-sensitivity-prices.csv `
  --short-window 2 --long-window 3 `
  --transaction-cost-bps 0 `
  --transaction-cost-bps 100 `
  --transaction-cost-bps 500 `
  --max-return-degradation 0.20
```

The report sorts unique tested costs, repeats the compact return, drawdown, volatility, turnover, and cost-drag diagnostics for each scenario, and measures every return change from the zero-cost run. It also identifies the first tested nonpositive result and, when present, the adjacent tested costs that bracket a positive-to-nonpositive change. That bracket is not an interpolated or estimated break-even cost. Optional gates bound the zero-to-highest-cost return degradation or require a minimum historical return at the highest tested cost. `--output` writes JSON only when it does not alias the source CSV.

Exit code `0` means configured gates passed, `1` reports structured threshold failures, and `2` identifies malformed prices, an invalid or duplicate cost grid, a missing zero baseline, unsafe output aliasing, or invalid configuration. Costs use the backtest's one-way turnover model; the final open position is not forcibly liquidated.

This is a finite historical scenario comparison, not an execution simulator, cost forecast, strategy recommendation, or profitability guarantee. The model omits spread variation, slippage, market impact, liquidity, partial fills, taxes, borrow, financing, and changing fees. A tested minimum return is a reproducibility gate for supplied data, not evidence of future performance, and the selected window parameters can still be overfit.

## Price-series data quality audit

Inspect a strict `date,close` research file before using it in a backtest:

```powershell
python -m quant_research_micro_lab.price_audit examples/price-quality.csv `
  --max-calendar-gap-days 3 `
  --max-unchanged-run 3 `
  --max-abs-return 0.30
```

The report measures the mean and maximum calendar-day gap, counts gaps longer than one day, identifies consecutive exact unchanged closes, and ranks the largest absolute simple returns. Summary extrema and bounded detail lists retain dates so a researcher can trace each warning to the input. An unchanged run is measured in transitions: three unchanged transitions span four consecutive observations at the same close.

Exit code `0` means every configured gap, unchanged-run, and absolute-return maximum passed; `1` reports all structured threshold failures; and `2` identifies malformed CSV, fewer than two observations, unsafe output aliasing, or invalid configuration. The source file is opened read-only, and the command never repairs prices or places trades.

Calendar gaps are not exchange-calendar missing-session tests: weekends, holidays, suspensions, and assets with different trading schedules can all produce valid gaps. Exact unchanged prices can be genuine for illiquid assets, while large returns can reflect real moves, splits, distributions, currency effects, or missing adjustment data. The audit flags observations for review; it does not prove a data error, validate corporate-action treatment, forecast returns, or support a trading signal. Set gates for the intended asset, frequency, vendor, and adjustment convention before reviewing the final dataset.

## Portfolio exposure audit

Inspect synthetic or research portfolio weights without placing orders:

```powershell
python -m quant_research_micro_lab.exposure examples/portfolio-weights.csv `
  --max-gross-exposure 1.00 --max-abs-net-exposure 1.00 `
  --max-single-position 0.55 --max-concentration-hhi 0.42
```

The input header must be exactly `date,asset,weight`. Dates use `YYYY-MM-DD`, appear in non-decreasing order, and may contain several unique assets per snapshot. Weights are finite signed decimals; positive values are long, negative values are short, and an all-zero snapshot is rejected.

Each dated snapshot reports long, short, gross, net, and absolute net exposure; the largest absolute position; normalized absolute-weight HHI; its reciprocal effective-position count; and one-way turnover from the prior snapshot. Turnover is `0.5 * sum(abs(current_weight - previous_weight))` over the union of assets, so additions and removals remain visible. The first snapshot has no prior turnover. Summary fields preserve dated extrema and average diagnostics, while optional maximums produce stable, structured failures.

Exit code `0` means every configured exposure and concentration limit passed, `1` means valid weights exceeded at least one limit, and `2` identifies invalid CSV, duplicate positions, zero-exposure snapshots, or invalid configuration. This is a weight-file audit, not a portfolio optimizer or execution system. It does not model prices, liquidity, correlation, currency, borrow, margin, transaction costs, or future returns, so passing limits is not evidence that a portfolio is safe or profitable.

## Portfolio turnover audit

The exposure report includes one turnover number per snapshot. Use the dedicated turnover audit when a review also needs position-level changes, additions, removals, sign flips, and explicit change budgets:

```powershell
python -m quant_research_micro_lab.turnover examples/portfolio-weights.csv `
  --max-transition-turnover 0.35 `
  --max-position-change 0.45 `
  --max-cumulative-turnover 0.50
```

For every consecutive snapshot pair, the report evaluates the union of asset identifiers and calculates signed weight changes, total absolute change, and one-way turnover as `0.5 * sum(abs(change))`. Per-asset details are sorted by absolute change and bounded with `--max-details`, while complete transition counts, cumulative turnover, the largest transition, and the largest individual change remain visible. Optional maximums produce stable failures; exit codes are `0` for pass, `1` for a valid threshold failure, and `2` for malformed or oversized input, invalid settings, or unsafe output aliasing.

This is a read-only comparison of supplied weight snapshots, not a trade ledger, transaction-cost estimate, optimizer, or execution plan. The one-way convention is most interpretable for comparable normalized portfolios; deposits, withdrawals, leverage changes, derivatives, corporate actions, stale prices, and valuation changes can all make weight changes differ from traded notional. Asset identifiers and dated changes can be sensitive, and passing finite historical limits does not establish liquidity, capacity, profitability, or future behavior.

## Historical risk contributions

Estimate how a static portfolio's assets contribute to volatility under a sample covariance matrix:

```powershell
python -m quant_research_micro_lab.risk_contribution `
  examples/portfolio-weights.csv examples/asset-returns.csv `
  --max-largest-risk-share 0.65 `
  --max-risk-concentration-hhi 0.50
```

The portfolio input reuses `date,asset,weight`; the latest snapshot is selected unless `--date` names another available snapshot. The returns input is a strict wide CSV whose first column is `date` and remaining headers exactly match the snapshot's nonzero assets. It needs at least two strictly increasing observations, and its last date cannot extend beyond the selected snapshot, preventing accidental look-ahead in the audit.

The command uses sample covariance with `n - 1` in the denominator. It reports per-period and annualized portfolio volatility, the complete covariance matrix, standalone asset volatility, marginal volatility, and component volatility `weight * (covariance * weights) / portfolio volatility`. Component volatilities sum back to portfolio volatility within floating-point precision. Signed component fractions can be negative when an asset offsets other risk. Concentration gates instead normalize absolute component magnitudes, then report the largest share, HHI, and reciprocal effective contributor count.

Exit code `0` means both optional concentration maximums passed, `1` reports structured failures, and `2` identifies malformed CSV, mismatched assets, look-ahead dates, zero historical portfolio variance, unsafe output aliasing, or invalid configuration. Historical covariance is sample-dependent and can change abruptly. Linear weights omit nonlinear instruments, changing exposures, liquidity, price impact, currency, financing, and estimation error. This audit is not an optimizer, VaR model, capital-adequacy assessment, forecast, trading recommendation, or guarantee of future diversification.

## Portfolio stress scenarios

Apply explicit synthetic asset-return shocks to one dated portfolio snapshot without placing orders or estimating future returns:

```powershell
python -m quant_research_micro_lab.stress `
  examples/portfolio-weights.csv examples/stress-scenarios.csv `
  --max-loss 0.10
```

The portfolio input reuses the strict `date,asset,weight` format from `quant-exposure`. The scenario header must be exactly `scenario,asset,return`, with one unique row for every nonzero portfolio asset in each scenario. Simple asset returns must be finite and cannot be below `-1`. The latest portfolio date is selected by default; use `--date YYYY-MM-DD` to audit another available snapshot.

Each scenario reports portfolio return as `sum(weight * asset return)`, long- and short-side contributions, weighted absolute shock, per-asset contributions, and the largest positive and negative contributors. The summary identifies the best and worst supplied scenarios. `--max-loss` returns exit code `1` with a stable failure for every scenario whose negative portfolio return exceeds the limit; malformed or incomplete inputs return `2`. Output files cannot alias either source CSV.

These are instantaneous linear shocks to static weights. The audit does not model nonlinear instruments, changing exposures, liquidity, price impact, correlation dynamics, currency, financing, margin, taxes, or the probability of a scenario. A passing synthetic scenario set is not evidence that a portfolio is safe, profitable, or robust to untested events.

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

## VaR forecast backtesting

A historical tail describes realized data; a VaR backtest asks whether previously produced one-period loss thresholds had the advertised unconditional exception rate. Supply a strictly increasing CSV with `date,realized_return,var`, where `var` is a non-negative fractional loss magnitude and an exception occurs only when `-realized_return > var`:

```powershell
python -m quant_research_micro_lab.var_backtest examples/var-forecasts.csv `
  --confidence 0.80 --max-exception-rate 0.25 `
  --min-kupiec-p-value 0.05 --max-exception-count 2
```

The report includes observed and expected exception rates and counts, mean and maximum exception loss and threshold shortfall, the longest exception streak, adjacent exception pairs, and the Kupiec likelihood-ratio unconditional-coverage test. Exception details are date-bounded with `--max-details`; `--output` writes JSON without allowing the source CSV to be overwritten. Exit code `0` means all configured gates passed, `1` reports coverage or count failures, and `2` identifies invalid data or configuration.

The asymptotic Kupiec p-value can be weak or unstable for small samples and tests only unconditional coverage. It does not establish exception independence, correct tail severity, forecast calibration under regime change, or future risk control; the streak diagnostics are descriptive rather than an independence test. Passing this audit is not a profitability, capital-adequacy, safety, or trading guarantee, and the tool does not estimate VaR or execute transactions.

## Rolling performance diagnostics

Aggregate statistics can hide when a strategy's behavior changed. Measure every overlapping fixed-length window in a strict backtest equity export:

```powershell
python -m quant_research_micro_lab.rolling examples/rolling-equity.csv `
  --window 3 --periods-per-year 252
```

`--window` is the number of consecutive returns, so each window contains one additional equity observation. Every dated window reports total and annualized return, annualized volatility, zero-rate Sharpe ratio, zero-target downside deviation, maximum drawdown, positive-return rate, and its worst individual return. The summary identifies the latest window plus the earliest worst-return, worst-drawdown, and highest-volatility windows under deterministic ties. Use `--output` to save the JSON report or `--column gross_equity` to inspect the pre-cost curve.

Overlapping windows are strongly dependent observations, and annualization assumes the supplied periods have a stable frequency matching `--periods-per-year`. A zero-volatility window reports Sharpe as `null` rather than inventing a ratio. These diagnostics describe the supplied curve and do not estimate trading capacity, future returns, or loss probabilities.

## Return dependence diagnostics

IID assumptions can be misleading when adjacent strategy returns are serially related. Inspect the simple returns implied by a dated equity export at every lag from one through a chosen maximum:

```powershell
python -m quant_research_micro_lab.dependence `
  examples/dependence-equity.csv `
  --max-lag 5 `
  --max-abs-autocorrelation 0.80
```

The report includes mean and standard deviation, positive and negative return rates, each lag's standard centered sample autocorrelation, the largest absolute autocorrelation, and the Ljung-Box portmanteau statistic through the selected lag. The CLI accepts the same strict `date,equity,gross_equity` export as other equity diagnostics and can analyze either curve. Exit code `0` means the optional maximum absolute-autocorrelation gate passed, `1` reports a structured excess, and `2` identifies invalid data, zero-variance returns, an infeasible lag, unsafe output aliasing, or invalid configuration.

This is a descriptive historical diagnostic, not a market-efficiency test, forecast, profitability claim, or trading signal. No Ljung-Box p-value is reported because finite-sample validity depends on assumptions and model choices outside this small tool. Nonstationarity, volatility clustering, overlapping construction, multiple lag inspection, and parameter selection on the same sample can all distort interpretation; use a predeclared lag horizon and stronger inference when decisions depend on statistical significance.

## Correlation drift diagnostics

Portfolio assumptions can change when asset relationships differ across historical windows. Compare strict wide return files with the same asset columns:

```powershell
python -m quant_research_micro_lab.correlation_drift `
  examples/correlation-baseline.csv `
  examples/correlation-candidate.csv `
  --max-abs-correlation-change 1.99 `
  --max-rms-correlation-change 1.25 `
  --max-sign-flips 3
```

Each CSV must begin with `date` followed by at least two unique asset columns, and each window needs at least three strictly ordered observations. The dates do not need to align across windows. The report compares every asset pair using sample Pearson correlation, then summarizes mean absolute change and root-mean-square change. It also reports the largest changed pair and the number of strict sign flips. Pair details are sorted by absolute change and bounded with `--max-details`. A constant asset fails closed because its correlation is undefined.

Exit code `0` means every configured maximum passed, `1` reports structured drift failures, and `2` identifies malformed returns, inconsistent asset schemas, zero-variance series, unsafe path aliasing, or invalid configuration. This is a descriptive comparison of two finite samples. Correlations can move because of sampling noise, window selection, outliers, nonstationarity, or regime changes, and overlapping windows make the comparison dependent. The report is not a significance test, covariance forecast, diversification guarantee, trading signal, or advice.

## Moving-block bootstrap diagnostics

Resample contiguous return blocks to see how path-dependent metrics vary under alternative orderings of the supplied historical returns:

```powershell
python -m quant_research_micro_lab.bootstrap examples/bootstrap-equity.csv `
  --block-size 3 --samples 2000 --confidence 0.95 `
  --seed 7 --periods-per-year 252
```

Each bootstrap path contains the same number of returns as the input. Blocks are drawn with replacement from every valid overlapping start and the final block is truncated to the required path length. The deterministic report includes the observed and bootstrap-mean total return, annualized volatility, and maximum drawdown; linearly interpolated percentile intervals; and the share of sampled paths with a negative total return. Use `--column gross_equity` for the pre-cost curve or `--output` to write JSON.

A block preserves dependence only within its selected span, while drawing blocks assumes that historical return behavior is sufficiently stable to recombine. Results can change materially with block size, sample period, seed, and the input strategy. These intervals summarize a resampling procedure on one observed curve; they are not forecast intervals, trading recommendations, or guarantees about future returns or losses.

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
