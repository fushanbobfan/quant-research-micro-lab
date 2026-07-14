"""Run the crossover backtest on a validated date-and-close CSV file."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from datetime import date
from pathlib import Path

from .backtest import backtest_crossover


def load_price_csv(path: Path) -> tuple[list[str], list[float]]:
    """Load strictly increasing ISO dates and finite positive closes."""

    dates: list[str] = []
    prices: list[float] = []
    previous_date: date | None = None
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["date", "close"]:
            raise ValueError("CSV header must be exactly: date,close")

        for row_number, row in enumerate(reader, start=2):
            raw_date = row.get("date")
            raw_close = row.get("close")
            try:
                parsed_date = date.fromisoformat(raw_date or "")
            except ValueError as error:
                raise ValueError(f"row {row_number} has an invalid ISO date") from error
            if previous_date is not None and parsed_date <= previous_date:
                raise ValueError(f"row {row_number} date must be strictly increasing")
            try:
                close = float(raw_close or "")
            except ValueError as error:
                raise ValueError(f"row {row_number} has an invalid close") from error
            if not math.isfinite(close) or close <= 0:
                raise ValueError(f"row {row_number} close must be finite and positive")

            dates.append(parsed_date.isoformat())
            prices.append(close)
            previous_date = parsed_date

    if not dates:
        raise ValueError("CSV must contain at least one price row")
    return dates, prices


def _write_equity_csv(
    path: Path,
    dates: list[str],
    equity: list[float],
    gross_equity: list[float],
) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["date", "equity", "gross_equity"])
    writer.writerows(zip(dates, equity, gross_equity))
    path.write_text(buffer.getvalue(), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--short-window", type=int, default=5)
    parser.add_argument("--long-window", type=int, default=20)
    parser.add_argument("--transaction-cost-bps", type=float, default=0.0)
    parser.add_argument(
        "--equity-output",
        type=Path,
        help="optionally write dated net and gross equity to CSV",
    )
    args = parser.parse_args(argv)

    try:
        dates, prices = load_price_csv(args.dataset)
        result = backtest_crossover(
            prices,
            short_window=args.short_window,
            long_window=args.long_window,
            transaction_cost_bps=args.transaction_cost_bps,
        )
        if args.equity_output:
            _write_equity_csv(
                args.equity_output,
                dates,
                result["equity"],
                result["gross_equity"],
            )
    except (OSError, UnicodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    report = {
        "observations": len(dates),
        "start_date": dates[0],
        "end_date": dates[-1],
        **result,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
