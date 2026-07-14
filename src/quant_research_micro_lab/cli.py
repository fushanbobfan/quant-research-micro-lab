"""Run the crossover backtest on a validated date-and-close CSV file."""

from __future__ import annotations

import csv
import math
from datetime import date
from pathlib import Path


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
