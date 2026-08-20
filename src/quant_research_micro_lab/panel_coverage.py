"""Audit missing-cell coverage in a dated wide numeric panel."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import date
from numbers import Real
from pathlib import Path
from typing import Any


def _validate_rate(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{name} must be a finite number between 0 and 1")
    return float(value)


def _validate_non_negative_integer(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def audit_panel_coverage(
    records: Sequence[Mapping[str, Any]],
    *,
    min_overall_coverage: float = 0.0,
    min_asset_coverage: float = 0.0,
    max_missing_streak: int | None = None,
    max_incomplete_row_rate: float = 1.0,
    max_details: int = 20,
) -> dict[str, Any]:
    """Return panel presence metrics, missing streaks, and optional gates."""

    if not records:
        raise ValueError("at least one panel record is required")
    thresholds = {
        "overall_coverage": _validate_rate(
            "min_overall_coverage", min_overall_coverage
        ),
        "minimum_asset_coverage": _validate_rate(
            "min_asset_coverage", min_asset_coverage
        ),
        "longest_missing_streak": _validate_non_negative_integer(
            "max_missing_streak", max_missing_streak
        ),
        "incomplete_row_rate": _validate_rate(
            "max_incomplete_row_rate", max_incomplete_row_rate
        ),
    }
    if (
        isinstance(max_details, bool)
        or not isinstance(max_details, int)
        or max_details < 0
    ):
        raise ValueError("max_details must be a non-negative integer")

    first_record = records[0]
    if not isinstance(first_record, Mapping):
        raise ValueError("panel record 0 must be an object")
    assets = [field for field in first_record if field != "date"]
    if not assets:
        raise ValueError("panel records must contain at least one asset column")
    if any(not isinstance(asset, str) or not asset for asset in assets):
        raise ValueError("asset column names must be non-empty strings")
    expected_fields = {"date", *assets}

    validated = []
    previous_date: str | None = None
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"panel record {index} must be an object")
        if set(record) != expected_fields:
            raise ValueError("panel records must have the same asset columns")
        date_value = record.get("date")
        if not isinstance(date_value, str):
            raise ValueError(f"panel record {index} date must be an ISO date")
        try:
            parsed_date = date.fromisoformat(date_value)
        except ValueError as error:
            raise ValueError(f"panel record {index} date must be an ISO date") from error
        if parsed_date.isoformat() != date_value:
            raise ValueError(f"panel record {index} date must use YYYY-MM-DD")
        if previous_date is not None and date_value <= previous_date:
            raise ValueError("panel dates must be strictly increasing")

        values: dict[str, float | None] = {}
        for asset in assets:
            value = record.get(asset)
            if value is None:
                values[asset] = None
            elif (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
            ):
                raise ValueError(
                    f"panel record {index} {asset} must be finite or missing"
                )
            else:
                values[asset] = float(value)
        validated.append((date_value, values))
        previous_date = date_value

    all_streaks = []
    asset_reports = []
    for asset in assets:
        missing_dates = [
            date_value
            for date_value, values in validated
            if values[asset] is None
        ]
        streaks = []
        run_start: str | None = None
        run_end: str | None = None
        run_length = 0
        for date_value, values in validated:
            if values[asset] is None:
                if run_start is None:
                    run_start = date_value
                run_end = date_value
                run_length += 1
            elif run_start is not None:
                streaks.append(
                    {
                        "asset": asset,
                        "start_date": run_start,
                        "end_date": run_end,
                        "observations": run_length,
                    }
                )
                run_start = None
                run_end = None
                run_length = 0
        if run_start is not None:
            streaks.append(
                {
                    "asset": asset,
                    "start_date": run_start,
                    "end_date": run_end,
                    "observations": run_length,
                }
            )
        all_streaks.extend(streaks)
        longest = min(
            streaks,
            key=lambda item: (-item["observations"], item["start_date"]),
            default=None,
        )
        observed_count = len(validated) - len(missing_dates)
        asset_reports.append(
            {
                "asset": asset,
                "observed_count": observed_count,
                "missing_count": len(missing_dates),
                "coverage": observed_count / len(validated),
                "missing_streak_count": len(streaks),
                "longest_missing_streak": longest,
            }
        )

    incomplete_rows = []
    for date_value, values in validated:
        missing_assets = [asset for asset in assets if values[asset] is None]
        if missing_assets:
            incomplete_rows.append(
                {
                    "date": date_value,
                    "missing_count": len(missing_assets),
                    "coverage": (len(assets) - len(missing_assets)) / len(assets),
                    "missing_assets": missing_assets,
                }
            )
    incomplete_rows.sort(key=lambda item: (-item["missing_count"], item["date"]))
    all_streaks.sort(
        key=lambda item: (
            -item["observations"],
            item["asset"],
            item["start_date"],
        )
    )
    longest_streak = all_streaks[0] if all_streaks else None
    minimum_asset = min(asset_reports, key=lambda item: (item["coverage"], item["asset"]))
    expected_cells = len(validated) * len(assets)
    missing_cells = sum(item["missing_count"] for item in asset_reports)
    observed_cells = expected_cells - missing_cells
    overall_coverage = observed_cells / expected_cells
    incomplete_row_rate = len(incomplete_rows) / len(validated)

    metrics = {
        "start_date": validated[0][0],
        "end_date": validated[-1][0],
        "observation_count": len(validated),
        "asset_count": len(assets),
        "expected_cells": expected_cells,
        "observed_cells": observed_cells,
        "missing_cells": missing_cells,
        "overall_coverage": overall_coverage,
        "complete_row_count": len(validated) - len(incomplete_rows),
        "incomplete_row_count": len(incomplete_rows),
        "incomplete_row_rate": incomplete_row_rate,
        "minimum_asset_coverage": minimum_asset["coverage"],
        "minimum_coverage_asset": minimum_asset["asset"],
        "longest_missing_streak": longest_streak,
    }

    failures = []
    if overall_coverage < thresholds["overall_coverage"]:
        failures.append(
            {
                "metric": "overall_coverage",
                "actual": overall_coverage,
                "minimum": thresholds["overall_coverage"],
                "shortfall": thresholds["overall_coverage"] - overall_coverage,
            }
        )
    if minimum_asset["coverage"] < thresholds["minimum_asset_coverage"]:
        failures.append(
            {
                "metric": "minimum_asset_coverage",
                "asset": minimum_asset["asset"],
                "actual": minimum_asset["coverage"],
                "minimum": thresholds["minimum_asset_coverage"],
                "shortfall": thresholds["minimum_asset_coverage"]
                - minimum_asset["coverage"],
            }
        )
    maximum_streak = thresholds["longest_missing_streak"]
    actual_streak = longest_streak["observations"] if longest_streak else 0
    if maximum_streak is not None and actual_streak > maximum_streak:
        failures.append(
            {
                "metric": "longest_missing_streak",
                "asset": longest_streak["asset"],
                "actual": actual_streak,
                "maximum": maximum_streak,
                "excess": actual_streak - maximum_streak,
            }
        )
    if incomplete_row_rate > thresholds["incomplete_row_rate"]:
        failures.append(
            {
                "metric": "incomplete_row_rate",
                "actual": incomplete_row_rate,
                "maximum": thresholds["incomplete_row_rate"],
                "excess": incomplete_row_rate
                - thresholds["incomplete_row_rate"],
            }
        )

    return {
        "passed": not failures,
        "metrics": metrics,
        "thresholds": thresholds,
        "failures": failures,
        "assets": asset_reports,
        "incomplete_rows": incomplete_rows[:max_details],
        "details_truncated": len(incomplete_rows) > max_details,
        "omitted_incomplete_row_count": max(
            0, len(incomplete_rows) - max_details
        ),
        "settings": {"max_details": max_details},
    }


def load_panel_csv(
    path: Path, *, max_file_bytes: int = 10 * 1024 * 1024
) -> list[dict[str, Any]]:
    """Load a bounded wide CSV, mapping blank asset cells to missing values."""

    if (
        isinstance(max_file_bytes, bool)
        or not isinstance(max_file_bytes, int)
        or max_file_bytes <= 0
    ):
        raise ValueError("max_file_bytes must be a positive integer")
    with path.open("rb") as handle:
        data = handle.read(max_file_bytes + 1)
    if len(data) > max_file_bytes:
        raise ValueError(f"panel CSV exceeds max_file_bytes ({max_file_bytes})")

    text = data.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    fieldnames = reader.fieldnames
    if (
        fieldnames is None
        or len(fieldnames) < 2
        or fieldnames[0] != "date"
        or any(not field or field.strip() != field for field in fieldnames)
        or len(set(fieldnames)) != len(fieldnames)
    ):
        raise ValueError(
            "panel CSV header must be date followed by unique asset columns"
        )

    records = []
    for row_number, row in enumerate(reader, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"panel row {row_number} must match the header field count")
        record: dict[str, Any] = {"date": row["date"]}
        for asset in fieldnames[1:]:
            raw_value = row[asset]
            if raw_value == "":
                record[asset] = None
            else:
                try:
                    record[asset] = float(raw_value)
                except ValueError as error:
                    raise ValueError(
                        f"panel row {row_number} has an invalid {asset} value"
                    ) from error
        records.append(record)
    if not records:
        raise ValueError("panel CSV must contain at least one row")
    return records


def _paths_alias(source: Path, output: Path) -> bool:
    if source.resolve() == output.resolve():
        return True
    try:
        return source.samefile(output)
    except (FileNotFoundError, OSError):
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("--min-overall-coverage", type=float, default=0.0)
    parser.add_argument("--min-asset-coverage", type=float, default=0.0)
    parser.add_argument("--max-missing-streak", type=int)
    parser.add_argument("--max-incomplete-row-rate", type=float, default=1.0)
    parser.add_argument("--max-details", type=int, default=20)
    parser.add_argument("--max-file-bytes", type=int, default=10 * 1024 * 1024)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        if args.output is not None and _paths_alias(args.panel, args.output):
            raise ValueError("output must not alias the input CSV")
        report = audit_panel_coverage(
            load_panel_csv(args.panel, max_file_bytes=args.max_file_bytes),
            min_overall_coverage=args.min_overall_coverage,
            min_asset_coverage=args.min_asset_coverage,
            max_missing_streak=args.max_missing_streak,
            max_incomplete_row_rate=args.max_incomplete_row_rate,
            max_details=args.max_details,
        )
        rendered = json.dumps(report, indent=2) + "\n"
        if args.output is None:
            print(rendered, end="")
        else:
            args.output.write_text(rendered, encoding="utf-8")
    except (OSError, UnicodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
