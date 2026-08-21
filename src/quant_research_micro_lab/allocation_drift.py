"""Audit drift between dated target and actual portfolio weights."""

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


def _non_negative(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative finite number")
    return float(value)


def _rate(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{name} must be a finite number between 0 and 1")
    return float(value)


def _weight(value: Any, *, field: str, record_index: int) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
    ):
        raise ValueError(f"record {record_index} {field} must be finite")
    return float(value)


def _exceeds(actual: float, limit: float) -> bool:
    return actual > limit and not math.isclose(
        actual, limit, rel_tol=1e-12, abs_tol=1e-15
    )


def _below(actual: float, minimum: float) -> bool:
    return actual < minimum and not math.isclose(
        actual, minimum, rel_tol=1e-12, abs_tol=1e-15
    )


def audit_allocation_drift(
    records: Sequence[Mapping[str, Any]],
    *,
    asset_tolerance: float = 0.0,
    max_average_l1_drift: float | None = None,
    max_snapshot_l1_drift: float | None = None,
    max_asset_drift: float | None = None,
    min_within_tolerance_rate: float = 0.0,
    max_details: int = 20,
) -> dict[str, Any]:
    """Return paired allocation-drift metrics and optional gate failures."""
    if not records:
        raise ValueError("at least one allocation record is required")
    if (
        isinstance(max_details, bool)
        or not isinstance(max_details, int)
        or max_details < 0
    ):
        raise ValueError("max_details must be a non-negative integer")
    tolerance = _non_negative("asset_tolerance", asset_tolerance)
    assert tolerance is not None
    thresholds = {
        "average_l1_drift": _non_negative(
            "max_average_l1_drift", max_average_l1_drift
        ),
        "snapshot_l1_drift": _non_negative(
            "max_snapshot_l1_drift", max_snapshot_l1_drift
        ),
        "asset_drift": _non_negative("max_asset_drift", max_asset_drift),
        "within_tolerance_rate": _rate(
            "min_within_tolerance_rate", min_within_tolerance_rate
        ),
    }

    by_date: dict[str, list[dict[str, Any]]] = {}
    seen_pairs: set[tuple[str, str]] = set()
    previous_date: str | None = None
    all_assets: set[str] = set()
    expected_fields = {"date", "asset", "target_weight", "actual_weight"}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"record {index} must be an object")
        if set(record) != expected_fields:
            raise ValueError(f"record {index} must contain exactly {sorted(expected_fields)}")
        date_value = record.get("date")
        if not isinstance(date_value, str):
            raise ValueError(f"record {index} date must use YYYY-MM-DD")
        try:
            parsed_date = date.fromisoformat(date_value)
        except ValueError as error:
            raise ValueError(f"record {index} date must use YYYY-MM-DD") from error
        if parsed_date.isoformat() != date_value:
            raise ValueError(f"record {index} date must use YYYY-MM-DD")
        if previous_date is not None and date_value < previous_date:
            raise ValueError("allocation dates must be grouped in increasing order")
        asset = record.get("asset")
        if not isinstance(asset, str) or not asset:
            raise ValueError(f"record {index} asset must be a non-empty string")
        pair = (date_value, asset)
        if pair in seen_pairs:
            raise ValueError(f"record {index} repeats date and asset {pair!r}")
        seen_pairs.add(pair)
        target = _weight(
            record.get("target_weight"), field="target_weight", record_index=index
        )
        actual = _weight(
            record.get("actual_weight"), field="actual_weight", record_index=index
        )
        drift = actual - target
        by_date.setdefault(date_value, []).append(
            {
                "asset": asset,
                "target_weight": target,
                "actual_weight": actual,
                "drift": drift,
                "absolute_drift": abs(drift),
                "within_tolerance": not _exceeds(abs(drift), tolerance),
            }
        )
        all_assets.add(asset)
        previous_date = date_value

    snapshot_reports = []
    all_details = []
    for date_value, details in by_date.items():
        details.sort(key=lambda item: (-item["absolute_drift"], item["asset"]))
        l1_drift = sum(item["absolute_drift"] for item in details)
        within_count = sum(item["within_tolerance"] for item in details)
        snapshot_reports.append(
            {
                "date": date_value,
                "asset_count": len(details),
                "target_net_weight": sum(item["target_weight"] for item in details),
                "actual_net_weight": sum(item["actual_weight"] for item in details),
                "net_weight_drift": sum(item["drift"] for item in details),
                "l1_drift": l1_drift,
                "root_mean_square_drift": math.sqrt(
                    sum(item["drift"] ** 2 for item in details) / len(details)
                ),
                "within_tolerance_count": within_count,
                "within_tolerance_rate": within_count / len(details),
                "asset_details": details[:max_details],
                "asset_details_truncated": len(details) > max_details,
                "omitted_asset_count": max(0, len(details) - max_details),
            }
        )
        all_details.extend({"date": date_value, **item} for item in details)

    snapshot_reports.sort(key=lambda item: (-item["l1_drift"], item["date"]))
    all_details.sort(
        key=lambda item: (-item["absolute_drift"], item["date"], item["asset"])
    )
    snapshot_count = len(snapshot_reports)
    comparison_count = len(all_details)
    average_l1_drift = sum(item["l1_drift"] for item in snapshot_reports) / snapshot_count
    maximum_snapshot = snapshot_reports[0]
    maximum_asset = all_details[0]
    within_count = sum(item["within_tolerance"] for item in all_details)
    within_rate = within_count / comparison_count

    failures = []
    average_limit = thresholds["average_l1_drift"]
    if average_limit is not None and _exceeds(average_l1_drift, average_limit):
        failures.append(
            {
                "metric": "average_l1_drift",
                "actual": average_l1_drift,
                "maximum": average_limit,
                "excess": average_l1_drift - average_limit,
            }
        )
    snapshot_limit = thresholds["snapshot_l1_drift"]
    if snapshot_limit is not None and _exceeds(
        maximum_snapshot["l1_drift"], snapshot_limit
    ):
        failures.append(
            {
                "metric": "snapshot_l1_drift",
                "date": maximum_snapshot["date"],
                "actual": maximum_snapshot["l1_drift"],
                "maximum": snapshot_limit,
                "excess": maximum_snapshot["l1_drift"] - snapshot_limit,
            }
        )
    asset_limit = thresholds["asset_drift"]
    if asset_limit is not None and _exceeds(
        maximum_asset["absolute_drift"], asset_limit
    ):
        failures.append(
            {
                "metric": "asset_drift",
                "date": maximum_asset["date"],
                "asset": maximum_asset["asset"],
                "actual": maximum_asset["absolute_drift"],
                "maximum": asset_limit,
                "excess": maximum_asset["absolute_drift"] - asset_limit,
            }
        )
    minimum_rate = thresholds["within_tolerance_rate"]
    if _below(within_rate, minimum_rate):
        failures.append(
            {
                "metric": "within_tolerance_rate",
                "actual": within_rate,
                "minimum": minimum_rate,
                "shortfall": minimum_rate - within_rate,
            }
        )

    return {
        "passed": not failures,
        "start_date": min(by_date),
        "end_date": max(by_date),
        "snapshot_count": snapshot_count,
        "asset_count": len(all_assets),
        "position_comparison_count": comparison_count,
        "summary": {
            "average_l1_drift": average_l1_drift,
            "maximum_snapshot_l1_drift": {
                "date": maximum_snapshot["date"],
                "value": maximum_snapshot["l1_drift"],
            },
            "maximum_asset_drift": {
                "date": maximum_asset["date"],
                "asset": maximum_asset["asset"],
                "drift": maximum_asset["drift"],
                "absolute_drift": maximum_asset["absolute_drift"],
            },
            "within_tolerance_count": within_count,
            "within_tolerance_rate": within_rate,
        },
        "thresholds": {"asset_tolerance": tolerance, **thresholds},
        "failures": failures,
        "snapshot_details": snapshot_reports[:max_details],
        "details_truncated": snapshot_count > max_details,
        "omitted_snapshot_count": max(0, snapshot_count - max_details),
        "settings": {"max_details": max_details},
    }


def load_allocation_csv(
    path: Path, *, max_file_bytes: int = 10 * 1024 * 1024
) -> list[dict[str, Any]]:
    """Load a bounded long-form target/actual allocation CSV."""
    if (
        isinstance(max_file_bytes, bool)
        or not isinstance(max_file_bytes, int)
        or max_file_bytes <= 0
    ):
        raise ValueError("max_file_bytes must be a positive integer")
    with path.open("rb") as handle:
        data = handle.read(max_file_bytes + 1)
    if len(data) > max_file_bytes:
        raise ValueError(f"allocation CSV exceeds max_file_bytes ({max_file_bytes})")

    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""))
    expected = ["date", "asset", "target_weight", "actual_weight"]
    if reader.fieldnames != expected:
        raise ValueError(f"allocation CSV header must be {','.join(expected)}")
    records = []
    for row_number, row in enumerate(reader, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"allocation row {row_number} must match the header")
        try:
            target = float(row["target_weight"])
            actual = float(row["actual_weight"])
        except ValueError as error:
            raise ValueError(f"allocation row {row_number} has an invalid weight") from error
        records.append(
            {
                "date": row["date"],
                "asset": row["asset"],
                "target_weight": target,
                "actual_weight": actual,
            }
        )
    if not records:
        raise ValueError("allocation CSV must contain at least one row")
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
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--asset-tolerance", type=float, default=0.0)
    parser.add_argument("--max-average-l1-drift", type=float)
    parser.add_argument("--max-snapshot-l1-drift", type=float)
    parser.add_argument("--max-asset-drift", type=float)
    parser.add_argument("--min-within-tolerance-rate", type=float, default=0.0)
    parser.add_argument("--max-details", type=int, default=20)
    parser.add_argument("--max-file-bytes", type=int, default=10 * 1024 * 1024)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        if args.output is not None and _paths_alias(args.dataset, args.output):
            raise ValueError("output must not alias the input CSV")
        report = audit_allocation_drift(
            load_allocation_csv(args.dataset, max_file_bytes=args.max_file_bytes),
            asset_tolerance=args.asset_tolerance,
            max_average_l1_drift=args.max_average_l1_drift,
            max_snapshot_l1_drift=args.max_snapshot_l1_drift,
            max_asset_drift=args.max_asset_drift,
            min_within_tolerance_rate=args.min_within_tolerance_rate,
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
