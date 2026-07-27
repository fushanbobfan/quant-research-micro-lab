"""Audit dated portfolio weights for exposure and concentration risk."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import date
from numbers import Real
from pathlib import Path
from typing import Any


def _validate_limit(
    name: str,
    value: float | None,
    *,
    unit_interval: bool = False,
) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value < 0
        or (unit_interval and value > 1)
    ):
        suffix = " between 0 and 1" if unit_interval else " non-negative"
        raise ValueError(f"{name} must be a finite{suffix} number")
    return float(value)


def _validate_records(
    records: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str, float]]:
    if not records:
        raise ValueError("at least one position record is required")

    validated = []
    seen: set[tuple[str, str]] = set()
    previous_date: str | None = None
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"record {index} must be an object")
        date_value = record.get("date")
        asset = record.get("asset")
        weight = record.get("weight")
        if not isinstance(date_value, str):
            raise ValueError(f"record {index} date must be an ISO date")
        try:
            parsed_date = date.fromisoformat(date_value)
        except ValueError as error:
            raise ValueError(f"record {index} date must be an ISO date") from error
        if parsed_date.isoformat() != date_value:
            raise ValueError(f"record {index} date must use YYYY-MM-DD")
        if previous_date is not None and date_value < previous_date:
            raise ValueError("position dates must be non-decreasing")
        if not isinstance(asset, str) or not asset.strip():
            raise ValueError(f"record {index} asset must be a non-empty string")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, Real)
            or not math.isfinite(weight)
        ):
            raise ValueError(f"record {index} weight must be a finite number")
        key = (date_value, asset)
        if key in seen:
            raise ValueError(
                f"duplicate position for date {date_value} and asset {asset}"
            )
        seen.add(key)
        validated.append((date_value, asset, float(weight)))
        previous_date = date_value
    return validated


def audit_portfolio_exposure(
    records: Sequence[Mapping[str, Any]],
    *,
    max_gross_exposure: float | None = None,
    max_abs_net_exposure: float | None = None,
    max_single_position: float | None = None,
    max_concentration_hhi: float | None = None,
) -> dict[str, Any]:
    """Return dated portfolio exposure, concentration, and turnover diagnostics."""

    thresholds = {
        "gross_exposure": _validate_limit(
            "max_gross_exposure", max_gross_exposure
        ),
        "abs_net_exposure": _validate_limit(
            "max_abs_net_exposure", max_abs_net_exposure
        ),
        "single_position": _validate_limit(
            "max_single_position", max_single_position
        ),
        "concentration_hhi": _validate_limit(
            "max_concentration_hhi",
            max_concentration_hhi,
            unit_interval=True,
        ),
    }
    validated = _validate_records(records)

    weights_by_date: dict[str, dict[str, float]] = {}
    for date_value, asset, weight in validated:
        weights_by_date.setdefault(date_value, {})[asset] = weight

    snapshots = []
    previous_weights: dict[str, float] | None = None
    for date_value, weights in weights_by_date.items():
        gross_exposure = sum(abs(weight) for weight in weights.values())
        if gross_exposure == 0:
            raise ValueError(f"portfolio on {date_value} must have non-zero exposure")
        long_exposure = sum(weight for weight in weights.values() if weight > 0)
        short_exposure = -sum(weight for weight in weights.values() if weight < 0)
        net_exposure = sum(weights.values())
        largest_asset, largest_weight = min(
            weights.items(),
            key=lambda item: (-abs(item[1]), item[0]),
        )
        concentration_hhi = sum(
            (abs(weight) / gross_exposure) ** 2 for weight in weights.values()
        )
        turnover = None
        if previous_weights is not None:
            assets = set(previous_weights) | set(weights)
            turnover = 0.5 * sum(
                abs(weights.get(asset, 0.0) - previous_weights.get(asset, 0.0))
                for asset in assets
            )
        snapshots.append(
            {
                "date": date_value,
                "asset_count": len(weights),
                "nonzero_positions": sum(weight != 0 for weight in weights.values()),
                "long_exposure": long_exposure,
                "short_exposure": short_exposure,
                "gross_exposure": gross_exposure,
                "net_exposure": net_exposure,
                "abs_net_exposure": abs(net_exposure),
                "largest_position_asset": largest_asset,
                "largest_position_weight": largest_weight,
                "largest_abs_position": abs(largest_weight),
                "largest_abs_share": abs(largest_weight) / gross_exposure,
                "concentration_hhi": concentration_hhi,
                "effective_positions": 1.0 / concentration_hhi,
                "turnover_from_previous": turnover,
            }
        )
        previous_weights = weights

    extrema = {
        "maximum_gross_exposure": max(
            snapshots, key=lambda item: item["gross_exposure"]
        ),
        "maximum_abs_net_exposure": max(
            snapshots, key=lambda item: item["abs_net_exposure"]
        ),
        "maximum_single_position": max(
            snapshots, key=lambda item: item["largest_abs_position"]
        ),
        "maximum_concentration": max(
            snapshots, key=lambda item: item["concentration_hhi"]
        ),
    }
    turnover_snapshots = [
        snapshot
        for snapshot in snapshots
        if snapshot["turnover_from_previous"] is not None
    ]
    maximum_turnover = (
        max(turnover_snapshots, key=lambda item: item["turnover_from_previous"])
        if turnover_snapshots
        else None
    )

    failures = []
    failure_specs = (
        (
            "gross_exposure",
            extrema["maximum_gross_exposure"],
            "gross_exposure",
        ),
        (
            "abs_net_exposure",
            extrema["maximum_abs_net_exposure"],
            "abs_net_exposure",
        ),
        (
            "single_position",
            extrema["maximum_single_position"],
            "largest_abs_position",
        ),
        (
            "concentration_hhi",
            extrema["maximum_concentration"],
            "concentration_hhi",
        ),
    )
    for metric, snapshot, field in failure_specs:
        maximum = thresholds[metric]
        actual = snapshot[field]
        if maximum is not None and actual > maximum:
            failures.append(
                {
                    "metric": metric,
                    "date": snapshot["date"],
                    "actual": actual,
                    "maximum": maximum,
                    "excess": actual - maximum,
                }
            )

    return {
        "passed": not failures,
        "start_date": snapshots[0]["date"],
        "end_date": snapshots[-1]["date"],
        "snapshot_count": len(snapshots),
        "asset_count": len({asset for _, asset, _ in validated}),
        "summary": {
            "average_gross_exposure": sum(
                snapshot["gross_exposure"] for snapshot in snapshots
            )
            / len(snapshots),
            "average_abs_net_exposure": sum(
                snapshot["abs_net_exposure"] for snapshot in snapshots
            )
            / len(snapshots),
            "average_concentration_hhi": sum(
                snapshot["concentration_hhi"] for snapshot in snapshots
            )
            / len(snapshots),
            "average_effective_positions": sum(
                snapshot["effective_positions"] for snapshot in snapshots
            )
            / len(snapshots),
            "average_turnover": (
                sum(
                    snapshot["turnover_from_previous"]
                    for snapshot in turnover_snapshots
                )
                / len(turnover_snapshots)
                if turnover_snapshots
                else None
            ),
        },
        "extrema": {
            "maximum_gross_exposure": {
                "date": extrema["maximum_gross_exposure"]["date"],
                "value": extrema["maximum_gross_exposure"]["gross_exposure"],
            },
            "maximum_abs_net_exposure": {
                "date": extrema["maximum_abs_net_exposure"]["date"],
                "value": extrema["maximum_abs_net_exposure"]["abs_net_exposure"],
            },
            "maximum_single_position": {
                "date": extrema["maximum_single_position"]["date"],
                "asset": extrema["maximum_single_position"][
                    "largest_position_asset"
                ],
                "weight": extrema["maximum_single_position"][
                    "largest_position_weight"
                ],
                "absolute_weight": extrema["maximum_single_position"][
                    "largest_abs_position"
                ],
            },
            "maximum_concentration_hhi": {
                "date": extrema["maximum_concentration"]["date"],
                "value": extrema["maximum_concentration"]["concentration_hhi"],
            },
            "maximum_turnover": (
                {
                    "date": maximum_turnover["date"],
                    "value": maximum_turnover["turnover_from_previous"],
                }
                if maximum_turnover
                else None
            ),
        },
        "thresholds": thresholds,
        "failures": failures,
        "snapshots": snapshots,
    }


def load_portfolio_csv(path: Path) -> list[dict[str, Any]]:
    """Load the strict date,asset,weight position format."""

    records = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["date", "asset", "weight"]:
            raise ValueError("CSV header must be exactly: date,asset,weight")
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"row {row_number} must contain exactly three fields")
            try:
                weight = float(row.get("weight") or "")
            except ValueError as error:
                raise ValueError(f"row {row_number} has an invalid weight") from error
            records.append(
                {
                    "date": row.get("date"),
                    "asset": row.get("asset"),
                    "weight": weight,
                }
            )
    if not records:
        raise ValueError("CSV must contain at least one position row")
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--max-gross-exposure", type=float)
    parser.add_argument("--max-abs-net-exposure", type=float)
    parser.add_argument("--max-single-position", type=float)
    parser.add_argument("--max-concentration-hhi", type=float)
    args = parser.parse_args(argv)

    try:
        report = audit_portfolio_exposure(
            load_portfolio_csv(args.dataset),
            max_gross_exposure=args.max_gross_exposure,
            max_abs_net_exposure=args.max_abs_net_exposure,
            max_single_position=args.max_single_position,
            max_concentration_hhi=args.max_concentration_hhi,
        )
    except (OSError, UnicodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(json.dumps(report, indent=2))
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
