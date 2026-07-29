"""Evaluate explicit asset-return shocks against a portfolio snapshot."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

from .exposure import _validate_records, load_portfolio_csv


def _validate_scenarios(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    if not records:
        raise ValueError("at least one scenario record is required")

    scenarios: dict[str, dict[str, float]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"scenario record {index} must be an object")
        scenario = record.get("scenario")
        asset = record.get("asset")
        asset_return = record.get("return")
        if not isinstance(scenario, str) or not scenario.strip():
            raise ValueError(
                f"scenario record {index} scenario must be a non-empty string"
            )
        if not isinstance(asset, str) or not asset.strip():
            raise ValueError(
                f"scenario record {index} asset must be a non-empty string"
            )
        if (
            isinstance(asset_return, bool)
            or not isinstance(asset_return, Real)
            or not math.isfinite(asset_return)
            or asset_return < -1.0
        ):
            raise ValueError(
                f"scenario record {index} return must be finite and at least -1"
            )
        shocks = scenarios.setdefault(scenario, {})
        if asset in shocks:
            raise ValueError(
                f"duplicate scenario shock for {scenario!r} and asset {asset!r}"
            )
        shocks[asset] = float(asset_return)
    return scenarios


def evaluate_portfolio_stress(
    position_records: Sequence[Mapping[str, Any]],
    scenario_records: Sequence[Mapping[str, Any]],
    *,
    snapshot_date: str | None = None,
    max_loss: float | None = None,
) -> dict[str, Any]:
    """Return deterministic contribution and loss diagnostics for each scenario."""

    if max_loss is not None and (
        isinstance(max_loss, bool)
        or not isinstance(max_loss, Real)
        or not math.isfinite(max_loss)
        or max_loss < 0.0
    ):
        raise ValueError("max_loss must be a finite non-negative number")

    positions = _validate_records(position_records)
    available_dates = sorted({date_value for date_value, _, _ in positions})
    selected_date = available_dates[-1] if snapshot_date is None else snapshot_date
    if not isinstance(selected_date, str) or selected_date not in available_dates:
        raise ValueError("snapshot_date must match an available portfolio date")

    weights = {
        asset: weight
        for date_value, asset, weight in positions
        if date_value == selected_date and weight != 0.0
    }
    if not weights:
        raise ValueError("selected portfolio snapshot must have non-zero exposure")

    scenarios = _validate_scenarios(scenario_records)
    portfolio_assets = set(weights)
    for scenario, shocks in scenarios.items():
        scenario_assets = set(shocks)
        if scenario_assets != portfolio_assets:
            missing = sorted(portfolio_assets - scenario_assets)
            unexpected = sorted(scenario_assets - portfolio_assets)
            raise ValueError(
                f"scenario {scenario!r} assets must match the selected portfolio; "
                f"missing={missing}, unexpected={unexpected}"
            )

    scenario_reports = []
    for scenario in sorted(scenarios):
        shocks = scenarios[scenario]
        assets = []
        for asset in sorted(weights):
            weight = weights[asset]
            asset_return = shocks[asset]
            contribution = weight * asset_return
            assets.append(
                {
                    "asset": asset,
                    "weight": weight,
                    "return": asset_return,
                    "contribution": contribution,
                }
            )

        portfolio_return = sum(item["contribution"] for item in assets)
        negative = [item for item in assets if item["contribution"] < 0.0]
        positive = [item for item in assets if item["contribution"] > 0.0]
        scenario_reports.append(
            {
                "scenario": scenario,
                "portfolio_return": portfolio_return,
                "loss": max(0.0, -portfolio_return),
                "long_contribution": sum(
                    item["contribution"] for item in assets if item["weight"] > 0
                ),
                "short_contribution": sum(
                    item["contribution"] for item in assets if item["weight"] < 0
                ),
                "weighted_absolute_shock": sum(
                    abs(item["contribution"]) for item in assets
                ),
                "largest_negative_contributor": (
                    min(negative, key=lambda item: (item["contribution"], item["asset"]))
                    if negative
                    else None
                ),
                "largest_positive_contributor": (
                    max(positive, key=lambda item: (item["contribution"], item["asset"]))
                    if positive
                    else None
                ),
                "assets": assets,
            }
        )

    worst = min(
        scenario_reports,
        key=lambda item: (item["portfolio_return"], item["scenario"]),
    )
    best = max(
        scenario_reports,
        key=lambda item: (item["portfolio_return"], item["scenario"]),
    )
    failures = []
    if max_loss is not None:
        for report in scenario_reports:
            if report["loss"] > max_loss:
                failures.append(
                    {
                        "scenario": report["scenario"],
                        "metric": "loss",
                        "actual": report["loss"],
                        "maximum": float(max_loss),
                        "excess": report["loss"] - float(max_loss),
                    }
                )

    gross_exposure = sum(abs(weight) for weight in weights.values())
    return {
        "passed": not failures,
        "portfolio": {
            "date": selected_date,
            "asset_count": len(weights),
            "long_exposure": sum(weight for weight in weights.values() if weight > 0),
            "short_exposure": -sum(
                weight for weight in weights.values() if weight < 0
            ),
            "gross_exposure": gross_exposure,
            "net_exposure": sum(weights.values()),
        },
        "scenario_count": len(scenario_reports),
        "summary": {
            "worst_scenario": {
                "scenario": worst["scenario"],
                "portfolio_return": worst["portfolio_return"],
                "loss": worst["loss"],
            },
            "best_scenario": {
                "scenario": best["scenario"],
                "portfolio_return": best["portfolio_return"],
                "loss": best["loss"],
            },
            "average_portfolio_return": sum(
                report["portfolio_return"] for report in scenario_reports
            )
            / len(scenario_reports),
        },
        "thresholds": {"max_loss": float(max_loss) if max_loss is not None else None},
        "failures": failures,
        "scenarios": scenario_reports,
    }


def load_scenario_csv(path: Path) -> list[dict[str, Any]]:
    """Load the strict scenario,asset,return stress format."""

    records = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["scenario", "asset", "return"]:
            raise ValueError("CSV header must be exactly: scenario,asset,return")
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"row {row_number} must contain exactly three fields")
            try:
                asset_return = float(row.get("return") or "")
            except ValueError as error:
                raise ValueError(f"row {row_number} has an invalid return") from error
            records.append(
                {
                    "scenario": row.get("scenario"),
                    "asset": row.get("asset"),
                    "return": asset_return,
                }
            )
    if not records:
        raise ValueError("CSV must contain at least one scenario row")
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("portfolio", type=Path)
    parser.add_argument("scenarios", type=Path)
    parser.add_argument("--date")
    parser.add_argument("--max-loss", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        report = evaluate_portfolio_stress(
            load_portfolio_csv(args.portfolio),
            load_scenario_csv(args.scenarios),
            snapshot_date=args.date,
            max_loss=args.max_loss,
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
