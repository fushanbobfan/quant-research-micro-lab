"""Estimate historical covariance risk contributions for portfolio weights."""

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

from .exposure import _validate_records, load_portfolio_csv


def _validate_unit_limit(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{name} must be a finite number between 0 and 1")
    return float(value)


def _validate_return_records(
    records: Sequence[Mapping[str, Any]], assets: Sequence[str]
) -> list[tuple[str, list[float]]]:
    if len(records) < 2:
        raise ValueError("at least two return observations are required")

    expected_fields = {"date", *assets}
    validated = []
    seen_dates: set[str] = set()
    previous_date: str | None = None
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"return record {index} must be an object")
        if set(record) != expected_fields:
            missing = sorted(expected_fields - set(record))
            unexpected = sorted(set(record) - expected_fields)
            raise ValueError(
                f"return record {index} fields must match portfolio assets; "
                f"missing={missing}, unexpected={unexpected}"
            )
        date_value = record.get("date")
        if not isinstance(date_value, str):
            raise ValueError(f"return record {index} date must be an ISO date")
        try:
            parsed_date = date.fromisoformat(date_value)
        except ValueError as error:
            raise ValueError(
                f"return record {index} date must be an ISO date"
            ) from error
        if parsed_date.isoformat() != date_value:
            raise ValueError(f"return record {index} date must use YYYY-MM-DD")
        if date_value in seen_dates:
            raise ValueError(f"duplicate return date {date_value}")
        if previous_date is not None and date_value <= previous_date:
            raise ValueError("return dates must be strictly increasing")

        values = []
        for asset in assets:
            asset_return = record.get(asset)
            if (
                isinstance(asset_return, bool)
                or not isinstance(asset_return, Real)
                or not math.isfinite(asset_return)
                or asset_return < -1.0
            ):
                raise ValueError(
                    f"return record {index} {asset} must be finite and at least -1"
                )
            values.append(float(asset_return))
        validated.append((date_value, values))
        seen_dates.add(date_value)
        previous_date = date_value
    return validated


def analyze_risk_contributions(
    position_records: Sequence[Mapping[str, Any]],
    return_records: Sequence[Mapping[str, Any]],
    *,
    snapshot_date: str | None = None,
    periods_per_year: int = 252,
    max_largest_risk_share: float | None = None,
    max_risk_concentration_hhi: float | None = None,
) -> dict[str, Any]:
    """Return additive historical-volatility contributions and concentration gates."""

    if (
        isinstance(periods_per_year, bool)
        or not isinstance(periods_per_year, int)
        or periods_per_year <= 0
    ):
        raise ValueError("periods_per_year must be a positive integer")
    thresholds = {
        "largest_absolute_risk_share": _validate_unit_limit(
            "max_largest_risk_share", max_largest_risk_share
        ),
        "risk_concentration_hhi": _validate_unit_limit(
            "max_risk_concentration_hhi", max_risk_concentration_hhi
        ),
    }

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
    assets = sorted(weights)
    returns = _validate_return_records(return_records, assets)
    if returns[-1][0] > selected_date:
        raise ValueError("return observations must not extend beyond the snapshot date")

    observation_count = len(returns)
    means = [
        sum(values[index] for _, values in returns) / observation_count
        for index in range(len(assets))
    ]
    covariance = []
    for row_index in range(len(assets)):
        row = []
        for column_index in range(len(assets)):
            cross_product = sum(
                (values[row_index] - means[row_index])
                * (values[column_index] - means[column_index])
                for _, values in returns
            )
            row.append(cross_product / (observation_count - 1))
        covariance.append(row)

    weight_vector = [weights[asset] for asset in assets]
    covariance_weight = [
        sum(
            covariance[row][column] * weight_vector[column]
            for column in range(len(assets))
        )
        for row in range(len(assets))
    ]
    portfolio_variance = sum(
        weight_vector[index] * covariance_weight[index]
        for index in range(len(assets))
    )
    if portfolio_variance <= 0.0:
        raise ValueError("historical portfolio variance must be positive")
    portfolio_volatility = math.sqrt(portfolio_variance)

    components = [
        weight_vector[index] * covariance_weight[index] / portfolio_volatility
        for index in range(len(assets))
    ]
    gross_component = sum(abs(component) for component in components)
    if gross_component == 0.0:
        raise ValueError("historical component contributions must be non-zero")
    absolute_shares = [abs(component) / gross_component for component in components]
    risk_hhi = sum(share**2 for share in absolute_shares)
    largest_index = min(
        range(len(assets)), key=lambda index: (-absolute_shares[index], assets[index])
    )
    annualization = math.sqrt(periods_per_year)

    asset_contributions = []
    for index, asset in enumerate(assets):
        standalone_volatility = math.sqrt(max(0.0, covariance[index][index]))
        asset_contributions.append(
            {
                "asset": asset,
                "weight": weight_vector[index],
                "mean_return_per_period": means[index],
                "standalone_volatility_per_period": standalone_volatility,
                "annualized_standalone_volatility": standalone_volatility
                * annualization,
                "marginal_volatility": covariance_weight[index]
                / portfolio_volatility,
                "component_volatility": components[index],
                "component_fraction_of_portfolio_volatility": components[index]
                / portfolio_volatility,
                "absolute_component_share": absolute_shares[index],
            }
        )

    metrics = {
        "observation_count": observation_count,
        "asset_count": len(assets),
        "portfolio_variance_per_period": portfolio_variance,
        "portfolio_volatility_per_period": portfolio_volatility,
        "annualized_portfolio_volatility": portfolio_volatility * annualization,
        "component_volatility_sum": sum(components),
        "component_sum_error": sum(components) - portfolio_volatility,
        "gross_absolute_component_volatility": gross_component,
        "largest_absolute_risk_share": absolute_shares[largest_index],
        "risk_concentration_hhi": risk_hhi,
        "effective_absolute_risk_contributors": 1.0 / risk_hhi,
    }
    failures = []
    for metric in ("largest_absolute_risk_share", "risk_concentration_hhi"):
        maximum = thresholds[metric]
        actual = metrics[metric]
        if maximum is not None and actual > maximum:
            failure = {
                "metric": metric,
                "actual": actual,
                "maximum": maximum,
                "excess": actual - maximum,
            }
            if metric == "largest_absolute_risk_share":
                failure["asset"] = assets[largest_index]
            failures.append(failure)

    return {
        "passed": not failures,
        "portfolio": {
            "snapshot_date": selected_date,
            "return_start_date": returns[0][0],
            "return_end_date": returns[-1][0],
            "gross_exposure": sum(abs(weight) for weight in weight_vector),
            "net_exposure": sum(weight_vector),
        },
        "metrics": metrics,
        "thresholds": thresholds,
        "failures": failures,
        "largest_absolute_risk_contributor": asset_contributions[largest_index],
        "asset_contributions": asset_contributions,
        "covariance": {"assets": assets, "sample_matrix": covariance},
        "settings": {"periods_per_year": periods_per_year},
    }


def load_returns_csv(path: Path) -> list[dict[str, Any]]:
    """Load a strict wide date plus one-column-per-asset return CSV."""

    records = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if (
            fieldnames is None
            or len(fieldnames) < 2
            or fieldnames[0] != "date"
            or any(not field or field.strip() != field for field in fieldnames)
            or len(set(fieldnames)) != len(fieldnames)
        ):
            raise ValueError(
                "returns CSV header must be date followed by unique asset columns"
            )
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(
                    f"returns row {row_number} must match the header field count"
                )
            record: dict[str, Any] = {"date": row.get("date")}
            for asset in fieldnames[1:]:
                try:
                    record[asset] = float(row.get(asset) or "")
                except ValueError as error:
                    raise ValueError(
                        f"returns row {row_number} has an invalid {asset} return"
                    ) from error
            records.append(record)
    if not records:
        raise ValueError("returns CSV must contain at least one row")
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
    parser.add_argument("portfolio", type=Path)
    parser.add_argument("returns", type=Path)
    parser.add_argument("--date")
    parser.add_argument("--periods-per-year", type=int, default=252)
    parser.add_argument("--max-largest-risk-share", type=float)
    parser.add_argument("--max-risk-concentration-hhi", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        if args.output is not None and (
            _paths_alias(args.portfolio, args.output)
            or _paths_alias(args.returns, args.output)
        ):
            raise ValueError("output must not alias an input CSV")
        report = analyze_risk_contributions(
            load_portfolio_csv(args.portfolio),
            load_returns_csv(args.returns),
            snapshot_date=args.date,
            periods_per_year=args.periods_per_year,
            max_largest_risk_share=args.max_largest_risk_share,
            max_risk_concentration_hhi=args.max_risk_concentration_hhi,
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
