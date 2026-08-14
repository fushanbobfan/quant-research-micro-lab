"""Compare sample return correlations across two historical windows."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from itertools import combinations
from numbers import Real
from pathlib import Path
from typing import Any

from .risk_contribution import _validate_return_records, load_returns_csv


def _optional_change_limit(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or not 0 <= value <= 2
    ):
        raise ValueError(f"{name} must be a finite number between 0 and 2")
    return float(value)


def _optional_count(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _asset_columns(
    records: Sequence[Mapping[str, Any]], *, window_name: str
) -> list[str]:
    if len(records) < 3:
        raise ValueError(f"{window_name} window requires at least three observations")
    first = records[0]
    if not isinstance(first, Mapping) or "date" not in first:
        raise ValueError(f"{window_name} records must include date and asset columns")
    assets = sorted(set(first) - {"date"})
    if len(assets) < 2 or any(
        not isinstance(asset, str) or not asset or asset.strip() != asset
        for asset in assets
    ):
        raise ValueError(f"{window_name} window requires at least two asset columns")
    return assets


def _pearson(values_a: Sequence[float], values_b: Sequence[float], label: str) -> float:
    mean_a = sum(values_a) / len(values_a)
    mean_b = sum(values_b) / len(values_b)
    centered_a = [value - mean_a for value in values_a]
    centered_b = [value - mean_b for value in values_b]
    sum_squares_a = sum(value * value for value in centered_a)
    sum_squares_b = sum(value * value for value in centered_b)
    if sum_squares_a <= 0 or sum_squares_b <= 0:
        raise ValueError(f"{label} correlation is undefined because an asset has zero variance")
    numerator = sum(
        value_a * value_b for value_a, value_b in zip(centered_a, centered_b)
    )
    correlation = numerator / math.sqrt(sum_squares_a * sum_squares_b)
    return max(-1.0, min(1.0, correlation))


def analyze_correlation_drift(
    baseline_records: Sequence[Mapping[str, Any]],
    candidate_records: Sequence[Mapping[str, Any]],
    *,
    max_abs_correlation_change: float | None = None,
    max_rms_correlation_change: float | None = None,
    max_sign_flips: int | None = None,
    max_details: int = 20,
) -> dict[str, Any]:
    """Return pairwise historical correlation changes and optional gate failures."""

    baseline_assets = _asset_columns(baseline_records, window_name="baseline")
    candidate_assets = _asset_columns(candidate_records, window_name="candidate")
    if baseline_assets != candidate_assets:
        raise ValueError("baseline and candidate must use the same asset columns")
    if isinstance(max_details, bool) or not isinstance(max_details, int) or max_details < 0:
        raise ValueError("max_details must be a non-negative integer")
    thresholds = {
        "maximum_abs_correlation_change": _optional_change_limit(
            "max_abs_correlation_change", max_abs_correlation_change
        ),
        "rms_correlation_change": _optional_change_limit(
            "max_rms_correlation_change", max_rms_correlation_change
        ),
        "sign_flip_count": _optional_count("max_sign_flips", max_sign_flips),
    }

    baseline = _validate_return_records(baseline_records, baseline_assets)
    candidate = _validate_return_records(candidate_records, baseline_assets)
    baseline_columns = {
        asset: [values[index] for _, values in baseline]
        for index, asset in enumerate(baseline_assets)
    }
    candidate_columns = {
        asset: [values[index] for _, values in candidate]
        for index, asset in enumerate(baseline_assets)
    }

    pair_changes = []
    for asset_a, asset_b in combinations(baseline_assets, 2):
        baseline_correlation = _pearson(
            baseline_columns[asset_a],
            baseline_columns[asset_b],
            f"baseline {asset_a}/{asset_b}",
        )
        candidate_correlation = _pearson(
            candidate_columns[asset_a],
            candidate_columns[asset_b],
            f"candidate {asset_a}/{asset_b}",
        )
        change = candidate_correlation - baseline_correlation
        pair_changes.append(
            {
                "asset_a": asset_a,
                "asset_b": asset_b,
                "baseline_correlation": baseline_correlation,
                "candidate_correlation": candidate_correlation,
                "change": change,
                "absolute_change": abs(change),
                "sign_flipped": baseline_correlation * candidate_correlation < 0,
            }
        )
    pair_changes.sort(
        key=lambda item: (-item["absolute_change"], item["asset_a"], item["asset_b"])
    )
    pair_count = len(pair_changes)
    maximum_change = pair_changes[0]["absolute_change"]
    rms_change = math.sqrt(
        sum(item["change"] ** 2 for item in pair_changes) / pair_count
    )
    metrics = {
        "baseline_observation_count": len(baseline),
        "candidate_observation_count": len(candidate),
        "asset_count": len(baseline_assets),
        "pair_count": pair_count,
        "mean_abs_correlation_change": sum(
            item["absolute_change"] for item in pair_changes
        )
        / pair_count,
        "rms_correlation_change": rms_change,
        "maximum_abs_correlation_change": maximum_change,
        "sign_flip_count": sum(item["sign_flipped"] for item in pair_changes),
    }

    failures = []
    failure_specs = (
        "maximum_abs_correlation_change",
        "rms_correlation_change",
        "sign_flip_count",
    )
    for metric in failure_specs:
        maximum = thresholds[metric]
        actual = metrics[metric]
        if maximum is not None and actual > maximum:
            failures.append(
                {
                    "metric": metric,
                    "actual": actual,
                    "maximum": maximum,
                    "excess": actual - maximum,
                }
            )

    return {
        "passed": not failures,
        "windows": {
            "baseline": {
                "start_date": baseline[0][0],
                "end_date": baseline[-1][0],
                "observation_count": len(baseline),
            },
            "candidate": {
                "start_date": candidate[0][0],
                "end_date": candidate[-1][0],
                "observation_count": len(candidate),
            },
        },
        "metrics": metrics,
        "thresholds": thresholds,
        "failures": failures,
        "largest_change": pair_changes[0],
        "pair_changes": pair_changes[:max_details],
        "details_truncated": pair_count > max_details,
        "settings": {
            "assets": baseline_assets,
            "correlation_method": "sample_pearson",
            "max_details": max_details,
        },
    }


def _paths_alias(first: Path, second: Path) -> bool:
    if first.resolve() == second.resolve():
        return True
    try:
        return first.samefile(second)
    except (FileNotFoundError, OSError):
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--max-abs-correlation-change", type=float)
    parser.add_argument("--max-rms-correlation-change", type=float)
    parser.add_argument("--max-sign-flips", type=int)
    parser.add_argument("--max-details", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        if _paths_alias(args.baseline, args.candidate):
            raise ValueError("baseline and candidate must be different input files")
        if args.output is not None and (
            _paths_alias(args.baseline, args.output)
            or _paths_alias(args.candidate, args.output)
        ):
            raise ValueError("output must not alias an input CSV")
        report = analyze_correlation_drift(
            load_returns_csv(args.baseline),
            load_returns_csv(args.candidate),
            max_abs_correlation_change=args.max_abs_correlation_change,
            max_rms_correlation_change=args.max_rms_correlation_change,
            max_sign_flips=args.max_sign_flips,
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
