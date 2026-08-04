"""Backtest one-period value-at-risk forecasts against realized returns."""

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


def _validate_rate(name: str, value: float | None) -> None:
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{name} must be a finite number between 0 and 1")


def _log_likelihood(successes: int, observations: int, probability: float) -> float:
    failures = observations - successes
    result = 0.0
    if successes:
        result += successes * math.log(probability)
    if failures:
        result += failures * math.log1p(-probability)
    return result


def backtest_var_forecasts(
    records: Sequence[Mapping[str, Any]],
    *,
    confidence: float = 0.99,
    max_exception_rate: float | None = None,
    min_kupiec_p_value: float | None = None,
    max_exception_count: int | None = None,
    max_details: int = 20,
) -> dict[str, Any]:
    """Return exception diagnostics and Kupiec unconditional coverage results."""

    if not records:
        raise ValueError("at least one record is required")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, Real)
        or not math.isfinite(confidence)
        or not 0.0 < confidence < 1.0
    ):
        raise ValueError("confidence must be a finite number between 0 and 1")
    _validate_rate("max_exception_rate", max_exception_rate)
    _validate_rate("min_kupiec_p_value", min_kupiec_p_value)
    if (
        max_exception_count is not None
        and (
            isinstance(max_exception_count, bool)
            or not isinstance(max_exception_count, int)
            or max_exception_count < 0
        )
    ):
        raise ValueError("max_exception_count must be a non-negative integer")
    if (
        isinstance(max_details, bool)
        or not isinstance(max_details, int)
        or max_details < 0
    ):
        raise ValueError("max_details must be a non-negative integer")

    validated: list[tuple[str, float, float]] = []
    previous_date: date | None = None
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"record {index} must be an object")
        date_value = record.get("date")
        realized_return = record.get("realized_return")
        var = record.get("var")
        if not isinstance(date_value, str):
            raise ValueError(f"record {index} date must use YYYY-MM-DD")
        try:
            parsed_date = date.fromisoformat(date_value)
        except ValueError as error:
            raise ValueError(f"record {index} date must use YYYY-MM-DD") from error
        if parsed_date.isoformat() != date_value:
            raise ValueError(f"record {index} date must use YYYY-MM-DD")
        if previous_date is not None and parsed_date <= previous_date:
            raise ValueError("dates must be unique and strictly increasing")
        if (
            isinstance(realized_return, bool)
            or not isinstance(realized_return, Real)
            or not math.isfinite(realized_return)
            or realized_return < -1.0
        ):
            raise ValueError(
                f"record {index} realized_return must be finite and at least -1"
            )
        if (
            isinstance(var, bool)
            or not isinstance(var, Real)
            or not math.isfinite(var)
            or not 0.0 <= var <= 1.0
        ):
            raise ValueError(f"record {index} var must be between 0 and 1")
        previous_date = parsed_date
        validated.append((date_value, float(realized_return), float(var)))

    exceptions: list[dict[str, Any]] = []
    exception_flags = []
    all_exception_losses = []
    all_shortfalls = []
    for date_value, realized_return, var in validated:
        loss = max(0.0, -realized_return)
        is_exception = loss > var
        exception_flags.append(is_exception)
        if is_exception:
            shortfall = loss - var
            all_exception_losses.append(loss)
            all_shortfalls.append(shortfall)
            if len(exceptions) < max_details:
                exceptions.append(
                    {
                        "date": date_value,
                        "realized_return": realized_return,
                        "var": var,
                        "loss": loss,
                        "shortfall": shortfall,
                    }
                )

    observations = len(validated)
    exception_count = len(all_exception_losses)
    exception_rate = exception_count / observations
    expected_exception_rate = 1.0 - float(confidence)
    null_log_likelihood = _log_likelihood(
        exception_count,
        observations,
        expected_exception_rate,
    )
    observed_log_likelihood = _log_likelihood(
        exception_count,
        observations,
        exception_rate,
    )
    kupiec_likelihood_ratio = max(
        0.0,
        -2.0 * (null_log_likelihood - observed_log_likelihood),
    )
    kupiec_p_value = math.erfc(math.sqrt(kupiec_likelihood_ratio / 2.0))

    longest_exception_streak = 0
    current_streak = 0
    for flag in exception_flags:
        current_streak = current_streak + 1 if flag else 0
        longest_exception_streak = max(longest_exception_streak, current_streak)
    adjacent_exception_pairs = sum(
        left and right
        for left, right in zip(exception_flags, exception_flags[1:], strict=False)
    )

    metrics = {
        "observations": observations,
        "confidence": float(confidence),
        "expected_exception_rate": expected_exception_rate,
        "expected_exception_count": expected_exception_rate * observations,
        "exception_count": exception_count,
        "exception_rate": exception_rate,
        "mean_exception_loss": (
            sum(all_exception_losses) / exception_count if exception_count else None
        ),
        "mean_exception_shortfall": (
            sum(all_shortfalls) / exception_count if exception_count else None
        ),
        "maximum_exception_loss": (
            max(all_exception_losses) if exception_count else None
        ),
        "maximum_exception_shortfall": (
            max(all_shortfalls) if exception_count else None
        ),
        "longest_exception_streak": longest_exception_streak,
        "adjacent_exception_pairs": adjacent_exception_pairs,
        "kupiec_likelihood_ratio": kupiec_likelihood_ratio,
        "kupiec_p_value": kupiec_p_value,
    }

    thresholds = {
        "exception_rate": (
            float(max_exception_rate) if max_exception_rate is not None else None
        ),
        "kupiec_p_value": (
            float(min_kupiec_p_value) if min_kupiec_p_value is not None else None
        ),
        "exception_count": max_exception_count,
    }
    failures = []
    if max_exception_rate is not None and exception_rate > max_exception_rate:
        failures.append(
            {
                "metric": "exception_rate",
                "actual": exception_rate,
                "maximum": float(max_exception_rate),
                "excess": exception_rate - float(max_exception_rate),
            }
        )
    if min_kupiec_p_value is not None and kupiec_p_value < min_kupiec_p_value:
        failures.append(
            {
                "metric": "kupiec_p_value",
                "actual": kupiec_p_value,
                "minimum": float(min_kupiec_p_value),
                "shortfall": float(min_kupiec_p_value) - kupiec_p_value,
            }
        )
    if max_exception_count is not None and exception_count > max_exception_count:
        failures.append(
            {
                "metric": "exception_count",
                "actual": exception_count,
                "maximum": max_exception_count,
                "excess": exception_count - max_exception_count,
            }
        )

    return {
        "passed": not failures,
        "start_date": validated[0][0],
        "end_date": validated[-1][0],
        "metrics": metrics,
        "thresholds": thresholds,
        "failures": failures,
        "exceptions": exceptions,
        "details_truncated": exception_count > len(exceptions),
        "settings": {"max_details": max_details},
    }


def load_var_csv(path: Path) -> list[dict[str, Any]]:
    """Load the strict date,realized_return,var forecast format."""

    records = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["date", "realized_return", "var"]:
            raise ValueError("CSV header must be exactly: date,realized_return,var")
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"row {row_number} must contain exactly three fields")
            try:
                realized_return = float(row.get("realized_return") or "")
                var = float(row.get("var") or "")
            except ValueError as error:
                raise ValueError(
                    f"row {row_number} has an invalid realized_return or var"
                ) from error
            records.append(
                {
                    "date": row.get("date"),
                    "realized_return": realized_return,
                    "var": var,
                }
            )
    if not records:
        raise ValueError("CSV must contain at least one forecast row")
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
    parser.add_argument("--confidence", type=float, default=0.99)
    parser.add_argument("--max-exception-rate", type=float)
    parser.add_argument("--min-kupiec-p-value", type=float)
    parser.add_argument("--max-exception-count", type=int)
    parser.add_argument("--max-details", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        if args.output is not None and _paths_alias(args.dataset, args.output):
            raise ValueError("output must not alias the source CSV")
        report = backtest_var_forecasts(
            load_var_csv(args.dataset),
            confidence=args.confidence,
            max_exception_rate=args.max_exception_rate,
            min_kupiec_p_value=args.min_kupiec_p_value,
            max_exception_count=args.max_exception_count,
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
