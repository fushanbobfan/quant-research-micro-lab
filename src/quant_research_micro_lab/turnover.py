"""Audit portfolio weight changes between dated snapshots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .exposure import _validate_limit, _validate_records


def audit_portfolio_turnover(
    records: Sequence[Mapping[str, Any]],
    *,
    max_transition_turnover: float | None = None,
    max_position_change: float | None = None,
    max_cumulative_turnover: float | None = None,
    max_details: int = 20,
) -> dict[str, Any]:
    """Return consecutive-snapshot turnover and position-change diagnostics."""

    thresholds = {
        "transition_turnover": _validate_limit(
            "max_transition_turnover", max_transition_turnover
        ),
        "position_change": _validate_limit(
            "max_position_change", max_position_change
        ),
        "cumulative_turnover": _validate_limit(
            "max_cumulative_turnover", max_cumulative_turnover
        ),
    }
    if (
        isinstance(max_details, bool)
        or not isinstance(max_details, int)
        or max_details < 0
    ):
        raise ValueError("max_details must be a non-negative integer")
    validated = _validate_records(records)
    weights_by_date: dict[str, dict[str, float]] = {}
    for date_value, asset, weight in validated:
        weights_by_date.setdefault(date_value, {})[asset] = weight

    dates = list(weights_by_date)
    if len(dates) < 2:
        raise ValueError("at least two distinct snapshot dates are required")
    transitions = []
    all_changes = []
    for previous_date, current_date in zip(dates, dates[1:]):
        previous = weights_by_date[previous_date]
        current = weights_by_date[current_date]
        changes = []
        for asset in sorted(set(previous) | set(current)):
            previous_weight = previous.get(asset, 0.0)
            current_weight = current.get(asset, 0.0)
            change = current_weight - previous_weight
            if change == 0:
                continue
            changes.append(
                {
                    "asset": asset,
                    "previous_weight": previous_weight,
                    "current_weight": current_weight,
                    "change": change,
                    "absolute_change": abs(change),
                }
            )

        ranked_changes = sorted(
            changes,
            key=lambda item: (-item["absolute_change"], item["asset"]),
        )
        total_absolute_change = sum(item["absolute_change"] for item in changes)
        transition = {
            "previous_date": previous_date,
            "date": current_date,
            "changed_asset_count": len(changes),
            "total_absolute_change": total_absolute_change,
            "one_way_turnover": 0.5 * total_absolute_change,
            "opened_positions": sum(
                item["previous_weight"] == 0 and item["current_weight"] != 0
                for item in changes
            ),
            "closed_positions": sum(
                item["previous_weight"] != 0 and item["current_weight"] == 0
                for item in changes
            ),
            "sign_flips": sum(
                item["previous_weight"] * item["current_weight"] < 0
                for item in changes
            ),
            "change_details": ranked_changes[:max_details],
            "details_truncated": len(changes) > max_details,
        }
        transitions.append(transition)
        all_changes.extend({"date": current_date, **item} for item in changes)

    maximum_turnover = min(
        transitions,
        key=lambda item: (-item["one_way_turnover"], item["date"]),
    )
    maximum_change = (
        min(
            all_changes,
            key=lambda item: (
                -item["absolute_change"],
                item["date"],
                item["asset"],
            ),
        )
        if all_changes
        else None
    )
    cumulative_turnover = sum(item["one_way_turnover"] for item in transitions)
    failures = []
    maximum_transition_limit = thresholds["transition_turnover"]
    if (
        maximum_transition_limit is not None
        and maximum_turnover["one_way_turnover"] > maximum_transition_limit
    ):
        failures.append(
            {
                "metric": "transition_turnover",
                "date": maximum_turnover["date"],
                "actual": maximum_turnover["one_way_turnover"],
                "maximum": maximum_transition_limit,
                "excess": (
                    maximum_turnover["one_way_turnover"]
                    - maximum_transition_limit
                ),
            }
        )
    maximum_position_limit = thresholds["position_change"]
    if (
        maximum_position_limit is not None
        and maximum_change is not None
        and maximum_change["absolute_change"] > maximum_position_limit
    ):
        failures.append(
            {
                "metric": "position_change",
                "date": maximum_change["date"],
                "asset": maximum_change["asset"],
                "actual": maximum_change["absolute_change"],
                "maximum": maximum_position_limit,
                "excess": maximum_change["absolute_change"] - maximum_position_limit,
            }
        )
    maximum_cumulative_limit = thresholds["cumulative_turnover"]
    if (
        maximum_cumulative_limit is not None
        and cumulative_turnover > maximum_cumulative_limit
    ):
        failures.append(
            {
                "metric": "cumulative_turnover",
                "actual": cumulative_turnover,
                "maximum": maximum_cumulative_limit,
                "excess": cumulative_turnover - maximum_cumulative_limit,
            }
        )
    return {
        "passed": not failures,
        "start_date": dates[0],
        "end_date": dates[-1],
        "snapshot_count": len(dates),
        "transition_count": len(transitions),
        "asset_count": len({asset for _, asset, _ in validated}),
        "summary": {
            "cumulative_turnover": cumulative_turnover,
            "average_turnover": cumulative_turnover / len(transitions),
            "maximum_turnover": {
                "date": maximum_turnover["date"],
                "value": maximum_turnover["one_way_turnover"],
            },
            "maximum_position_change": (
                {
                    "date": maximum_change["date"],
                    "asset": maximum_change["asset"],
                    "change": maximum_change["change"],
                    "absolute_change": maximum_change["absolute_change"],
                }
                if maximum_change
                else None
            ),
        },
        "thresholds": thresholds,
        "failures": failures,
        "transitions": transitions,
        "settings": {"max_details": max_details},
    }
