"""Audit dated portfolio weights for group-level exposure concentration."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date
from numbers import Real
from typing import Any


def _stable_metric(value: float) -> float:
    return round(value, 15)


def _exceeds(actual: float, maximum: float) -> bool:
    return actual > maximum and not math.isclose(
        actual, maximum, rel_tol=1e-12, abs_tol=1e-12
    )


def _maximum(name: str, value: float | None, *, unit_interval: bool = False) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value < 0.0
        or (unit_interval and value > 1.0)
    ):
        suffix = " between 0 and 1" if unit_interval else " non-negative"
        raise ValueError(f"{name} must be a finite{suffix} number")
    return float(value)


def _minimum(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value < 1.0
    ):
        raise ValueError(f"{name} must be a finite number at least 1")
    return float(value)


def _validate_records(
    records: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str, str, float]]:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("records must be a sequence")
    if not records:
        raise ValueError("at least one position record is required")

    expected_fields = {"date", "asset", "group", "weight"}
    validated = []
    seen: set[tuple[str, str]] = set()
    previous_date: str | None = None
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"record {index} must be an object")
        if set(record) != expected_fields:
            raise ValueError(
                f"record {index} must contain exactly date, asset, group, and weight"
            )
        date_value = record.get("date")
        asset = record.get("asset")
        group = record.get("group")
        weight = record.get("weight")
        if not isinstance(date_value, str):
            raise ValueError(f"record {index} date must use YYYY-MM-DD")
        try:
            parsed_date = date.fromisoformat(date_value)
        except ValueError as error:
            raise ValueError(f"record {index} date must use YYYY-MM-DD") from error
        if parsed_date.isoformat() != date_value:
            raise ValueError(f"record {index} date must use YYYY-MM-DD")
        if previous_date is not None and date_value < previous_date:
            raise ValueError("position dates must be in non-decreasing order")
        if not isinstance(asset, str) or not asset.strip():
            raise ValueError(f"record {index} asset must be a non-empty string")
        if not isinstance(group, str) or not group.strip():
            raise ValueError(f"record {index} group must be a non-empty string")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, Real)
            or not math.isfinite(weight)
        ):
            raise ValueError(f"record {index} weight must be a finite number")
        key = (date_value, asset)
        if key in seen:
            raise ValueError(
                f"record {index} repeats date {date_value!r} and asset {asset!r}"
            )
        seen.add(key)
        validated.append((date_value, asset, group, float(weight)))
        previous_date = date_value
    return validated


def audit_group_exposure(
    records: Sequence[Mapping[str, Any]],
    *,
    max_group_gross_share: float | None = None,
    max_abs_group_net_exposure: float | None = None,
    max_group_concentration_hhi: float | None = None,
    min_effective_groups: float | None = None,
    max_details: int = 20,
) -> dict[str, Any]:
    """Return group-level gross, net, and concentration diagnostics."""
    thresholds = {
        "group_gross_share": _maximum(
            "max_group_gross_share", max_group_gross_share, unit_interval=True
        ),
        "abs_group_net_exposure": _maximum(
            "max_abs_group_net_exposure", max_abs_group_net_exposure
        ),
        "group_concentration_hhi": _maximum(
            "max_group_concentration_hhi",
            max_group_concentration_hhi,
            unit_interval=True,
        ),
        "effective_groups": _minimum("min_effective_groups", min_effective_groups),
    }
    if isinstance(max_details, bool) or not isinstance(max_details, int) or max_details < 0:
        raise ValueError("max_details must be a non-negative integer")
    validated = _validate_records(records)

    snapshots_by_date: dict[str, list[tuple[str, str, float]]] = {}
    for date_value, asset, group, weight in validated:
        snapshots_by_date.setdefault(date_value, []).append((asset, group, weight))

    snapshots = []
    for date_value, positions in snapshots_by_date.items():
        gross_exposure = sum(abs(weight) for _, _, weight in positions)
        if gross_exposure == 0.0:
            raise ValueError(f"portfolio on {date_value} must have non-zero gross exposure")
        grouped: dict[str, dict[str, Any]] = {}
        for asset, group, weight in positions:
            values = grouped.setdefault(
                group,
                {
                    "assets": set(),
                    "long_exposure": 0.0,
                    "short_exposure": 0.0,
                    "net_exposure": 0.0,
                    "gross_exposure": 0.0,
                },
            )
            values["assets"].add(asset)
            values["long_exposure"] += max(weight, 0.0)
            values["short_exposure"] += max(-weight, 0.0)
            values["net_exposure"] += weight
            values["gross_exposure"] += abs(weight)

        group_details = []
        for group, values in grouped.items():
            group_details.append(
                {
                    "group": group,
                    "asset_count": len(values["assets"]),
                    "long_exposure": values["long_exposure"],
                    "short_exposure": values["short_exposure"],
                    "net_exposure": values["net_exposure"],
                    "abs_net_exposure": abs(values["net_exposure"]),
                    "gross_exposure": values["gross_exposure"],
                    "gross_share": values["gross_exposure"] / gross_exposure,
                }
            )
        group_details.sort(
            key=lambda item: (-_stable_metric(item["gross_share"]), item["group"])
        )
        concentration_hhi = sum(item["gross_share"] ** 2 for item in group_details)
        snapshots.append(
            {
                "date": date_value,
                "asset_count": len(positions),
                "group_count": len(group_details),
                "gross_exposure": gross_exposure,
                "net_exposure": sum(weight for _, _, weight in positions),
                "group_concentration_hhi": concentration_hhi,
                "effective_groups": 1.0 / concentration_hhi,
                "largest_group": group_details[0]["group"],
                "largest_group_gross_share": group_details[0]["gross_share"],
                "groups": group_details,
            }
        )

    maximum_share = min(
        (
            {"date": snapshot["date"], **group}
            for snapshot in snapshots
            for group in snapshot["groups"]
        ),
        key=lambda item: (
            -_stable_metric(item["gross_share"]),
            item["date"],
            item["group"],
        ),
    )
    maximum_abs_net = min(
        (
            {"date": snapshot["date"], **group}
            for snapshot in snapshots
            for group in snapshot["groups"]
        ),
        key=lambda item: (
            -_stable_metric(item["abs_net_exposure"]),
            item["date"],
            item["group"],
        ),
    )
    maximum_concentration = min(
        snapshots,
        key=lambda item: (
            -_stable_metric(item["group_concentration_hhi"]),
            item["date"],
        ),
    )
    minimum_effective = min(
        snapshots,
        key=lambda item: (_stable_metric(item["effective_groups"]), item["date"]),
    )

    failures = []
    maximum_checks = (
        (
            "group_gross_share",
            maximum_share["gross_share"],
            thresholds["group_gross_share"],
            maximum_share,
        ),
        (
            "abs_group_net_exposure",
            maximum_abs_net["abs_net_exposure"],
            thresholds["abs_group_net_exposure"],
            maximum_abs_net,
        ),
        (
            "group_concentration_hhi",
            maximum_concentration["group_concentration_hhi"],
            thresholds["group_concentration_hhi"],
            maximum_concentration,
        ),
    )
    for metric, actual, maximum, detail in maximum_checks:
        if maximum is not None and _exceeds(actual, maximum):
            failure = {
                "metric": metric,
                "date": detail["date"],
                "actual": actual,
                "maximum": maximum,
                "excess": actual - maximum,
            }
            if "group" in detail:
                failure["group"] = detail["group"]
            failures.append(failure)
    minimum = thresholds["effective_groups"]
    if (
        minimum is not None
        and minimum_effective["effective_groups"] < minimum
        and not math.isclose(
            minimum_effective["effective_groups"],
            minimum,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        failures.append(
            {
                "metric": "effective_groups",
                "date": minimum_effective["date"],
                "actual": minimum_effective["effective_groups"],
                "minimum": minimum,
                "shortfall": minimum - minimum_effective["effective_groups"],
            }
        )

    sorted_details = sorted(
        snapshots,
        key=lambda item: (
            -_stable_metric(item["group_concentration_hhi"]),
            item["date"],
        ),
    )
    bounded_details = []
    for snapshot in sorted_details[:max_details]:
        groups = snapshot["groups"]
        bounded_details.append(
            {
                **{key: value for key, value in snapshot.items() if key != "groups"},
                "groups": groups[:max_details],
                "groups_truncated": len(groups) > max_details,
                "omitted_group_count": max(0, len(groups) - max_details),
            }
        )

    return {
        "passed": not failures,
        "start_date": snapshots[0]["date"],
        "end_date": snapshots[-1]["date"],
        "snapshot_count": len(snapshots),
        "asset_count": len({asset for _, asset, _, _ in validated}),
        "group_count": len({group for _, _, group, _ in validated}),
        "summary": {
            "average_group_concentration_hhi": sum(
                snapshot["group_concentration_hhi"] for snapshot in snapshots
            )
            / len(snapshots),
            "average_effective_groups": sum(
                snapshot["effective_groups"] for snapshot in snapshots
            )
            / len(snapshots),
            "maximum_group_gross_share": {
                "date": maximum_share["date"],
                "group": maximum_share["group"],
                "value": maximum_share["gross_share"],
            },
            "maximum_abs_group_net_exposure": {
                "date": maximum_abs_net["date"],
                "group": maximum_abs_net["group"],
                "value": maximum_abs_net["abs_net_exposure"],
            },
            "maximum_group_concentration_hhi": {
                "date": maximum_concentration["date"],
                "value": maximum_concentration["group_concentration_hhi"],
            },
            "minimum_effective_groups": {
                "date": minimum_effective["date"],
                "value": minimum_effective["effective_groups"],
            },
        },
        "thresholds": thresholds,
        "failures": failures,
        "snapshot_details": bounded_details,
        "details_truncated": len(snapshots) > max_details,
        "omitted_snapshot_count": max(0, len(snapshots) - max_details),
        "settings": {"max_details": max_details},
    }
