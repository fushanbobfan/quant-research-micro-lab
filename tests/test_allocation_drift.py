import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

from quant_research_micro_lab.allocation_drift import (
    audit_allocation_drift,
    load_allocation_csv,
    main,
)


class AllocationDriftTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"date": "2026-01-01", "asset": "AAA", "target_weight": 0.6, "actual_weight": 0.55},
            {"date": "2026-01-01", "asset": "BBB", "target_weight": 0.4, "actual_weight": 0.45},
            {"date": "2026-01-02", "asset": "AAA", "target_weight": 0.5, "actual_weight": 0.7},
            {"date": "2026-01-02", "asset": "BBB", "target_weight": 0.3, "actual_weight": 0.2},
            {"date": "2026-01-02", "asset": "CCC", "target_weight": 0.2, "actual_weight": 0.1},
        ]

    def test_reports_snapshot_and_asset_drift(self):
        report = audit_allocation_drift(self.records, asset_tolerance=0.1)

        self.assertTrue(report["passed"])
        self.assertEqual(report["snapshot_count"], 2)
        self.assertEqual(report["asset_count"], 3)
        self.assertEqual(report["position_comparison_count"], 5)
        self.assertAlmostEqual(report["summary"]["average_l1_drift"], 0.25)
        self.assertEqual(
            report["summary"]["maximum_snapshot_l1_drift"]["date"],
            "2026-01-02",
        )
        self.assertEqual(
            report["summary"]["maximum_asset_drift"]["asset"], "AAA"
        )
        self.assertAlmostEqual(report["summary"]["within_tolerance_rate"], 0.8)
        worst = report["snapshot_details"][0]
        self.assertAlmostEqual(worst["target_net_weight"], 1.0)
        self.assertAlmostEqual(worst["actual_net_weight"], 1.0)
        self.assertEqual(worst["asset_details"][0]["asset"], "AAA")

    def test_threshold_failures_are_reported_in_stable_order(self):
        report = audit_allocation_drift(
            self.records,
            asset_tolerance=0.01,
            max_average_l1_drift=0.1,
            max_snapshot_l1_drift=0.2,
            max_asset_drift=0.15,
            min_within_tolerance_rate=0.5,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(
            [failure["metric"] for failure in report["failures"]],
            [
                "average_l1_drift",
                "snapshot_l1_drift",
                "asset_drift",
                "within_tolerance_rate",
            ],
        )

    def test_detail_output_is_bounded_and_deterministic(self):
        report = audit_allocation_drift(self.records, max_details=1)

        self.assertEqual(len(report["snapshot_details"]), 1)
        self.assertTrue(report["details_truncated"])
        self.assertEqual(report["omitted_snapshot_count"], 1)
        self.assertEqual(len(report["snapshot_details"][0]["asset_details"]), 1)
        self.assertTrue(report["snapshot_details"][0]["asset_details_truncated"])

    def test_short_allocations_and_net_weight_drift_are_preserved(self):
        report = audit_allocation_drift(
            [
                {"date": "2026-01-01", "asset": "LONG", "target_weight": 1.2, "actual_weight": 1.1},
                {"date": "2026-01-01", "asset": "SHORT", "target_weight": -0.2, "actual_weight": -0.15},
            ]
        )

        detail = report["snapshot_details"][0]
        self.assertAlmostEqual(detail["target_net_weight"], 1.0)
        self.assertAlmostEqual(detail["actual_net_weight"], 0.95)
        self.assertAlmostEqual(detail["net_weight_drift"], -0.05)

    def test_invalid_records_and_thresholds_are_rejected(self):
        cases = [
            ([], {}, "at least one"),
            (self.records, {"asset_tolerance": -1}, "asset_tolerance"),
            (self.records, {"max_asset_drift": math.inf}, "max_asset_drift"),
            (self.records, {"min_within_tolerance_rate": 1.1}, "between 0 and 1"),
            (self.records, {"max_details": -1}, "max_details"),
            ([{**self.records[0], "extra": 1}], {}, "exactly"),
            ([{**self.records[0], "date": "01-01-2026"}], {}, "YYYY-MM-DD"),
            ([{**self.records[0], "actual_weight": math.nan}], {}, "finite"),
        ]
        for records, kwargs, message in cases:
            with self.subTest(records=records, kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    audit_allocation_drift(records, **kwargs)

        with self.assertRaisesRegex(ValueError, "repeats date and asset"):
            audit_allocation_drift([self.records[0], self.records[0]])
        with self.assertRaisesRegex(ValueError, "increasing order"):
            audit_allocation_drift([self.records[2], self.records[0]])

    def test_csv_loader_requires_exact_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "allocation.csv"
            path.write_text("date,asset,target_weight\n2026-01-01,AAA,1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "header"):
                load_allocation_csv(path)

    def test_cli_writes_a_report_and_returns_one_for_a_failed_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "allocation.csv"
            output = Path(directory) / "report.json"
            dataset.write_text(
                "date,asset,target_weight,actual_weight\n"
                "2026-01-01,AAA,1,0.7\n"
                "2026-01-01,BBB,0,0.3\n",
                encoding="utf-8",
            )

            exit_code = main(
                [str(dataset), "--max-asset-drift", "0.2", "--output", str(output)]
            )

            self.assertEqual(exit_code, 1)
            self.assertFalse(json.loads(output.read_text(encoding="utf-8"))["passed"])

    def test_cli_rejects_output_alias_and_oversized_input(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "allocation.csv"
            dataset.write_text(
                "date,asset,target_weight,actual_weight\n2026-01-01,AAA,1,1\n",
                encoding="utf-8",
            )
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([str(dataset), "--output", str(dataset)]), 2)
                self.assertEqual(main([str(dataset), "--max-file-bytes", "5"]), 2)


if __name__ == "__main__":
    unittest.main()
