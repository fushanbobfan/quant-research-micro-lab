import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import quant_research_micro_lab
from quant_research_micro_lab.group_exposure import (
    audit_group_exposure,
    load_group_exposure_csv,
    main,
)


class GroupExposureTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"date": "2026-01-01", "asset": "AAA", "group": "Tech", "weight": 0.6},
            {"date": "2026-01-01", "asset": "BBB", "group": "Health", "weight": 0.4},
            {"date": "2026-01-02", "asset": "AAA", "group": "Tech", "weight": 0.4},
            {"date": "2026-01-02", "asset": "CCC", "group": "Tech", "weight": -0.2},
            {"date": "2026-01-02", "asset": "BBB", "group": "Health", "weight": 0.3},
            {"date": "2026-01-02", "asset": "DDD", "group": "Utilities", "weight": 0.1},
        ]

    def test_reports_group_exposure_and_concentration(self):
        report = audit_group_exposure(self.records)

        self.assertIs(quant_research_micro_lab.audit_group_exposure, audit_group_exposure)
        self.assertIs(
            quant_research_micro_lab.load_group_exposure_csv,
            load_group_exposure_csv,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["snapshot_count"], 2)
        self.assertEqual(report["asset_count"], 4)
        self.assertEqual(report["group_count"], 3)
        self.assertAlmostEqual(
            report["summary"]["maximum_group_gross_share"]["value"], 0.6
        )
        self.assertEqual(
            report["summary"]["maximum_group_gross_share"]["date"], "2026-01-01"
        )
        self.assertAlmostEqual(
            report["summary"]["maximum_group_concentration_hhi"]["value"], 0.52
        )
        self.assertAlmostEqual(
            report["summary"]["minimum_effective_groups"]["value"], 1 / 0.52
        )

    def test_short_positions_preserve_group_gross_and_net_exposure(self):
        report = audit_group_exposure(self.records)
        second = next(
            detail
            for detail in report["snapshot_details"]
            if detail["date"] == "2026-01-02"
        )
        tech = next(group for group in second["groups"] if group["group"] == "Tech")

        self.assertAlmostEqual(tech["long_exposure"], 0.4)
        self.assertAlmostEqual(tech["short_exposure"], 0.2)
        self.assertAlmostEqual(tech["net_exposure"], 0.2)
        self.assertAlmostEqual(tech["gross_exposure"], 0.6)
        self.assertAlmostEqual(tech["gross_share"], 0.6)

    def test_decimal_boundary_does_not_create_a_false_breach(self):
        report = audit_group_exposure(
            self.records,
            max_group_gross_share=0.6,
            max_abs_group_net_exposure=0.6,
            max_group_concentration_hhi=0.52,
            min_effective_groups=1 / 0.52,
        )

        self.assertTrue(report["passed"])

    def test_threshold_failures_have_stable_order(self):
        report = audit_group_exposure(
            self.records,
            max_group_gross_share=0.5,
            max_abs_group_net_exposure=0.5,
            max_group_concentration_hhi=0.5,
            min_effective_groups=2.0,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(
            [failure["metric"] for failure in report["failures"]],
            [
                "group_gross_share",
                "abs_group_net_exposure",
                "group_concentration_hhi",
                "effective_groups",
            ],
        )

    def test_snapshot_and_group_details_are_bounded(self):
        report = audit_group_exposure(self.records, max_details=1)

        self.assertEqual(len(report["snapshot_details"]), 1)
        self.assertTrue(report["details_truncated"])
        self.assertEqual(report["omitted_snapshot_count"], 1)
        self.assertEqual(len(report["snapshot_details"][0]["groups"]), 1)
        self.assertTrue(report["snapshot_details"][0]["groups_truncated"])

    def test_invalid_records_and_configuration_are_rejected(self):
        cases = [
            ([], {}, "at least one"),
            (self.records, {"max_group_gross_share": 1.1}, "between 0 and 1"),
            (self.records, {"max_abs_group_net_exposure": math.inf}, "non-negative"),
            (self.records, {"min_effective_groups": 0.9}, "at least 1"),
            (self.records, {"max_details": True}, "max_details"),
            ([{**self.records[0], "extra": 1}], {}, "exactly"),
            ([{**self.records[0], "date": "01-01-2026"}], {}, "YYYY-MM-DD"),
            ([{**self.records[0], "group": ""}], {}, "non-empty"),
            ([{**self.records[0], "weight": math.nan}], {}, "finite"),
        ]
        for records, kwargs, message in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    audit_group_exposure(records, **kwargs)

        with self.assertRaisesRegex(ValueError, "repeats date"):
            audit_group_exposure([self.records[0], self.records[0]])
        with self.assertRaisesRegex(ValueError, "non-decreasing"):
            audit_group_exposure([self.records[2], self.records[0]])
        with self.assertRaisesRegex(ValueError, "non-zero gross"):
            audit_group_exposure(
                [
                    {
                        "date": "2026-01-01",
                        "asset": "AAA",
                        "group": "Tech",
                        "weight": 0.0,
                    }
                ]
            )

    def test_csv_loader_requires_the_exact_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "groups.csv"
            dataset.write_text(
                "date,asset,weight\n2026-01-01,AAA,1\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "header"):
                load_group_exposure_csv(dataset)

    def test_cli_writes_a_report_and_returns_one_for_a_failed_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "groups.csv"
            output = Path(directory) / "report.json"
            dataset.write_text(
                "date,asset,group,weight\n"
                "2026-01-01,AAA,Tech,0.7\n"
                "2026-01-01,BBB,Health,0.3\n",
                encoding="utf-8",
            )

            exit_code = main(
                [
                    str(dataset),
                    "--max-group-gross-share",
                    "0.6",
                    "--output",
                    str(output),
                ]
            )

            self.assertEqual(exit_code, 1)
            self.assertFalse(json.loads(output.read_text(encoding="utf-8"))["passed"])

    def test_cli_prints_a_passing_report(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "groups.csv"
            dataset.write_text(
                "date,asset,group,weight\n"
                "2026-01-01,AAA,Tech,0.5\n"
                "2026-01-01,BBB,Health,0.5\n",
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                exit_code = main(
                    [str(dataset), "--max-group-gross-share", "0.5"]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(json.loads(stdout.getvalue())["passed"])

    def test_cli_rejects_output_alias_and_oversized_input(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "groups.csv"
            dataset.write_text(
                "date,asset,group,weight\n2026-01-01,AAA,Tech,1\n",
                encoding="utf-8",
            )

            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([str(dataset), "--output", str(dataset)]), 2)
                self.assertEqual(main([str(dataset), "--max-file-bytes", "5"]), 2)


if __name__ == "__main__":
    unittest.main()
