import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from quant_research_micro_lab.panel_coverage import audit_panel_coverage, main


class PanelCoverageAuditTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"date": "2026-01-01", "ALPHA": 0.01, "BETA": 0.02, "GAMMA": None},
            {"date": "2026-01-02", "ALPHA": None, "BETA": 0.02, "GAMMA": 0.03},
            {"date": "2026-01-03", "ALPHA": None, "BETA": 0.02, "GAMMA": 0.03},
            {"date": "2026-01-04", "ALPHA": 0.01, "BETA": None, "GAMMA": 0.03},
        ]

    def test_reports_panel_asset_and_missing_streak_coverage(self):
        report = audit_panel_coverage(self.records)

        self.assertTrue(report["passed"])
        self.assertEqual(report["metrics"]["observation_count"], 4)
        self.assertEqual(report["metrics"]["asset_count"], 3)
        self.assertEqual(report["metrics"]["expected_cells"], 12)
        self.assertEqual(report["metrics"]["observed_cells"], 8)
        self.assertAlmostEqual(report["metrics"]["overall_coverage"], 2 / 3)
        self.assertEqual(report["metrics"]["incomplete_row_rate"], 1.0)
        self.assertEqual(
            report["metrics"]["longest_missing_streak"],
            {
                "asset": "ALPHA",
                "start_date": "2026-01-02",
                "end_date": "2026-01-03",
                "observations": 2,
            },
        )
        self.assertEqual(report["assets"][0]["asset"], "ALPHA")
        self.assertEqual(report["assets"][0]["coverage"], 0.5)

    def test_all_threshold_failures_are_reported_in_stable_order(self):
        report = audit_panel_coverage(
            self.records,
            min_overall_coverage=0.70,
            min_asset_coverage=0.60,
            max_missing_streak=1,
            max_incomplete_row_rate=0.50,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(
            [failure["metric"] for failure in report["failures"]],
            [
                "overall_coverage",
                "minimum_asset_coverage",
                "longest_missing_streak",
                "incomplete_row_rate",
            ],
        )

    def test_complete_panel_has_no_missing_streak(self):
        report = audit_panel_coverage(
            [
                {"date": "2026-02-01", "A": 1.0, "B": 2.0},
                {"date": "2026-02-02", "A": 1.1, "B": 2.1},
            ]
        )

        self.assertEqual(report["metrics"]["overall_coverage"], 1.0)
        self.assertEqual(report["metrics"]["incomplete_row_count"], 0)
        self.assertIsNone(report["metrics"]["longest_missing_streak"])

    def test_incomplete_row_details_are_deterministic_and_bounded(self):
        report = audit_panel_coverage(self.records, max_details=2)

        self.assertEqual(len(report["incomplete_rows"]), 2)
        self.assertEqual(report["incomplete_rows"][0]["date"], "2026-01-01")
        self.assertTrue(report["details_truncated"])
        self.assertEqual(report["omitted_incomplete_row_count"], 2)

    def test_invalid_records_and_settings_are_rejected(self):
        valid = [{"date": "2026-01-01", "A": 1.0}]
        invalid_cases = [
            ([], {}, "at least one"),
            ([{"date": "bad", "A": 1.0}], {}, "ISO date"),
            ([{"date": "2026-01-01"}], {}, "asset"),
            ([{"date": "2026-01-01", "A": True}], {}, "finite"),
            ([{"date": "2026-01-01", "A": float("inf")}], {}, "finite"),
            (valid, {"min_overall_coverage": 1.1}, "min_overall_coverage"),
            (valid, {"max_missing_streak": -1}, "max_missing_streak"),
            (valid, {"max_details": True}, "max_details"),
        ]
        for records, settings, message in invalid_cases:
            with self.subTest(records=records, settings=settings):
                with self.assertRaisesRegex(ValueError, message):
                    audit_panel_coverage(records, **settings)

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            audit_panel_coverage(valid + valid)
        with self.assertRaisesRegex(ValueError, "same asset columns"):
            audit_panel_coverage(
                valid + [{"date": "2026-01-02", "B": 2.0}]
            )

    def test_cli_loads_blank_cells_and_writes_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            panel = root / "panel.csv"
            output = root / "report.json"
            panel.write_text(
                "date,ALPHA,BETA\n"
                "2026-01-01,0.01,\n"
                "2026-01-02,,0.02\n"
                "2026-01-03,0.03,0.04\n",
                encoding="utf-8",
            )

            exit_code = main(
                [
                    str(panel),
                    "--min-overall-coverage",
                    "0.60",
                    "--output",
                    str(output),
                ]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertAlmostEqual(report["metrics"]["overall_coverage"], 2 / 3)

    def test_cli_uses_stable_failure_and_error_exit_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            panel = Path(directory) / "panel.csv"
            panel.write_text(
                "date,A\n2026-01-01,\n2026-01-02,1\n",
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main([str(panel), "--min-overall-coverage", "0.75"]), 1
                )
            original = panel.read_text(encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([str(panel), "--output", str(panel)]), 2)
                self.assertEqual(main([str(panel), "--max-file-bytes", "2"]), 2)
            self.assertEqual(panel.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
