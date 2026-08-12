import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import quant_research_micro_lab
from quant_research_micro_lab.price_audit import audit_price_series, main


class PriceAuditTests(unittest.TestCase):
    def setUp(self):
        self.dates = [
            "2026-01-01",
            "2026-01-02",
            "2026-01-05",
            "2026-01-06",
            "2026-01-07",
            "2026-01-08",
        ]
        self.prices = [100.0, 100.0, 100.0, 150.0, 75.0, 75.0]

    def test_public_api_and_diagnostics(self):
        self.assertIs(quant_research_micro_lab.audit_price_series, audit_price_series)
        report = audit_price_series(self.dates, self.prices)

        self.assertTrue(report["passed"])
        self.assertEqual(report["metrics"]["observations"], 6)
        self.assertEqual(report["metrics"]["return_observations"], 5)
        self.assertEqual(report["metrics"]["maximum_calendar_gap"]["calendar_gap_days"], 3)
        self.assertEqual(report["metrics"]["calendar_gaps_over_one_day"], 1)
        self.assertEqual(report["metrics"]["unchanged_return_count"], 3)
        self.assertEqual(report["metrics"]["unchanged_run_count"], 2)
        self.assertEqual(
            report["metrics"]["longest_unchanged_run"]["unchanged_transitions"],
            2,
        )
        self.assertAlmostEqual(
            report["metrics"]["maximum_absolute_return"]["absolute_return"],
            0.5,
        )

    def test_all_threshold_failures_are_reported(self):
        report = audit_price_series(
            self.dates,
            self.prices,
            max_calendar_gap_days=2,
            max_unchanged_run=1,
            max_abs_return=0.4,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(
            [failure["metric"] for failure in report["failures"]],
            [
                "maximum_calendar_gap_days",
                "longest_unchanged_run",
                "maximum_absolute_return",
            ],
        )

    def test_detail_lists_are_deterministic_and_bounded(self):
        report = audit_price_series(self.dates, self.prices, max_details=1)

        self.assertEqual(
            report["largest_calendar_gaps"],
            [
                {
                    "start_date": "2026-01-02",
                    "end_date": "2026-01-05",
                    "calendar_gap_days": 3,
                }
            ],
        )
        self.assertEqual(
            report["unchanged_runs"][0]["unchanged_transitions"], 2
        )
        self.assertEqual(
            report["largest_absolute_returns"][0]["end_date"], "2026-01-06"
        )
        self.assertEqual(
            report["details_truncated"],
            {
                "calendar_gaps": True,
                "unchanged_runs": True,
                "absolute_returns": True,
            },
        )

    def test_no_unchanged_prices_is_explicit(self):
        report = audit_price_series(
            ["2026-02-01", "2026-02-02", "2026-02-03"],
            [10.0, 11.0, 12.0],
        )

        self.assertEqual(report["metrics"]["unchanged_return_count"], 0)
        self.assertEqual(report["metrics"]["unchanged_run_count"], 0)
        self.assertIsNone(report["metrics"]["longest_unchanged_run"])

    def test_input_shape_dates_and_prices_are_validated(self):
        invalid = [
            (["2026-01-01"], [1.0]),
            (["2026-01-01", "2026-01-02"], [1.0]),
            (["bad", "2026-01-02"], [1.0, 2.0]),
            (["2026-01-02", "2026-01-01"], [1.0, 2.0]),
            (["2026-01-01", "2026-01-02"], [1.0, 0.0]),
            (["2026-01-01", "2026-01-02"], [1.0, float("inf")]),
        ]
        for dates, prices in invalid:
            with self.subTest(dates=dates, prices=prices), self.assertRaises(ValueError):
                audit_price_series(dates, prices)

    def test_thresholds_are_validated(self):
        invalid_settings = [
            {"max_calendar_gap_days": 0},
            {"max_calendar_gap_days": True},
            {"max_unchanged_run": -1},
            {"max_abs_return": -0.1},
            {"max_abs_return": float("inf")},
            {"max_details": True},
        ]
        for settings in invalid_settings:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                audit_price_series(self.dates, self.prices, **settings)

    def test_cli_reads_strict_price_csv_and_writes_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "prices.csv"
            output = root / "audit.json"
            dataset.write_text(
                "date,close\n"
                "2026-01-01,100\n"
                "2026-01-02,100\n"
                "2026-01-05,100\n"
                "2026-01-06,150\n",
                encoding="utf-8",
            )

            exit_code = main(
                [
                    str(dataset),
                    "--max-calendar-gap-days",
                    "3",
                    "--max-unchanged-run",
                    "2",
                    "--max-abs-return",
                    "0.5",
                    "--output",
                    str(output),
                ]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["metrics"]["start_date"], "2026-01-01")
            self.assertEqual(
                report["metrics"]["maximum_calendar_gap"]["calendar_gap_days"],
                3,
            )

    def test_cli_gate_failure_and_alias_use_stable_exit_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "prices.csv"
            dataset.write_text(
                "date,close\n2026-01-01,100\n2026-01-05,100\n",
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main([str(dataset), "--max-calendar-gap-days", "2"]), 1
                )
            original = dataset.read_text(encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(
                    main([str(dataset), "--output", str(dataset)]), 2
                )
            self.assertEqual(dataset.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
