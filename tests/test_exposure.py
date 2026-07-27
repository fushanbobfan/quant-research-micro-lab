import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import quant_research_micro_lab
from quant_research_micro_lab.exposure import (
    audit_portfolio_exposure,
    load_portfolio_csv,
    main,
)


class PortfolioExposureTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"date": "2026-01-01", "asset": "AAA", "weight": 0.6},
            {"date": "2026-01-01", "asset": "BBB", "weight": 0.4},
            {"date": "2026-01-02", "asset": "AAA", "weight": 0.5},
            {"date": "2026-01-02", "asset": "BBB", "weight": 0.25},
            {"date": "2026-01-02", "asset": "CCC", "weight": 0.25},
            {"date": "2026-01-03", "asset": "AAA", "weight": 0.4},
            {"date": "2026-01-03", "asset": "BBB", "weight": -0.2},
            {"date": "2026-01-03", "asset": "CCC", "weight": 0.3},
        ]

    def test_reports_dated_exposure_concentration_and_turnover(self):
        report = audit_portfolio_exposure(self.records)

        self.assertIs(
            quant_research_micro_lab.audit_portfolio_exposure,
            audit_portfolio_exposure,
        )
        self.assertIs(
            quant_research_micro_lab.load_portfolio_csv,
            load_portfolio_csv,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["snapshot_count"], 3)
        self.assertEqual(report["asset_count"], 3)
        self.assertAlmostEqual(
            report["summary"]["average_gross_exposure"],
            2.9 / 3,
        )
        self.assertAlmostEqual(
            report["summary"]["average_abs_net_exposure"],
            5 / 6,
        )
        self.assertAlmostEqual(report["summary"]["average_turnover"], 0.275)
        self.assertEqual(
            report["extrema"]["maximum_gross_exposure"],
            {"date": "2026-01-01", "value": 1.0},
        )
        self.assertEqual(
            report["extrema"]["maximum_single_position"],
            {
                "date": "2026-01-01",
                "asset": "AAA",
                "weight": 0.6,
                "absolute_weight": 0.6,
            },
        )
        self.assertAlmostEqual(
            report["extrema"]["maximum_concentration_hhi"]["value"],
            0.52,
        )
        self.assertEqual(
            report["extrema"]["maximum_turnover"]["date"],
            "2026-01-03",
        )
        self.assertAlmostEqual(
            report["extrema"]["maximum_turnover"]["value"],
            0.3,
        )
        final = report["snapshots"][-1]
        self.assertAlmostEqual(final["long_exposure"], 0.7)
        self.assertAlmostEqual(final["short_exposure"], 0.2)
        self.assertAlmostEqual(final["net_exposure"], 0.5)
        self.assertAlmostEqual(final["concentration_hhi"], 29 / 81)
        self.assertAlmostEqual(final["effective_positions"], 81 / 29)

    def test_single_snapshot_has_no_turnover_extreme(self):
        report = audit_portfolio_exposure(self.records[:2])

        self.assertIsNone(report["summary"]["average_turnover"])
        self.assertIsNone(report["extrema"]["maximum_turnover"])
        self.assertIsNone(report["snapshots"][0]["turnover_from_previous"])

    def test_threshold_failures_are_reported_in_stable_order(self):
        report = audit_portfolio_exposure(
            self.records,
            max_gross_exposure=0.9,
            max_abs_net_exposure=0.8,
            max_single_position=0.5,
            max_concentration_hhi=0.5,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(
            [failure["metric"] for failure in report["failures"]],
            [
                "gross_exposure",
                "abs_net_exposure",
                "single_position",
                "concentration_hhi",
            ],
        )
        self.assertEqual(report["failures"][0]["date"], "2026-01-01")
        self.assertAlmostEqual(report["failures"][3]["excess"], 0.02)

    def test_largest_position_ties_are_resolved_by_asset(self):
        report = audit_portfolio_exposure(
            [
                {"date": "2026-01-01", "asset": "BBB", "weight": -0.5},
                {"date": "2026-01-01", "asset": "AAA", "weight": 0.5},
            ]
        )

        self.assertEqual(
            report["snapshots"][0]["largest_position_asset"],
            "AAA",
        )

    def test_invalid_records_are_rejected(self):
        cases = [
            ([], "at least one"),
            ([3], "object"),
            (
                [{"date": "2026/01/01", "asset": "AAA", "weight": 1}],
                "ISO date",
            ),
            (
                [
                    {"date": "2026-01-02", "asset": "AAA", "weight": 1},
                    {"date": "2026-01-01", "asset": "AAA", "weight": 1},
                ],
                "non-decreasing",
            ),
            (
                [{"date": "2026-01-01", "asset": "", "weight": 1}],
                "non-empty",
            ),
            (
                [{"date": "2026-01-01", "asset": "AAA", "weight": True}],
                "finite",
            ),
            (
                [
                    {"date": "2026-01-01", "asset": "AAA", "weight": 1},
                    {"date": "2026-01-01", "asset": "AAA", "weight": 0},
                ],
                "duplicate",
            ),
            (
                [
                    {"date": "2026-01-01", "asset": "AAA", "weight": 0},
                    {"date": "2026-01-01", "asset": "BBB", "weight": 0},
                ],
                "non-zero",
            ),
        ]
        for records, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    audit_portfolio_exposure(records)

    def test_invalid_thresholds_are_rejected(self):
        for value in (-0.1, True, float("inf"), "high"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "max_gross_exposure"):
                    audit_portfolio_exposure(
                        self.records,
                        max_gross_exposure=value,
                    )
        with self.assertRaisesRegex(ValueError, "max_concentration_hhi"):
            audit_portfolio_exposure(
                self.records,
                max_concentration_hhi=1.1,
            )

    def test_loads_strict_portfolio_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "weights.csv"
            dataset.write_text(
                "date,asset,weight\n"
                "2026-01-01,AAA,0.6\n"
                "2026-01-01,BBB,0.4\n",
                encoding="utf-8",
            )

            self.assertEqual(
                load_portfolio_csv(dataset),
                [
                    {"date": "2026-01-01", "asset": "AAA", "weight": 0.6},
                    {"date": "2026-01-01", "asset": "BBB", "weight": 0.4},
                ],
            )

    def test_loader_rejects_bad_headers_empty_rows_and_weights(self):
        cases = [
            ("wrong,header\n", "header"),
            ("date,asset,weight\n", "at least one"),
            ("date,asset,weight\n2026-01-01,AAA\n", "three fields"),
            ("date,asset,weight\n2026-01-01,AAA,1,extra\n", "three fields"),
            ("date,asset,weight\n2026-01-01,AAA,nope\n", "weight"),
        ]
        for contents, message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as directory:
                    dataset = Path(directory) / "weights.csv"
                    dataset.write_text(contents, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        load_portfolio_csv(dataset)

    def test_cli_returns_zero_for_a_passing_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "weights.csv"
            dataset.write_text(
                "date,asset,weight\n"
                "2026-01-01,AAA,0.6\n"
                "2026-01-01,BBB,0.4\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        str(dataset),
                        "--max-gross-exposure",
                        "1",
                        "--max-single-position",
                        "0.6",
                        "--max-concentration-hhi",
                        "0.52",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(json.loads(stdout.getvalue())["passed"])

    def test_cli_returns_one_with_structured_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "weights.csv"
            dataset.write_text(
                "date,asset,weight\n"
                "2026-01-01,AAA,0.8\n"
                "2026-01-01,BBB,0.2\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        str(dataset),
                        "--max-single-position",
                        "0.6",
                        "--max-concentration-hhi",
                        "0.6",
                    ]
                )

            report = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 1)
            self.assertEqual(
                [failure["metric"] for failure in report["failures"]],
                ["single_position", "concentration_hhi"],
            )

    def test_cli_returns_two_for_invalid_input(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "weights.csv"
            dataset.write_text("wrong,header\n", encoding="utf-8")

            with contextlib.redirect_stderr(io.StringIO()):
                exit_code = main([str(dataset)])

            self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()
