import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import quant_research_micro_lab
from quant_research_micro_lab.stress import (
    evaluate_portfolio_stress,
    load_scenario_csv,
    main,
)


class PortfolioStressTests(unittest.TestCase):
    def setUp(self):
        self.positions = [
            {"date": "2026-01-01", "asset": "AAA", "weight": 0.6},
            {"date": "2026-01-01", "asset": "BBB", "weight": 0.4},
            {"date": "2026-02-01", "asset": "AAA", "weight": 0.7},
            {"date": "2026-02-01", "asset": "BBB", "weight": -0.2},
            {"date": "2026-02-01", "asset": "CASH", "weight": 0.1},
        ]
        self.scenarios = [
            {"scenario": "broad selloff", "asset": "AAA", "return": -0.2},
            {"scenario": "broad selloff", "asset": "BBB", "return": -0.1},
            {"scenario": "broad selloff", "asset": "CASH", "return": 0.0},
            {"scenario": "rotation", "asset": "AAA", "return": -0.05},
            {"scenario": "rotation", "asset": "BBB", "return": 0.15},
            {"scenario": "rotation", "asset": "CASH", "return": 0.0},
        ]

    def test_reports_contributions_for_the_latest_snapshot(self):
        report = evaluate_portfolio_stress(self.positions, self.scenarios)

        self.assertIs(
            quant_research_micro_lab.evaluate_portfolio_stress,
            evaluate_portfolio_stress,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["portfolio"]["date"], "2026-02-01")
        self.assertAlmostEqual(report["portfolio"]["gross_exposure"], 1.0)
        self.assertAlmostEqual(report["portfolio"]["net_exposure"], 0.6)
        self.assertEqual(report["scenario_count"], 2)
        self.assertEqual(
            [item["scenario"] for item in report["scenarios"]],
            ["broad selloff", "rotation"],
        )

        selloff = report["scenarios"][0]
        self.assertAlmostEqual(selloff["portfolio_return"], -0.12)
        self.assertAlmostEqual(selloff["long_contribution"], -0.14)
        self.assertAlmostEqual(selloff["short_contribution"], 0.02)
        self.assertEqual(
            selloff["largest_negative_contributor"]["asset"],
            "AAA",
        )
        self.assertEqual(
            selloff["largest_positive_contributor"]["asset"],
            "BBB",
        )
        self.assertEqual(report["summary"]["worst_scenario"]["scenario"], "broad selloff")

    def test_explicit_snapshot_date_selects_an_earlier_portfolio(self):
        scenarios = [
            {"scenario": "down", "asset": "AAA", "return": -0.1},
            {"scenario": "down", "asset": "BBB", "return": -0.2},
        ]

        report = evaluate_portfolio_stress(
            self.positions,
            scenarios,
            snapshot_date="2026-01-01",
        )

        self.assertEqual(report["portfolio"]["asset_count"], 2)
        self.assertAlmostEqual(report["scenarios"][0]["portfolio_return"], -0.14)

    def test_loss_threshold_failures_are_sorted_by_scenario(self):
        report = evaluate_portfolio_stress(
            self.positions,
            self.scenarios,
            max_loss=0.04,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(
            [failure["scenario"] for failure in report["failures"]],
            ["broad selloff", "rotation"],
        )
        self.assertAlmostEqual(report["failures"][0]["excess"], 0.08)

    def test_scenario_assets_must_match_nonzero_portfolio_assets(self):
        with self.assertRaisesRegex(ValueError, "assets must match"):
            evaluate_portfolio_stress(
                self.positions,
                [{"scenario": "partial", "asset": "AAA", "return": -0.1}],
            )

    def test_invalid_scenarios_and_thresholds_are_rejected(self):
        cases = [
            ([], "at least one"),
            ([3], "object"),
            (
                [{"scenario": "", "asset": "AAA", "return": 0.0}],
                "non-empty",
            ),
            (
                [{"scenario": "x", "asset": "", "return": 0.0}],
                "non-empty",
            ),
            (
                [{"scenario": "x", "asset": "AAA", "return": -1.1}],
                "at least -1",
            ),
            (
                [
                    {"scenario": "x", "asset": "AAA", "return": 0.0},
                    {"scenario": "x", "asset": "AAA", "return": 0.1},
                ],
                "duplicate",
            ),
        ]
        for scenarios, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    evaluate_portfolio_stress(self.positions, scenarios)
        for max_loss in (-0.1, True, float("inf"), "low"):
            with self.subTest(max_loss=max_loss):
                with self.assertRaisesRegex(ValueError, "max_loss"):
                    evaluate_portfolio_stress(
                        self.positions,
                        self.scenarios,
                        max_loss=max_loss,
                    )

    def test_loads_strict_scenario_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "scenarios.csv"
            dataset.write_text(
                "scenario,asset,return\n"
                "down,AAA,-0.1\n"
                "down,BBB,0.2\n",
                encoding="utf-8",
            )

            self.assertEqual(
                load_scenario_csv(dataset),
                [
                    {"scenario": "down", "asset": "AAA", "return": -0.1},
                    {"scenario": "down", "asset": "BBB", "return": 0.2},
                ],
            )

    def test_scenario_loader_rejects_bad_shapes_and_values(self):
        cases = [
            ("wrong,header\n", "header"),
            ("scenario,asset,return\n", "at least one"),
            ("scenario,asset,return\ndown,AAA\n", "three fields"),
            ("scenario,asset,return\ndown,AAA,0.1,extra\n", "three fields"),
            ("scenario,asset,return\ndown,AAA,nope\n", "return"),
        ]
        for contents, message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as directory:
                    dataset = Path(directory) / "scenarios.csv"
                    dataset.write_text(contents, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        load_scenario_csv(dataset)

    def test_cli_returns_one_and_writes_a_failed_gate_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            portfolio = root / "portfolio.csv"
            scenarios = root / "scenarios.csv"
            output = root / "report.json"
            portfolio.write_text(
                "date,asset,weight\n"
                "2026-01-01,AAA,0.6\n"
                "2026-01-01,BBB,0.4\n",
                encoding="utf-8",
            )
            scenarios.write_text(
                "scenario,asset,return\n"
                "down,AAA,-0.2\n"
                "down,BBB,-0.1\n",
                encoding="utf-8",
            )

            exit_code = main(
                [
                    str(portfolio),
                    str(scenarios),
                    "--max-loss",
                    "0.1",
                    "--output",
                    str(output),
                ]
            )

            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 1)
            self.assertFalse(report["passed"])
            self.assertAlmostEqual(report["failures"][0]["actual"], 0.16)

    def test_cli_prints_a_passing_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            portfolio = root / "portfolio.csv"
            scenarios = root / "scenarios.csv"
            portfolio.write_text(
                "date,asset,weight\n"
                "2026-01-01,AAA,0.6\n"
                "2026-01-01,BBB,0.4\n",
                encoding="utf-8",
            )
            scenarios.write_text(
                "scenario,asset,return\n"
                "up,AAA,0.1\n"
                "up,BBB,0.2\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = main([str(portfolio), str(scenarios)])

            self.assertEqual(exit_code, 0)
            self.assertTrue(json.loads(stdout.getvalue())["passed"])

    def test_cli_returns_two_for_invalid_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            portfolio = root / "portfolio.csv"
            scenarios = root / "scenarios.csv"
            portfolio.write_text(
                "date,asset,weight\n2026-01-01,AAA,1\n",
                encoding="utf-8",
            )
            scenarios.write_text("wrong,header\n", encoding="utf-8")

            with contextlib.redirect_stderr(io.StringIO()):
                exit_code = main([str(portfolio), str(scenarios)])

            self.assertEqual(exit_code, 2)

    def test_cli_refuses_to_overwrite_either_source_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            portfolio = root / "portfolio.csv"
            scenarios = root / "scenarios.csv"
            portfolio_contents = "date,asset,weight\n2026-01-01,AAA,1\n"
            scenario_contents = "scenario,asset,return\nup,AAA,0.1\n"
            portfolio.write_text(portfolio_contents, encoding="utf-8")
            scenarios.write_text(scenario_contents, encoding="utf-8")

            for output in (portfolio, scenarios):
                with self.subTest(output=output.name):
                    with contextlib.redirect_stderr(io.StringIO()):
                        exit_code = main(
                            [
                                str(portfolio),
                                str(scenarios),
                                "--output",
                                str(output),
                            ]
                        )
                    self.assertEqual(exit_code, 2)

            self.assertEqual(
                portfolio.read_text(encoding="utf-8"),
                portfolio_contents,
            )
            self.assertEqual(
                scenarios.read_text(encoding="utf-8"),
                scenario_contents,
            )


if __name__ == "__main__":
    unittest.main()
