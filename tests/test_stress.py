import unittest

import quant_research_micro_lab
from quant_research_micro_lab.stress import evaluate_portfolio_stress


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


if __name__ == "__main__":
    unittest.main()
