import unittest

from quant_research_micro_lab.cost_sensitivity import analyze_cost_sensitivity


class CostSensitivityTests(unittest.TestCase):
    prices = [10, 10, 10, 11, 12, 13, 14, 13, 12, 11, 10]

    def test_cost_grid_is_sorted_and_reports_degradation(self):
        report = analyze_cost_sensitivity(
            self.prices,
            short_window=2,
            long_window=3,
            transaction_costs_bps=[500, 0, 100],
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["settings"]["transaction_costs_bps"], [0.0, 100.0, 500.0])
        returns = [scenario["total_return"] for scenario in report["scenarios"]]
        self.assertGreaterEqual(returns[0], returns[1])
        self.assertGreaterEqual(returns[1], returns[2])
        self.assertTrue(report["metrics"]["monotonic_nonincreasing"])
        self.assertAlmostEqual(
            report["metrics"]["return_degradation"], returns[0] - returns[-1]
        )
        self.assertEqual(report["scenarios"][0]["return_change_from_zero_cost"], 0.0)
        self.assertLess(report["scenarios"][-1]["return_change_from_zero_cost"], 0.0)

    def test_report_identifies_only_tested_nonpositive_boundary(self):
        report = analyze_cost_sensitivity(
            self.prices,
            short_window=2,
            long_window=3,
            transaction_costs_bps=[0, 100, 500],
        )

        first_nonpositive = report["metrics"]["first_nonpositive_tested_cost_bps"]
        bracket = report["metrics"]["positive_to_nonpositive_bracket"]
        self.assertEqual(first_nonpositive, 500.0)
        self.assertEqual(
            bracket,
            {"lower_tested_cost_bps": 100.0, "upper_tested_cost_bps": 500.0},
        )

    def test_gates_report_failures_in_stable_order(self):
        report = analyze_cost_sensitivity(
            self.prices,
            short_window=2,
            long_window=3,
            transaction_costs_bps=[0, 500],
            max_return_degradation=0.01,
            min_total_return_at_highest_cost=0.01,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(
            [failure["metric"] for failure in report["failures"]],
            ["return_degradation", "total_return_at_highest_cost"],
        )

    def test_flat_strategy_has_no_cost_sensitivity(self):
        report = analyze_cost_sensitivity(
            [10] * 8,
            short_window=2,
            long_window=3,
            transaction_costs_bps=[0, 1000],
        )

        self.assertEqual(report["metrics"]["return_degradation"], 0.0)
        self.assertEqual(report["metrics"]["first_nonpositive_tested_cost_bps"], 0.0)
        self.assertIsNone(report["metrics"]["positive_to_nonpositive_bracket"])

    def test_invalid_cost_grids_and_thresholds_are_rejected(self):
        for costs in ([], [10], [0, 0], [0, -1], [0, 10_000], [0, float("inf")], [0, True]):
            with self.subTest(costs=costs):
                with self.assertRaises(ValueError):
                    analyze_cost_sensitivity(
                        self.prices,
                        short_window=2,
                        long_window=3,
                        transaction_costs_bps=costs,
                    )
        for threshold in (True, -1, float("inf")):
            with self.subTest(threshold=threshold):
                with self.assertRaisesRegex(ValueError, "finite non-negative"):
                    analyze_cost_sensitivity(
                        self.prices,
                        short_window=2,
                        long_window=3,
                        transaction_costs_bps=[0, 10],
                        max_return_degradation=threshold,
                    )


if __name__ == "__main__":
    unittest.main()
