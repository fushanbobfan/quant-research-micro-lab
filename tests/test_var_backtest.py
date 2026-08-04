import math
import unittest

from quant_research_micro_lab.var_backtest import backtest_var_forecasts


class VarBacktestTests(unittest.TestCase):
    def setUp(self):
        returns = [-0.06, 0.01, -0.03, 0.02, 0.00, -0.08, 0.03, -0.01, 0.01, -0.04]
        self.records = [
            {
                "date": f"2026-01-{index:02d}",
                "realized_return": realized_return,
                "var": 0.05,
            }
            for index, realized_return in enumerate(returns, start=1)
        ]

    def test_report_contains_exception_severity_and_coverage(self):
        report = backtest_var_forecasts(
            self.records,
            confidence=0.8,
            max_details=1,
        )

        metrics = report["metrics"]
        self.assertEqual(metrics["observations"], 10)
        self.assertEqual(metrics["exception_count"], 2)
        self.assertAlmostEqual(metrics["exception_rate"], 0.2)
        self.assertAlmostEqual(metrics["expected_exception_count"], 2.0)
        self.assertAlmostEqual(metrics["mean_exception_loss"], 0.07)
        self.assertAlmostEqual(metrics["mean_exception_shortfall"], 0.02)
        self.assertAlmostEqual(metrics["maximum_exception_loss"], 0.08)
        self.assertEqual(metrics["longest_exception_streak"], 1)
        self.assertEqual(metrics["adjacent_exception_pairs"], 0)
        self.assertAlmostEqual(metrics["kupiec_likelihood_ratio"], 0.0)
        self.assertAlmostEqual(metrics["kupiec_p_value"], 1.0)
        self.assertEqual(report["exceptions"][0]["date"], "2026-01-01")
        self.assertTrue(report["details_truncated"])

    def test_value_equal_to_var_is_not_an_exception(self):
        report = backtest_var_forecasts(
            [
                {"date": "2026-01-01", "realized_return": -0.05, "var": 0.05},
                {"date": "2026-01-02", "realized_return": 0.01, "var": 0.05},
            ],
            confidence=0.95,
        )

        self.assertEqual(report["metrics"]["exception_count"], 0)
        self.assertIsNone(report["metrics"]["mean_exception_loss"])
        self.assertTrue(math.isfinite(report["metrics"]["kupiec_p_value"]))

    def test_threshold_failures_are_reported_in_stable_order(self):
        report = backtest_var_forecasts(
            self.records[:5],
            confidence=0.99,
            max_exception_rate=0.1,
            min_kupiec_p_value=0.1,
            max_exception_count=0,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(
            [failure["metric"] for failure in report["failures"]],
            ["exception_rate", "kupiec_p_value", "exception_count"],
        )

    def test_invalid_records_and_configuration_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            backtest_var_forecasts([])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            backtest_var_forecasts([self.records[1], self.records[0]])
        for record in (
            {"date": "01/01/2026", "realized_return": 0.0, "var": 0.05},
            {"date": "2026-01-01", "realized_return": float("nan"), "var": 0.05},
            {"date": "2026-01-01", "realized_return": -1.1, "var": 0.05},
            {"date": "2026-01-01", "realized_return": 0.0, "var": -0.01},
            {"date": "2026-01-01", "realized_return": 0.0, "var": True},
        ):
            with self.subTest(record=record):
                with self.assertRaises(ValueError):
                    backtest_var_forecasts([record])
        for confidence in (0.0, 1.0, True, float("inf")):
            with self.subTest(confidence=confidence):
                with self.assertRaisesRegex(ValueError, "confidence"):
                    backtest_var_forecasts(self.records, confidence=confidence)


if __name__ == "__main__":
    unittest.main()
