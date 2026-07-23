import unittest

import quant_research_micro_lab
from quant_research_micro_lab.bootstrap import bootstrap_equity_performance


class BootstrapPerformanceTests(unittest.TestCase):
    def test_bootstrap_api_is_available_from_package(self):
        self.assertIs(
            quant_research_micro_lab.bootstrap_equity_performance,
            bootstrap_equity_performance,
        )

    def test_constant_return_path_has_degenerate_intervals(self):
        report = bootstrap_equity_performance(
            [100.0, 110.0, 121.0, 133.1],
            block_size=1,
            samples=25,
            confidence=0.8,
            periods_per_year=3,
        )

        self.assertEqual(report["observations"], 4)
        self.assertEqual(report["return_observations"], 3)
        self.assertEqual(report["block_start_count"], 3)
        self.assertAlmostEqual(report["observed"]["total_return"], 0.331)
        self.assertAlmostEqual(
            report["intervals"]["total_return"]["lower"], 0.331
        )
        self.assertAlmostEqual(
            report["intervals"]["total_return"]["upper"], 0.331
        )
        self.assertAlmostEqual(
            report["intervals"]["annualized_volatility"]["upper"], 0.0
        )
        self.assertEqual(
            report["intervals"]["maximum_drawdown"],
            {"lower": 0.0, "upper": 0.0},
        )
        self.assertEqual(report["negative_total_return_rate"], 0.0)

    def test_seed_makes_mixed_path_report_reproducible(self):
        kwargs = {
            "block_size": 2,
            "samples": 50,
            "confidence": 0.9,
            "seed": 17,
            "periods_per_year": 4,
        }
        first = bootstrap_equity_performance(
            [100, 110, 99, 108.9, 87.12], **kwargs
        )
        second = bootstrap_equity_performance(
            [100, 110, 99, 108.9, 87.12], **kwargs
        )

        self.assertEqual(first, second)
        self.assertAlmostEqual(first["observed"]["maximum_drawdown"], -0.208)
        self.assertLessEqual(
            first["intervals"]["maximum_drawdown"]["lower"],
            first["intervals"]["maximum_drawdown"]["upper"],
        )
        self.assertGreater(first["negative_total_return_rate"], 0.0)

    def test_invalid_values_and_configuration_are_rejected(self):
        valid = [100, 101, 99]
        cases = [
            ([100], {}, "at least two"),
            ([100, 0], {}, "finite positive"),
            (valid, {"block_size": 3}, "return count"),
            (valid, {"block_size": True}, "positive integer"),
            (valid, {"samples": 0}, "positive integer"),
            (valid, {"confidence": 1.0}, "between 0 and 1"),
            (valid, {"seed": True}, "seed"),
            (valid, {"periods_per_year": 0}, "positive integer"),
        ]
        for equity, kwargs, message in cases:
            with self.subTest(equity=equity, kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    bootstrap_equity_performance(equity, **kwargs)


if __name__ == "__main__":
    unittest.main()
