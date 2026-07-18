import math
import unittest

from quant_research_micro_lab.benchmark import compare_to_benchmark


def _curve(returns):
    values = [100.0]
    for value in returns:
        values.append(values[-1] * (1.0 + value))
    return values


class BenchmarkComparisonTests(unittest.TestCase):
    def test_reports_relative_performance_and_exposure_metrics(self):
        benchmark_returns = [0.10, -0.05, 0.02]
        strategy_returns = [0.20, -0.10, 0.04]

        report = compare_to_benchmark(
            _curve(strategy_returns),
            _curve(benchmark_returns),
            periods_per_year=12,
        )

        self.assertEqual(report["observations"], 4)
        self.assertEqual(report["return_observations"], 3)
        self.assertAlmostEqual(report["beta"], 2.0)
        self.assertAlmostEqual(report["correlation"], 1.0)
        self.assertAlmostEqual(report["active_return_hit_rate"], 2 / 3)
        benchmark_variance = sum(
            (value - sum(benchmark_returns) / 3) ** 2
            for value in benchmark_returns
        ) / 3
        expected_tracking_error = math.sqrt(benchmark_variance) * math.sqrt(12)
        self.assertAlmostEqual(report["tracking_error"], expected_tracking_error)
        self.assertAlmostEqual(
            report["information_ratio"],
            (sum(benchmark_returns) / 3) * 12 / expected_tracking_error,
        )

    def test_relative_return_uses_growth_ratios(self):
        report = compare_to_benchmark([100, 121], [100, 110])

        self.assertAlmostEqual(report["strategy_total_return"], 0.21)
        self.assertAlmostEqual(report["benchmark_total_return"], 0.10)
        self.assertAlmostEqual(report["relative_total_return"], 0.10)

    def test_constant_returns_make_ratio_metrics_explicitly_undefined(self):
        report = compare_to_benchmark(
            _curve([0.05, 0.05, 0.05]),
            _curve([0.05, 0.05, 0.05]),
        )

        self.assertEqual(report["tracking_error"], 0.0)
        self.assertIsNone(report["information_ratio"])
        self.assertIsNone(report["beta"])
        self.assertIsNone(report["correlation"])

    def test_rejects_unaligned_or_insufficient_series(self):
        with self.assertRaisesRegex(ValueError, "same number"):
            compare_to_benchmark([1.0, 1.1], [1.0])
        with self.assertRaisesRegex(ValueError, "at least two"):
            compare_to_benchmark([1.0], [1.0])

    def test_rejects_invalid_values_and_annualization(self):
        for values in ([1.0, 0.0], [1.0, float("nan")], [1.0, True]):
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "finite positive"):
                    compare_to_benchmark(values, [1.0, 1.1])
        for periods in (0, -1, 12.5, True):
            with self.subTest(periods=periods):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    compare_to_benchmark(
                        [1.0, 1.1],
                        [1.0, 1.05],
                        periods_per_year=periods,
                    )


if __name__ == "__main__":
    unittest.main()
