import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import quant_research_micro_lab
from quant_research_micro_lab.benchmark import compare_to_benchmark, main


def _curve(returns):
    values = [100.0]
    for value in returns:
        values.append(values[-1] * (1.0 + value))
    return values


class BenchmarkComparisonTests(unittest.TestCase):
    def test_benchmark_api_is_available_from_package(self):
        self.assertIs(
            quant_research_micro_lab.compare_to_benchmark,
            compare_to_benchmark,
        )

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

    def test_cli_compares_aligned_equity_and_price_files(self):
        with tempfile.TemporaryDirectory() as directory:
            strategy = Path(directory) / "strategy.csv"
            benchmark = Path(directory) / "benchmark.csv"
            strategy.write_text(
                "date,equity,gross_equity\n"
                "2026-01-01,1.0,1.0\n"
                "2026-01-02,1.1,1.2\n"
                "2026-01-03,1.045,1.08\n",
                encoding="utf-8",
            )
            benchmark.write_text(
                "date,close\n"
                "2026-01-01,100\n"
                "2026-01-02,105\n"
                "2026-01-03,102.9\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        str(strategy),
                        str(benchmark),
                        "--periods-per-year",
                        "12",
                    ]
                )

            report = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(report["start_date"], "2026-01-01")
            self.assertEqual(report["end_date"], "2026-01-03")
            self.assertEqual(report["strategy_column"], "equity")
            self.assertEqual(report["observations"], 3)

    def test_cli_rejects_date_mismatches(self):
        with tempfile.TemporaryDirectory() as directory:
            strategy = Path(directory) / "strategy.csv"
            benchmark = Path(directory) / "benchmark.csv"
            strategy.write_text(
                "date,equity,gross_equity\n2026-01-01,1.0,1.0\n2026-01-02,1.1,1.1\n",
                encoding="utf-8",
            )
            benchmark.write_text(
                "date,close\n2026-01-01,100\n2026-01-03,101\n",
                encoding="utf-8",
            )

            with contextlib.redirect_stderr(io.StringIO()):
                exit_code = main([str(strategy), str(benchmark)])

            self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()
