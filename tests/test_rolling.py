import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import quant_research_micro_lab
from quant_research_micro_lab.rolling import analyze_rolling_performance, main


class RollingPerformanceTests(unittest.TestCase):
    def test_rolling_api_is_available_from_package(self):
        self.assertIs(
            quant_research_micro_lab.analyze_rolling_performance,
            analyze_rolling_performance,
        )

    def test_reports_overlapping_windows_and_stable_extremes(self):
        report = analyze_rolling_performance(
            [8, 16, 32, 8, 16],
            window=2,
            periods_per_year=2,
        )

        self.assertEqual(report["window_count"], 3)
        first = report["windows"][0]
        self.assertEqual(first["start_index"], 0)
        self.assertEqual(first["end_index"], 2)
        self.assertEqual(first["total_return"], 3.0)
        self.assertEqual(first["annualized_return"], 3.0)
        self.assertEqual(first["annualized_volatility"], 0.0)
        self.assertIsNone(first["annualized_sharpe"])
        self.assertEqual(first["maximum_drawdown"], 0.0)

        middle = report["windows"][1]
        self.assertEqual(middle["total_return"], -0.5)
        self.assertAlmostEqual(
            middle["annualized_volatility"], math.sqrt(1.53125)
        )
        self.assertEqual(middle["maximum_drawdown"], -0.75)
        self.assertEqual(middle["positive_return_rate"], 0.5)
        self.assertEqual(report["worst_total_return_window"]["start_index"], 1)
        self.assertEqual(report["worst_drawdown_window"]["start_index"], 1)
        self.assertEqual(report["highest_volatility_window"]["start_index"], 1)
        self.assertEqual(report["latest_window"]["start_index"], 2)

    def test_downside_and_sharpe_use_zero_as_the_reference_return(self):
        report = analyze_rolling_performance(
            [100, 110, 99],
            window=2,
            periods_per_year=2,
        )
        window = report["windows"][0]

        self.assertAlmostEqual(window["annualized_downside_deviation"], 0.1)
        self.assertAlmostEqual(window["annualized_sharpe"], 0.0)
        self.assertAlmostEqual(window["worst_period_return"], -0.1)

    def test_invalid_values_and_configuration_are_rejected(self):
        cases = [
            ([1, 2], {"window": 0}, "window"),
            ([1, 2], {"window": 1, "periods_per_year": True}, "periods_per_year"),
            ([1, 2], {"window": 2}, r"window \+ 1"),
            ([1, 0], {"window": 1}, "finite positive"),
            ([1, float("inf")], {"window": 1}, "finite positive"),
        ]
        for equity, kwargs, message in cases:
            with self.subTest(equity=equity, kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    analyze_rolling_performance(equity, **kwargs)

    def test_cli_adds_dates_and_writes_a_json_report(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "equity.csv"
            output = Path(directory) / "rolling.json"
            dataset.write_text(
                "date,equity,gross_equity\n"
                "2026-01-01,100,100\n"
                "2026-01-02,110,111\n"
                "2026-01-03,99,100\n"
                "2026-01-04,108.9,110\n",
                encoding="utf-8",
            )

            exit_code = main(
                [
                    str(dataset),
                    "--window",
                    "2",
                    "--periods-per-year",
                    "2",
                    "--output",
                    str(output),
                ]
            )

            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(report["window_count"], 2)
            self.assertEqual(report["windows"][0]["start_date"], "2026-01-01")
            self.assertEqual(report["latest_window"]["end_date"], "2026-01-04")
            self.assertEqual(report["column"], "equity")

    def test_cli_returns_two_for_an_oversized_window(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "equity.csv"
            dataset.write_text(
                "date,equity,gross_equity\n2026-01-01,100,100\n",
                encoding="utf-8",
            )

            with contextlib.redirect_stderr(io.StringIO()):
                exit_code = main([str(dataset), "--window", "2"])

            self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()
