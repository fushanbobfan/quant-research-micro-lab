import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import quant_research_micro_lab
from quant_research_micro_lab.risk import analyze_drawdowns, load_equity_csv, main


class DrawdownAnalysisTests(unittest.TestCase):
    def test_risk_apis_are_available_from_package(self):
        self.assertIs(quant_research_micro_lab.analyze_drawdowns, analyze_drawdowns)
        self.assertIs(quant_research_micro_lab.load_equity_csv, load_equity_csv)

    def test_reports_recovered_and_open_drawdown_episodes(self):
        report = analyze_drawdowns([1.0, 0.9, 0.8, 1.0, 1.1, 1.0, 0.99])

        self.assertEqual(report["episode_count"], 2)
        first, second = report["episodes"]
        self.assertEqual(
            (first["peak_index"], first["trough_index"], first["recovery_index"]),
            (0, 2, 3),
        )
        self.assertTrue(first["recovered"])
        self.assertAlmostEqual(first["depth"], -0.2)
        self.assertEqual(first["underwater_observations"], 2)
        self.assertIsNone(second["recovery_index"])
        self.assertFalse(second["recovered"])
        self.assertIs(report["maximum_drawdown"], first)
        self.assertIs(report["longest_underwater"], first)
        self.assertAlmostEqual(report["current_drawdown"], -0.1)

    def test_new_highs_without_drawdowns_produce_an_empty_episode_list(self):
        report = analyze_drawdowns([1.0, 1.0, 1.2])

        self.assertEqual(report["episode_count"], 0)
        self.assertEqual(report["current_drawdown"], 0.0)
        self.assertIsNone(report["maximum_drawdown"])
        self.assertIsNone(report["longest_underwater"])

    def test_open_episode_counts_observations_after_the_peak(self):
        report = analyze_drawdowns([1.0, 0.9, 0.8])

        self.assertEqual(
            report["episodes"][0]["underwater_observations"],
            2,
        )

    def test_empty_or_invalid_equity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            analyze_drawdowns([])
        for values in ([1.0, 0.0], [1.0, float("nan")], [1.0, True], [1.0, "x"]):
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "finite positive"):
                    analyze_drawdowns(values)

    def test_loads_net_or_gross_equity_from_backtest_export(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "equity.csv"
            dataset.write_text(
                "date,equity,gross_equity\n"
                "2026-01-01,1.0,1.0\n"
                "2026-01-02,0.9,0.95\n",
                encoding="utf-8",
            )

            dates, gross = load_equity_csv(dataset, "gross_equity")

            self.assertEqual(dates, ["2026-01-01", "2026-01-02"])
            self.assertEqual(gross, [1.0, 0.95])

    def test_loader_rejects_bad_headers_dates_and_values(self):
        cases = [
            ("date,equity\n2026-01-01,1\n", "header"),
            (
                "date,equity,gross_equity\n2026-01-02,1,1\n2026-01-01,1,1\n",
                "date",
            ),
            ("date,equity,gross_equity\n2026-01-01,nan,1\n", "equity"),
        ]
        for contents, message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as directory:
                    dataset = Path(directory) / "equity.csv"
                    dataset.write_text(contents, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        load_equity_csv(dataset)

    def test_cli_adds_dates_to_drawdown_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "equity.csv"
            dataset.write_text(
                "date,equity,gross_equity\n"
                "2026-01-01,1.0,1.0\n"
                "2026-01-02,0.8,0.9\n"
                "2026-01-03,1.0,1.1\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = main([str(dataset)])

            report = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(report["maximum_drawdown"]["peak_date"], "2026-01-01")
            self.assertEqual(report["maximum_drawdown"]["trough_date"], "2026-01-02")
            self.assertEqual(report["maximum_drawdown"]["recovery_date"], "2026-01-03")

    def test_cli_returns_two_for_invalid_input(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "equity.csv"
            dataset.write_text("wrong,header\n", encoding="utf-8")

            with contextlib.redirect_stderr(io.StringIO()):
                exit_code = main([str(dataset)])

            self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()
