import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import quant_research_micro_lab
from quant_research_micro_lab.tail_risk import analyze_return_tail, main


class ReturnTailAnalysisTests(unittest.TestCase):
    def test_reports_fixed_count_worst_return_tail_and_downside_metrics(self):
        report = analyze_return_tail(
            [100.0, 90.0, 99.0, 79.2, 87.12, 104.544],
            confidence=0.6,
        )

        self.assertIs(
            quant_research_micro_lab.analyze_return_tail,
            analyze_return_tail,
        )
        self.assertEqual(report["observations"], 6)
        self.assertEqual(report["return_observations"], 5)
        self.assertEqual(report["tail_observation_count"], 2)
        self.assertAlmostEqual(report["tail_cutoff_return"], -0.1)
        self.assertAlmostEqual(report["mean_tail_return"], -0.15)
        self.assertAlmostEqual(report["worst_return"], -0.2)
        self.assertAlmostEqual(report["best_return"], 0.2)
        self.assertEqual(report["loss_period_count"], 2)
        self.assertAlmostEqual(report["loss_period_rate"], 0.4)
        self.assertAlmostEqual(report["downside_deviation"], 0.1)
        self.assertEqual(
            [(item["start_index"], item["end_index"]) for item in report["tail_periods"]],
            [(2, 3), (0, 1)],
        )

    def test_tail_size_is_fixed_when_boundary_returns_are_tied(self):
        report = analyze_return_tail([100.0, 90.0, 81.0, 89.1, 106.92], confidence=0.75)

        self.assertEqual(report["tail_observation_count"], 1)
        self.assertEqual(report["tail_periods"][0]["end_index"], 1)

    def test_invalid_equity_or_confidence_is_rejected(self):
        for equity in ([], [1.0], [1.0, 0.0], [1.0, float("nan")], [1.0, True]):
            with self.subTest(equity=equity):
                with self.assertRaisesRegex(ValueError, "equity"):
                    analyze_return_tail(equity)
        for confidence in (0.0, 1.0, True, float("inf"), "high"):
            with self.subTest(confidence=confidence):
                with self.assertRaisesRegex(ValueError, "confidence"):
                    analyze_return_tail([1.0, 1.1], confidence=confidence)

    def test_cli_adds_dates_to_selected_tail_periods(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "equity.csv"
            dataset.write_text(
                "date,equity,gross_equity\n"
                "2026-01-01,1.0,1.0\n"
                "2026-01-02,0.9,0.95\n"
                "2026-01-03,1.0,1.05\n"
                "2026-01-04,0.8,0.9\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = main([str(dataset), "--confidence", "0.5"])

            report = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(report["column"], "equity")
            self.assertEqual(report["tail_periods"][0]["start_date"], "2026-01-03")
            self.assertEqual(report["tail_periods"][0]["end_date"], "2026-01-04")

    def test_cli_returns_two_for_invalid_input(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "equity.csv"
            dataset.write_text("wrong,header\n", encoding="utf-8")

            with contextlib.redirect_stderr(io.StringIO()):
                exit_code = main([str(dataset)])

            self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()
