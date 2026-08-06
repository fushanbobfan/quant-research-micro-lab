import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import quant_research_micro_lab
from quant_research_micro_lab.dependence import analyze_return_dependence, main


class ReturnDependenceTests(unittest.TestCase):
    def setUp(self):
        self.returns = [-0.1, 0.1, -0.1, 0.1, -0.1, 0.1]

    def test_report_contains_standard_autocorrelations_and_ljung_box_statistic(self):
        report = analyze_return_dependence(self.returns, max_lag=2)

        self.assertIs(
            quant_research_micro_lab.analyze_return_dependence,
            analyze_return_dependence,
        )
        self.assertEqual(report["return_count"], 6)
        self.assertAlmostEqual(
            report["autocorrelations"][0]["autocorrelation"], -5 / 6
        )
        self.assertAlmostEqual(
            report["autocorrelations"][1]["autocorrelation"], 2 / 3
        )
        self.assertEqual(
            report["summary"]["maximum_absolute_autocorrelation"]["lag"], 1
        )
        expected_q = 6 * 8 * (((5 / 6) ** 2) / 5 + ((2 / 3) ** 2) / 4)
        self.assertAlmostEqual(report["summary"]["ljung_box_statistic"], expected_q)

    def test_autocorrelation_gate_reports_the_extreme_lag(self):
        report = analyze_return_dependence(
            self.returns, max_lag=2, max_abs_autocorrelation=0.8
        )

        self.assertFalse(report["passed"])
        self.assertEqual(report["failures"][0]["lag"], 1)
        self.assertAlmostEqual(report["failures"][0]["actual"], 5 / 6)

    def test_invalid_returns_lags_and_thresholds_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            analyze_return_dependence([0.1])
        with self.assertRaisesRegex(ValueError, "greater than -1"):
            analyze_return_dependence([0.1, -1.0, 0.2], max_lag=1)
        with self.assertRaisesRegex(ValueError, "non-zero variance"):
            analyze_return_dependence([0.1, 0.1, 0.1], max_lag=1)
        for value in (True, 0, 6, 1.2):
            with self.subTest(max_lag=value):
                with self.assertRaisesRegex(ValueError, "max_lag"):
                    analyze_return_dependence(self.returns, max_lag=value)
        for value in (True, -0.1, 1.1, float("inf")):
            with self.subTest(threshold=value):
                with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                    analyze_return_dependence(
                        self.returns, max_abs_autocorrelation=value
                    )

    def test_cli_reads_equity_export_and_writes_dated_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "equity.csv"
            output = root / "report.json"
            rows = [
                "date,equity,gross_equity",
                "2026-01-01,100,100",
                "2026-01-02,90,90",
                "2026-01-03,99,99",
                "2026-01-04,89.1,89.1",
                "2026-01-05,98.01,98.01",
                "2026-01-06,88.209,88.209",
                "2026-01-07,97.0299,97.0299",
            ]
            dataset.write_text("\n".join(rows) + "\n", encoding="utf-8")

            exit_code = main(
                [
                    str(dataset),
                    "--max-lag",
                    "2",
                    "--output",
                    str(output),
                ]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["start_date"], "2026-01-01")
            self.assertEqual(report["end_date"], "2026-01-07")
            self.assertEqual(report["return_count"], 6)

    def test_cli_returns_one_for_gate_failure_and_two_for_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "equity.csv"
            dataset.write_text(
                "date,equity,gross_equity\n"
                "2026-01-01,100,100\n"
                "2026-01-02,90,90\n"
                "2026-01-03,99,99\n"
                "2026-01-04,89.1,89.1\n",
                encoding="utf-8",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            str(dataset),
                            "--max-lag",
                            "1",
                            "--max-abs-autocorrelation",
                            "0.5",
                        ]
                    ),
                    1,
                )

            original = dataset.read_text(encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(
                    main([str(dataset), "--output", str(dataset)]),
                    2,
                )
            self.assertEqual(dataset.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
