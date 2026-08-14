import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import quant_research_micro_lab
from quant_research_micro_lab.correlation_drift import (
    analyze_correlation_drift,
    main,
)


class CorrelationDriftTests(unittest.TestCase):
    def setUp(self):
        self.baseline = [
            {"date": "2026-01-02", "ALPHA": 0.01, "BETA": 0.02},
            {"date": "2026-01-05", "ALPHA": 0.02, "BETA": 0.04},
            {"date": "2026-01-06", "ALPHA": -0.01, "BETA": -0.02},
            {"date": "2026-01-07", "ALPHA": 0.00, "BETA": 0.00},
        ]
        self.candidate = [
            {"date": "2026-02-02", "ALPHA": 0.01, "BETA": -0.02},
            {"date": "2026-02-03", "ALPHA": 0.02, "BETA": -0.04},
            {"date": "2026-02-04", "ALPHA": -0.01, "BETA": 0.02},
            {"date": "2026-02-05", "ALPHA": 0.00, "BETA": 0.00},
        ]

    def test_public_api_and_exact_pair_change(self):
        self.assertIs(
            quant_research_micro_lab.analyze_correlation_drift,
            analyze_correlation_drift,
        )

        report = analyze_correlation_drift(self.baseline, self.candidate)

        self.assertTrue(report["passed"])
        self.assertEqual(report["metrics"]["asset_count"], 2)
        self.assertEqual(report["metrics"]["pair_count"], 1)
        self.assertAlmostEqual(
            report["metrics"]["maximum_abs_correlation_change"], 2.0
        )
        self.assertAlmostEqual(
            report["metrics"]["mean_abs_correlation_change"], 2.0
        )
        self.assertAlmostEqual(report["metrics"]["rms_correlation_change"], 2.0)
        self.assertEqual(report["metrics"]["sign_flip_count"], 1)
        self.assertEqual(
            report["largest_change"],
            {
                "asset_a": "ALPHA",
                "asset_b": "BETA",
                "baseline_correlation": 1.0,
                "candidate_correlation": -1.0,
                "change": -2.0,
                "absolute_change": 2.0,
                "sign_flipped": True,
            },
        )

    def test_threshold_failures_are_reported_together(self):
        report = analyze_correlation_drift(
            self.baseline,
            self.candidate,
            max_abs_correlation_change=1.5,
            max_rms_correlation_change=1.5,
            max_sign_flips=0,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(
            [failure["metric"] for failure in report["failures"]],
            [
                "maximum_abs_correlation_change",
                "rms_correlation_change",
                "sign_flip_count",
            ],
        )

    def test_pair_details_are_deterministic_and_bounded(self):
        report = analyze_correlation_drift(
            self.baseline,
            self.candidate,
            max_details=0,
        )

        self.assertEqual(report["pair_changes"], [])
        self.assertTrue(report["details_truncated"])
        self.assertEqual(report["largest_change"]["asset_a"], "ALPHA")

    def test_multi_asset_pairs_use_sorted_asset_names(self):
        baseline = [
            {**record, "GAMMA": value}
            for record, value in zip(self.baseline, [0.03, -0.01, 0.02, 0.00])
        ]
        candidate = [
            {**record, "GAMMA": value}
            for record, value in zip(self.candidate, [-0.01, 0.03, 0.01, -0.02])
        ]

        report = analyze_correlation_drift(baseline, candidate)

        self.assertEqual(report["metrics"]["pair_count"], 3)
        self.assertEqual(
            sorted(
                (pair["asset_a"], pair["asset_b"])
                for pair in report["pair_changes"]
            ),
            [("ALPHA", "BETA"), ("ALPHA", "GAMMA"), ("BETA", "GAMMA")],
        )

    def test_invalid_inputs_are_rejected(self):
        invalid_calls = [
            (self.baseline[:2], self.candidate, {}),
            (self.baseline, self.candidate, {"max_abs_correlation_change": 2.1}),
            (self.baseline, self.candidate, {"max_rms_correlation_change": -0.1}),
            (self.baseline, self.candidate, {"max_sign_flips": True}),
            (self.baseline, self.candidate, {"max_details": -1}),
        ]
        for baseline, candidate, kwargs in invalid_calls:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                analyze_correlation_drift(baseline, candidate, **kwargs)

        mismatched = [{**record, "GAMMA": 0.01} for record in self.candidate]
        with self.assertRaisesRegex(ValueError, "same asset columns"):
            analyze_correlation_drift(self.baseline, mismatched)

        constant = [{**record, "BETA": 0.0} for record in self.baseline]
        with self.assertRaisesRegex(ValueError, "zero variance"):
            analyze_correlation_drift(constant, self.candidate)

    def test_cli_writes_report_and_uses_gate_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline_path = Path(directory) / "baseline.csv"
            candidate_path = Path(directory) / "candidate.csv"
            output_path = Path(directory) / "report.json"
            self._write_csv(baseline_path, self.baseline)
            self._write_csv(candidate_path, self.candidate)

            exit_code = main(
                [
                    str(baseline_path),
                    str(candidate_path),
                    "--max-abs-correlation-change",
                    "1.5",
                    "--output",
                    str(output_path),
                ]
            )

            self.assertEqual(exit_code, 1)
            self.assertFalse(
                json.loads(output_path.read_text(encoding="utf-8"))["passed"]
            )

    def test_cli_rejects_input_and_output_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline_path = Path(directory) / "baseline.csv"
            candidate_path = Path(directory) / "candidate.csv"
            self._write_csv(baseline_path, self.baseline)
            self._write_csv(candidate_path, self.candidate)

            with contextlib.redirect_stderr(io.StringIO()):
                same_input_exit = main([str(baseline_path), str(baseline_path)])
                output_alias_exit = main(
                    [
                        str(baseline_path),
                        str(candidate_path),
                        "--output",
                        str(candidate_path),
                    ]
                )

        self.assertEqual(same_input_exit, 2)
        self.assertEqual(output_alias_exit, 2)

    @staticmethod
    def _write_csv(path, records):
        assets = sorted(set(records[0]) - {"date"})
        lines = ["date," + ",".join(assets)]
        for record in records:
            lines.append(
                record["date"]
                + ","
                + ",".join(str(record[asset]) for asset in assets)
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
