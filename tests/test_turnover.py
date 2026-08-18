import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from quant_research_micro_lab.turnover import (
    MAX_INPUT_BYTES,
    audit_portfolio_turnover,
    main,
)


class PortfolioTurnoverTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"date": "2026-01-01", "asset": "AAA", "weight": 0.6},
            {"date": "2026-01-01", "asset": "BBB", "weight": 0.4},
            {"date": "2026-01-02", "asset": "AAA", "weight": 0.5},
            {"date": "2026-01-02", "asset": "BBB", "weight": 0.25},
            {"date": "2026-01-02", "asset": "CCC", "weight": 0.25},
            {"date": "2026-01-03", "asset": "AAA", "weight": 0.4},
            {"date": "2026-01-03", "asset": "BBB", "weight": -0.2},
            {"date": "2026-01-03", "asset": "CCC", "weight": 0.3},
        ]

    def test_reports_transition_and_per_asset_change_diagnostics(self):
        report = audit_portfolio_turnover(self.records, max_details=2)

        self.assertEqual(report["snapshot_count"], 3)
        self.assertEqual(report["transition_count"], 2)
        self.assertEqual(report["asset_count"], 3)
        self.assertAlmostEqual(report["summary"]["cumulative_turnover"], 0.55)
        self.assertAlmostEqual(report["summary"]["average_turnover"], 0.275)
        self.assertEqual(
            report["summary"]["maximum_turnover"],
            {"date": "2026-01-03", "value": 0.3},
        )
        self.assertEqual(
            report["summary"]["maximum_position_change"],
            {"date": "2026-01-03", "asset": "BBB", "change": -0.45, "absolute_change": 0.45},
        )
        first, second = report["transitions"]
        self.assertAlmostEqual(first["one_way_turnover"], 0.25)
        self.assertEqual(first["opened_positions"], 1)
        self.assertEqual(first["closed_positions"], 0)
        self.assertEqual(first["sign_flips"], 0)
        self.assertEqual(
            [detail["asset"] for detail in first["change_details"]],
            ["CCC", "BBB"],
        )
        self.assertTrue(first["details_truncated"])
        self.assertEqual(second["sign_flips"], 1)

    def test_unchanged_snapshots_report_zero_turnover(self):
        records = [
            {"date": "2026-01-01", "asset": "AAA", "weight": 1.0},
            {"date": "2026-01-02", "asset": "AAA", "weight": 1.0},
        ]

        report = audit_portfolio_turnover(records)

        self.assertEqual(report["summary"]["cumulative_turnover"], 0.0)
        self.assertIsNone(report["summary"]["maximum_position_change"])
        self.assertEqual(report["transitions"][0]["change_details"], [])
        self.assertFalse(report["transitions"][0]["details_truncated"])

    def test_threshold_failures_are_reported_in_stable_order(self):
        report = audit_portfolio_turnover(
            self.records,
            max_transition_turnover=0.2,
            max_position_change=0.3,
            max_cumulative_turnover=0.4,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(
            [failure["metric"] for failure in report["failures"]],
            ["transition_turnover", "position_change", "cumulative_turnover"],
        )
        self.assertEqual(report["failures"][0]["date"], "2026-01-03")
        self.assertEqual(report["failures"][1]["asset"], "BBB")
        self.assertAlmostEqual(report["failures"][2]["excess"], 0.15)

    def test_invalid_snapshot_count_details_and_thresholds_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "two distinct snapshot dates"):
            audit_portfolio_turnover(self.records[:2])
        for max_details in (True, -1, 1.5):
            with self.subTest(max_details=max_details):
                with self.assertRaisesRegex(ValueError, "max_details"):
                    audit_portfolio_turnover(self.records, max_details=max_details)
        for value in (True, -0.1, float("inf")):
            with self.subTest(max_transition_turnover=value):
                with self.assertRaisesRegex(ValueError, "max_transition_turnover"):
                    audit_portfolio_turnover(
                        self.records,
                        max_transition_turnover=value,
                    )

    def test_cli_writes_a_report_and_uses_gate_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "weights.csv"
            output = Path(directory) / "report.json"
            dataset.write_text(
                "date,asset,weight\n"
                "2026-01-01,AAA,0.6\n"
                "2026-01-01,BBB,0.4\n"
                "2026-01-02,AAA,0.2\n"
                "2026-01-02,BBB,0.8\n",
                encoding="utf-8",
            )

            exit_code = main(
                [
                    str(dataset),
                    "--max-transition-turnover",
                    "0.2",
                    "--output",
                    str(output),
                ]
            )

            self.assertEqual(exit_code, 1)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(report["passed"])
            self.assertEqual(report["failures"][0]["metric"], "transition_turnover")

    def test_cli_rejects_oversized_input_and_output_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "weights.csv"
            dataset.write_bytes(b" " * (MAX_INPUT_BYTES + 1))
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([str(dataset)]), 2)

            dataset.write_text(
                "date,asset,weight\n"
                "2026-01-01,AAA,1\n"
                "2026-01-02,AAA,1\n",
                encoding="utf-8",
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
