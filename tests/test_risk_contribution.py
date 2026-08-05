import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

from quant_research_micro_lab.risk_contribution import (
    analyze_risk_contributions,
    load_returns_csv,
    main,
)


def positions(weights=None):
    weights = weights or {"A": 0.5, "B": 0.5}
    return [
        {"date": "2026-01-05", "asset": asset, "weight": weight}
        for asset, weight in weights.items()
    ]


def returns():
    return [
        {"date": "2026-01-01", "A": -0.01, "B": -0.01},
        {"date": "2026-01-02", "A": 0.01, "B": -0.01},
        {"date": "2026-01-03", "A": -0.01, "B": 0.01},
        {"date": "2026-01-05", "A": 0.01, "B": 0.01},
    ]


class RiskContributionTests(unittest.TestCase):
    def test_equal_independent_assets_split_absolute_risk_evenly(self):
        report = analyze_risk_contributions(positions(), returns(), periods_per_year=4)

        expected_variance = (2.0 / 3.0) * 0.0001
        self.assertTrue(report["passed"])
        self.assertAlmostEqual(
            report["metrics"]["portfolio_variance_per_period"], expected_variance
        )
        self.assertAlmostEqual(
            report["metrics"]["component_volatility_sum"],
            report["metrics"]["portfolio_volatility_per_period"],
        )
        self.assertAlmostEqual(
            report["metrics"]["annualized_portfolio_volatility"],
            math.sqrt(expected_variance) * 2,
        )
        self.assertEqual(
            [
                item["absolute_component_share"]
                for item in report["asset_contributions"]
            ],
            [0.5, 0.5],
        )
        self.assertAlmostEqual(report["metrics"]["risk_concentration_hhi"], 0.5)

    def test_unequal_weights_show_component_concentration(self):
        report = analyze_risk_contributions(
            positions({"A": 0.75, "B": 0.25}), returns()
        )

        self.assertAlmostEqual(
            report["asset_contributions"][0]["absolute_component_share"], 0.9
        )
        self.assertAlmostEqual(report["metrics"]["risk_concentration_hhi"], 0.82)
        self.assertEqual(report["largest_absolute_risk_contributor"]["asset"], "A")

    def test_reports_both_concentration_gate_failures(self):
        report = analyze_risk_contributions(
            positions(),
            returns(),
            max_largest_risk_share=0.49,
            max_risk_concentration_hhi=0.49,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(
            {failure["metric"] for failure in report["failures"]},
            {"largest_absolute_risk_share", "risk_concentration_hhi"},
        )

    def test_rejects_lookahead_asset_mismatch_and_zero_variance(self):
        lookahead = returns()
        lookahead[-1] = {"date": "2026-01-06", "A": 0.01, "B": 0.01}
        mismatch = [dict(record) for record in returns()]
        mismatch[0]["C"] = mismatch[0].pop("B")
        flat = [
            {"date": "2026-01-01", "A": 0.0, "B": 0.0},
            {"date": "2026-01-02", "A": 0.0, "B": 0.0},
        ]

        for values in (lookahead, mismatch, flat):
            with self.subTest(values=values), self.assertRaises(ValueError):
                analyze_risk_contributions(positions(), values)

    def test_rejects_invalid_configuration_and_return_values(self):
        invalid_calls = [
            lambda: analyze_risk_contributions(
                positions(), returns(), periods_per_year=True
            ),
            lambda: analyze_risk_contributions(
                positions(), returns(), max_largest_risk_share=1.1
            ),
            lambda: analyze_risk_contributions(
                positions(), returns()[:1]
            ),
            lambda: analyze_risk_contributions(
                positions(),
                [
                    {"date": "2026-01-02", "A": 0.0, "B": 0.0},
                    {"date": "2026-01-01", "A": 0.1, "B": 0.1},
                ],
            ),
        ]

        for call in invalid_calls:
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()

    def test_load_returns_csv_is_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "returns.csv"
            path.write_text(
                "date,A,B\n2026-01-01,0.01,-0.02\n2026-01-02,0.02,0.01\n",
                encoding="utf-8",
            )
            loaded = load_returns_csv(path)
            self.assertEqual(loaded[0]["A"], 0.01)

            path.write_text("A,date\n0.01,2026-01-01\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_returns_csv(path)

    def test_cli_returns_zero_or_one_and_writes_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            portfolio = root / "portfolio.csv"
            returns_path = root / "returns.csv"
            output = root / "report.json"
            portfolio.write_text(
                "date,asset,weight\n2026-01-05,A,0.5\n2026-01-05,B,0.5\n",
                encoding="utf-8",
            )
            returns_path.write_text(
                "date,A,B\n"
                "2026-01-01,-0.01,-0.01\n"
                "2026-01-02,0.01,-0.01\n"
                "2026-01-03,-0.01,0.01\n"
                "2026-01-05,0.01,0.01\n",
                encoding="utf-8",
            )

            exit_code = main(
                [
                    str(portfolio),
                    str(returns_path),
                    "--max-largest-risk-share",
                    "0.6",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(json.loads(output.read_text(encoding="utf-8"))["passed"])

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        str(portfolio),
                        str(returns_path),
                        "--max-largest-risk-share",
                        "0.4",
                    ]
                )
            self.assertEqual(exit_code, 1)

    def test_cli_refuses_to_overwrite_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            portfolio = root / "portfolio.csv"
            returns_path = root / "returns.csv"
            portfolio_contents = (
                "date,asset,weight\n2026-01-05,A,0.5\n2026-01-05,B,0.5\n"
            )
            returns_contents = (
                "date,A,B\n2026-01-01,-0.01,-0.01\n2026-01-02,0.01,0.01\n"
            )
            portfolio.write_text(portfolio_contents, encoding="utf-8")
            returns_path.write_text(returns_contents, encoding="utf-8")

            for output in (portfolio, returns_path):
                with self.subTest(output=output), contextlib.redirect_stderr(
                    io.StringIO()
                ):
                    exit_code = main(
                        [str(portfolio), str(returns_path), "--output", str(output)]
                    )
                self.assertEqual(exit_code, 2)
            self.assertEqual(portfolio.read_text(encoding="utf-8"), portfolio_contents)
            self.assertEqual(returns_path.read_text(encoding="utf-8"), returns_contents)


if __name__ == "__main__":
    unittest.main()
