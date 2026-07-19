import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import quant_research_micro_lab
from quant_research_micro_lab.trades import build_trade_ledger, main


class TradeLedgerTests(unittest.TestCase):
    dates = [
        "2026-01-01",
        "2026-01-02",
        "2026-01-03",
        "2026-01-04",
        "2026-01-05",
        "2026-01-06",
        "2026-01-07",
        "2026-01-08",
    ]

    def test_trade_ledger_api_is_available_from_package(self):
        self.assertIs(quant_research_micro_lab.build_trade_ledger, build_trade_ledger)

    def test_closed_trade_uses_lagged_execution_dates_and_both_costs(self):
        prices = [100, 100, 101, 103, 106, 105, 102, 100]

        report = build_trade_ledger(
            self.dates,
            prices,
            short_window=2,
            long_window=3,
            transaction_cost_bps=10,
        )

        self.assertEqual(report["summary"]["trade_count"], 1)
        self.assertEqual(report["summary"]["closed_trade_count"], 1)
        self.assertEqual(report["summary"]["open_trade_count"], 0)
        self.assertEqual(report["summary"]["total_turnover"], 2.0)
        trade = report["trades"][0]
        self.assertEqual(trade["status"], "closed")
        self.assertEqual(trade["entry_date"], "2026-01-03")
        self.assertEqual(trade["entry_price"], 101.0)
        self.assertEqual(trade["exit_date"], "2026-01-07")
        self.assertEqual(trade["exit_price"], 102.0)
        self.assertEqual(trade["holding_observations"], 4)
        self.assertAlmostEqual(trade["gross_return"], 102 / 101 - 1)
        self.assertAlmostEqual(trade["net_return"], (102 / 101) * 0.999**2 - 1)
        self.assertAlmostEqual(
            trade["cost_drag"], trade["gross_return"] - trade["net_return"]
        )

    def test_open_trade_is_marked_without_an_assumed_exit_cost(self):
        prices = [100, 100, 101, 103, 106, 108, 110, 112]

        report = build_trade_ledger(
            self.dates,
            prices,
            short_window=2,
            long_window=3,
            transaction_cost_bps=25,
        )

        trade = report["trades"][0]
        self.assertEqual(report["summary"]["closed_trade_count"], 0)
        self.assertEqual(report["summary"]["open_trade_count"], 1)
        self.assertIsNone(report["summary"]["closed_win_rate"])
        self.assertEqual(trade["status"], "open")
        self.assertIsNone(trade["exit_date"])
        self.assertEqual(trade["mark_date"], "2026-01-08")
        self.assertEqual(trade["holding_observations"], 5)
        self.assertAlmostEqual(trade["net_return"], (112 / 101) * 0.9975 - 1)

    def test_flat_market_has_no_trades(self):
        report = build_trade_ledger(
            self.dates,
            [100] * len(self.dates),
            short_window=2,
            long_window=3,
        )

        self.assertEqual(report["trades"], [])
        self.assertEqual(report["summary"]["trade_count"], 0)
        self.assertEqual(report["summary"]["total_return"], 0.0)

    def test_invalid_observations_are_rejected(self):
        cases = [
            (["2026-01-01"], [1, 2], "same length"),
            (["not-a-date"] * 4, [1, 2, 3, 4], "valid ISO"),
            (
                ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-03"],
                [1, 2, 3, 4],
                "strictly increasing",
            ),
            (
                ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
                [1, float("nan"), 3, 4],
                "finite and positive",
            ),
        ]
        for dates, prices, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    build_trade_ledger(
                        dates,
                        prices,
                        short_window=2,
                        long_window=3,
                    )

    def test_cli_writes_trade_ledger_json(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "prices.csv"
            output = Path(directory) / "trades.json"
            dataset.write_text(
                "date,close\n"
                + "\n".join(
                    f"{day},{price}"
                    for day, price in zip(
                        self.dates,
                        [100, 100, 101, 103, 106, 105, 102, 100],
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            exit_code = main(
                [
                    str(dataset),
                    "--short-window",
                    "2",
                    "--long-window",
                    "3",
                    "--transaction-cost-bps",
                    "10",
                    "--output",
                    str(output),
                ]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["trades"][0]["entry_date"], "2026-01-03")
            self.assertEqual(report["summary"]["closed_trade_count"], 1)

    def test_cli_returns_two_for_invalid_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "prices.csv"
            dataset.write_text("wrong,header\n1,2\n", encoding="utf-8")

            with contextlib.redirect_stderr(io.StringIO()):
                exit_code = main([str(dataset)])

            self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()
