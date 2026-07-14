import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from quant_research_micro_lab.cli import load_price_csv, main


class PriceCsvTests(unittest.TestCase):
    def write_csv(self, directory: str, contents: str) -> Path:
        path = Path(directory) / "prices.csv"
        path.write_text(contents, encoding="utf-8")
        return path

    def test_loads_iso_dates_and_positive_closes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(
                directory,
                "date,close\n2026-01-02,10\n2026-01-05,10.5\n",
            )

            dates, prices = load_price_csv(path)

            self.assertEqual(dates, ["2026-01-02", "2026-01-05"])
            self.assertEqual(prices, [10.0, 10.5])

    def test_rejects_noncanonical_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(directory, "close,date\n10,2026-01-02\n")

            with self.assertRaisesRegex(ValueError, "header"):
                load_price_csv(path)

    def test_rejects_invalid_or_unordered_dates(self):
        cases = [
            "date,close\nnot-a-date,10\n",
            "date,close\n2026-01-03,10\n2026-01-02,11\n",
            "date,close\n2026-01-02,10\n2026-01-02,11\n",
        ]
        for contents in cases:
            with self.subTest(contents=contents):
                with tempfile.TemporaryDirectory() as directory:
                    path = self.write_csv(directory, contents)
                    with self.assertRaisesRegex(ValueError, "date"):
                        load_price_csv(path)

    def test_rejects_nonfinite_nonpositive_and_empty_closes(self):
        for close in ("nan", "inf", "0", "-1", ""):
            with self.subTest(close=close):
                with tempfile.TemporaryDirectory() as directory:
                    path = self.write_csv(directory, f"date,close\n2026-01-02,{close}\n")
                    with self.assertRaisesRegex(ValueError, "close"):
                        load_price_csv(path)

    def test_cli_runs_backtest_and_writes_dated_equity(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.write_csv(
                directory,
                "date,close\n"
                "2026-01-01,10\n"
                "2026-01-02,10\n"
                "2026-01-03,10\n"
                "2026-01-04,11\n"
                "2026-01-05,12\n",
            )
            equity_output = Path(directory) / "equity.csv"
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        str(dataset),
                        "--short-window",
                        "2",
                        "--long-window",
                        "3",
                        "--transaction-cost-bps",
                        "10",
                        "--equity-output",
                        str(equity_output),
                    ]
                )

            self.assertEqual(exit_code, 0)
            report = json.loads(stdout.getvalue())
            self.assertEqual(report["observations"], 5)
            self.assertEqual(report["start_date"], "2026-01-01")
            self.assertEqual(len(report["equity"]), 5)
            rows = equity_output.read_text(encoding="utf-8").splitlines()
            self.assertEqual(rows[0], "date,equity,gross_equity")
            self.assertEqual(len(rows), 6)

    def test_cli_returns_two_for_invalid_input(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.write_csv(directory, "wrong,header\n1,2\n")

            with contextlib.redirect_stderr(io.StringIO()):
                exit_code = main([str(dataset)])

            self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()
