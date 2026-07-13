import unittest

from quant_research_micro_lab.backtest import backtest_crossover, maximum_drawdown


class BacktestTests(unittest.TestCase):
    def test_maximum_drawdown(self):
        self.assertAlmostEqual(maximum_drawdown([1.0, 1.2, 0.9, 1.1]), -0.25)

    def test_crossover_produces_equity_for_each_price(self):
        prices = [10, 10, 10, 11, 12, 13, 12]
        result = backtest_crossover(prices, short_window=2, long_window=3)
        self.assertEqual(len(result["equity"]), len(prices))
        self.assertEqual(len(result["gross_equity"]), len(prices))
        self.assertGreater(result["total_return"], 0)
        self.assertEqual(result["total_return"], result["gross_total_return"])

    def test_transaction_cost_is_charged_on_entry(self):
        prices = [10, 10, 10, 11, 12, 13, 12]

        result = backtest_crossover(
            prices,
            short_window=2,
            long_window=3,
            transaction_cost_bps=100,
        )

        self.assertEqual(result["total_turnover"], 1.0)
        self.assertAlmostEqual(
            result["cost_drag"], result["gross_equity"][-1] * 0.01
        )
        self.assertLess(result["total_return"], result["gross_total_return"])

    def test_cost_factor_keeps_equity_positive_after_large_loss(self):
        prices = [10, 10, 10, 11, 0.11]

        result = backtest_crossover(
            prices,
            short_window=2,
            long_window=3,
            transaction_cost_bps=100,
        )

        self.assertGreater(result["equity"][-1], 0)

    def test_exit_adds_turnover_after_signal_lag(self):
        prices = [10, 10, 10, 11, 12, 11, 10, 10]

        result = backtest_crossover(
            prices,
            short_window=2,
            long_window=3,
            transaction_cost_bps=10,
        )

        self.assertEqual(result["total_turnover"], 2.0)

    def test_invalid_transaction_cost_is_rejected(self):
        for cost in (-1, 10_000):
            with self.subTest(cost=cost):
                with self.assertRaises(ValueError):
                    backtest_crossover(
                        [10, 10, 10, 11],
                        short_window=2,
                        long_window=3,
                        transaction_cost_bps=cost,
                    )

    def test_rejects_lookback_that_is_too_large(self):
        with self.assertRaises(ValueError):
            backtest_crossover([1, 2, 3], short_window=2, long_window=3)


if __name__ == "__main__":
    unittest.main()
