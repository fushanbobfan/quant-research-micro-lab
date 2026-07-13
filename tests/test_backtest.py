import unittest

from quant_research_micro_lab.backtest import backtest_crossover, maximum_drawdown


class BacktestTests(unittest.TestCase):
    def test_maximum_drawdown(self):
        self.assertAlmostEqual(maximum_drawdown([1.0, 1.2, 0.9, 1.1]), -0.25)

    def test_crossover_produces_equity_for_each_price(self):
        prices = [10, 10, 10, 11, 12, 13, 12]
        result = backtest_crossover(prices, short_window=2, long_window=3)
        self.assertEqual(len(result["equity"]), len(prices))
        self.assertGreater(result["total_return"], 0)

    def test_rejects_lookback_that_is_too_large(self):
        with self.assertRaises(ValueError):
            backtest_crossover([1, 2, 3], short_window=2, long_window=3)


if __name__ == "__main__":
    unittest.main()

