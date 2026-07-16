import unittest

from quant_research_micro_lab.sweep import sweep_crossover


class ParameterSweepTests(unittest.TestCase):
    prices = [10, 10, 10, 11, 12, 13, 12, 11, 12, 13]

    def test_sweep_ranks_every_valid_pair_and_records_skips(self):
        report = sweep_crossover(
            self.prices,
            short_windows=[2, 4],
            long_windows=[3, 5],
            transaction_cost_bps=10,
        )

        self.assertEqual(report["candidate_count"], 3)
        self.assertEqual([result["rank"] for result in report["results"]], [1, 2, 3])
        self.assertGreaterEqual(
            report["results"][0]["total_return"],
            report["results"][1]["total_return"],
        )
        self.assertEqual(
            report["skipped_pairs"],
            [
                {
                    "short_window": 4,
                    "long_window": 3,
                    "reason": "short_window must be smaller than long_window",
                }
            ],
        )

    def test_minimize_metric_sorts_ascending_with_stable_ties(self):
        report = sweep_crossover(
            [10] * 8,
            short_windows=[2, 1],
            long_windows=[4, 3],
            rank_by="annualized_volatility",
        )

        pairs = [
            (result["short_window"], result["long_window"])
            for result in report["results"]
        ]
        self.assertEqual(report["direction"], "minimize")
        self.assertEqual(pairs, [(1, 3), (1, 4), (2, 3), (2, 4)])

    def test_invalid_windows_and_rank_metric_are_rejected(self):
        cases = [
            {"short_windows": [], "long_windows": [3]},
            {"short_windows": [0], "long_windows": [3]},
            {"short_windows": [2, 2], "long_windows": [3]},
            {"short_windows": [3], "long_windows": [2]},
            {"short_windows": [2], "long_windows": [3], "rank_by": "sharpe"},
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    sweep_crossover(self.prices, **arguments)

    def test_grid_preserves_backtest_data_requirements(self):
        with self.assertRaisesRegex(ValueError, "observations"):
            sweep_crossover(
                [10, 11, 12],
                short_windows=[2],
                long_windows=[3],
            )


if __name__ == "__main__":
    unittest.main()
