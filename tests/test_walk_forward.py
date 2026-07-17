import unittest

from quant_research_micro_lab.sweep import sweep_crossover
from quant_research_micro_lab.walk_forward import walk_forward_crossover


class WalkForwardTests(unittest.TestCase):
    prices = [
        10,
        10,
        11,
        12,
        11,
        10,
        11,
        12,
        13,
        12,
        11,
        12,
        13,
        14,
        13,
        12,
        13,
        14,
        15,
        14,
        13,
    ]

    def test_rolling_folds_select_only_on_training_data(self):
        report = walk_forward_crossover(
            self.prices,
            short_windows=[1, 2],
            long_windows=[3, 4],
            train_size=8,
            test_size=4,
            transaction_cost_bps=10,
        )
        first_training_sweep = sweep_crossover(
            self.prices[:8],
            short_windows=[1, 2],
            long_windows=[3, 4],
            transaction_cost_bps=10,
        )
        expected = first_training_sweep["results"][0]

        self.assertEqual(report["fold_count"], 3)
        self.assertEqual(report["evaluated_observations"], 12)
        self.assertEqual(report["unused_trailing_observations"], 1)
        self.assertEqual(
            [
                (
                    fold["train_start_index"],
                    fold["train_end_index"],
                    fold["test_start_index"],
                    fold["test_end_index"],
                )
                for fold in report["folds"]
            ],
            [(0, 7, 8, 11), (4, 11, 12, 15), (8, 15, 16, 19)],
        )
        self.assertEqual(
            (
                report["folds"][0]["selected_short_window"],
                report["folds"][0]["selected_long_window"],
            ),
            (expected["short_window"], expected["long_window"]),
        )

    def test_summary_compounds_fold_returns_and_counts_selections(self):
        report = walk_forward_crossover(
            self.prices[:20],
            short_windows=[1, 2],
            long_windows=[3, 4],
            train_size=8,
            test_size=4,
        )

        compounded = 1.0
        for fold in report["folds"]:
            compounded *= 1.0 + fold["test_total_return"]
        self.assertAlmostEqual(
            report["out_of_sample"]["total_return"], compounded - 1.0
        )
        self.assertEqual(
            sum(item["fold_count"] for item in report["selection_counts"]),
            report["fold_count"],
        )
        self.assertLessEqual(report["out_of_sample"]["maximum_drawdown"], 0.0)

    def test_invalid_sizes_prices_and_grids_are_rejected(self):
        base = {
            "short_windows": [1],
            "long_windows": [3],
            "train_size": 5,
            "test_size": 2,
        }
        for updates in (
            {"train_size": 0},
            {"test_size": True},
            {"long_windows": []},
        ):
            with self.subTest(updates=updates):
                with self.assertRaises(ValueError):
                    walk_forward_crossover(self.prices, **(base | updates))
        with self.assertRaisesRegex(ValueError, r"train_size \+ test_size"):
            walk_forward_crossover(self.prices[:6], **base)
        for bad_price in (-1, float("nan"), True, "10"):
            with self.subTest(bad_price=bad_price):
                with self.assertRaisesRegex(ValueError, "finite positive"):
                    walk_forward_crossover([10, 11, bad_price, 13, 14, 15, 16], **base)


if __name__ == "__main__":
    unittest.main()
