import unittest

from quant_research_micro_lab.risk import analyze_drawdowns


class DrawdownAnalysisTests(unittest.TestCase):
    def test_reports_recovered_and_open_drawdown_episodes(self):
        report = analyze_drawdowns([1.0, 0.9, 0.8, 1.0, 1.1, 1.0, 0.99])

        self.assertEqual(report["episode_count"], 2)
        first, second = report["episodes"]
        self.assertEqual(
            (first["peak_index"], first["trough_index"], first["recovery_index"]),
            (0, 2, 3),
        )
        self.assertTrue(first["recovered"])
        self.assertAlmostEqual(first["depth"], -0.2)
        self.assertEqual(first["underwater_observations"], 2)
        self.assertIsNone(second["recovery_index"])
        self.assertFalse(second["recovered"])
        self.assertIs(report["maximum_drawdown"], first)
        self.assertIs(report["longest_underwater"], first)
        self.assertAlmostEqual(report["current_drawdown"], -0.1)

    def test_new_highs_without_drawdowns_produce_an_empty_episode_list(self):
        report = analyze_drawdowns([1.0, 1.0, 1.2])

        self.assertEqual(report["episode_count"], 0)
        self.assertEqual(report["current_drawdown"], 0.0)
        self.assertIsNone(report["maximum_drawdown"])
        self.assertIsNone(report["longest_underwater"])

    def test_open_episode_counts_observations_after_the_peak(self):
        report = analyze_drawdowns([1.0, 0.9, 0.8])

        self.assertEqual(
            report["episodes"][0]["underwater_observations"],
            2,
        )

    def test_empty_or_invalid_equity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            analyze_drawdowns([])
        for values in ([1.0, 0.0], [1.0, float("nan")], [1.0, True], [1.0, "x"]):
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "finite positive"):
                    analyze_drawdowns(values)


if __name__ == "__main__":
    unittest.main()
