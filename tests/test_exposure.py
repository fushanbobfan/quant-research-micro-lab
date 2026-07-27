import unittest

from quant_research_micro_lab.exposure import audit_portfolio_exposure


class PortfolioExposureTests(unittest.TestCase):
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

    def test_reports_dated_exposure_concentration_and_turnover(self):
        report = audit_portfolio_exposure(self.records)

        self.assertTrue(report["passed"])
        self.assertEqual(report["snapshot_count"], 3)
        self.assertEqual(report["asset_count"], 3)
        self.assertAlmostEqual(
            report["summary"]["average_gross_exposure"],
            2.9 / 3,
        )
        self.assertAlmostEqual(
            report["summary"]["average_abs_net_exposure"],
            5 / 6,
        )
        self.assertAlmostEqual(report["summary"]["average_turnover"], 0.275)
        self.assertEqual(
            report["extrema"]["maximum_gross_exposure"],
            {"date": "2026-01-01", "value": 1.0},
        )
        self.assertEqual(
            report["extrema"]["maximum_single_position"],
            {
                "date": "2026-01-01",
                "asset": "AAA",
                "weight": 0.6,
                "absolute_weight": 0.6,
            },
        )
        self.assertAlmostEqual(
            report["extrema"]["maximum_concentration_hhi"]["value"],
            0.52,
        )
        self.assertEqual(
            report["extrema"]["maximum_turnover"]["date"],
            "2026-01-03",
        )
        self.assertAlmostEqual(
            report["extrema"]["maximum_turnover"]["value"],
            0.3,
        )
        final = report["snapshots"][-1]
        self.assertAlmostEqual(final["long_exposure"], 0.7)
        self.assertAlmostEqual(final["short_exposure"], 0.2)
        self.assertAlmostEqual(final["net_exposure"], 0.5)
        self.assertAlmostEqual(final["concentration_hhi"], 29 / 81)
        self.assertAlmostEqual(final["effective_positions"], 81 / 29)

    def test_single_snapshot_has_no_turnover_extreme(self):
        report = audit_portfolio_exposure(self.records[:2])

        self.assertIsNone(report["summary"]["average_turnover"])
        self.assertIsNone(report["extrema"]["maximum_turnover"])
        self.assertIsNone(report["snapshots"][0]["turnover_from_previous"])

    def test_threshold_failures_are_reported_in_stable_order(self):
        report = audit_portfolio_exposure(
            self.records,
            max_gross_exposure=0.9,
            max_abs_net_exposure=0.8,
            max_single_position=0.5,
            max_concentration_hhi=0.5,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(
            [failure["metric"] for failure in report["failures"]],
            [
                "gross_exposure",
                "abs_net_exposure",
                "single_position",
                "concentration_hhi",
            ],
        )
        self.assertEqual(report["failures"][0]["date"], "2026-01-01")
        self.assertAlmostEqual(report["failures"][3]["excess"], 0.02)

    def test_largest_position_ties_are_resolved_by_asset(self):
        report = audit_portfolio_exposure(
            [
                {"date": "2026-01-01", "asset": "BBB", "weight": -0.5},
                {"date": "2026-01-01", "asset": "AAA", "weight": 0.5},
            ]
        )

        self.assertEqual(
            report["snapshots"][0]["largest_position_asset"],
            "AAA",
        )

    def test_invalid_records_are_rejected(self):
        cases = [
            ([], "at least one"),
            ([3], "object"),
            (
                [{"date": "2026/01/01", "asset": "AAA", "weight": 1}],
                "ISO date",
            ),
            (
                [
                    {"date": "2026-01-02", "asset": "AAA", "weight": 1},
                    {"date": "2026-01-01", "asset": "AAA", "weight": 1},
                ],
                "non-decreasing",
            ),
            (
                [{"date": "2026-01-01", "asset": "", "weight": 1}],
                "non-empty",
            ),
            (
                [{"date": "2026-01-01", "asset": "AAA", "weight": True}],
                "finite",
            ),
            (
                [
                    {"date": "2026-01-01", "asset": "AAA", "weight": 1},
                    {"date": "2026-01-01", "asset": "AAA", "weight": 0},
                ],
                "duplicate",
            ),
            (
                [
                    {"date": "2026-01-01", "asset": "AAA", "weight": 0},
                    {"date": "2026-01-01", "asset": "BBB", "weight": 0},
                ],
                "non-zero",
            ),
        ]
        for records, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    audit_portfolio_exposure(records)

    def test_invalid_thresholds_are_rejected(self):
        for value in (-0.1, True, float("inf"), "high"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "max_gross_exposure"):
                    audit_portfolio_exposure(
                        self.records,
                        max_gross_exposure=value,
                    )
        with self.assertRaisesRegex(ValueError, "max_concentration_hhi"):
            audit_portfolio_exposure(
                self.records,
                max_concentration_hhi=1.1,
            )


if __name__ == "__main__":
    unittest.main()
