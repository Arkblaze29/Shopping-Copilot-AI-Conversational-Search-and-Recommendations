from __future__ import annotations

import unittest

from experiments.optimize import stratified_folds, technical_metrics


class OptimizerTest(unittest.TestCase):
    def test_stratified_folds_distribute_each_scenario(self) -> None:
        samples = [
            {"sample_id": f"{scenario}-{index}", "scenario_type": scenario}
            for scenario in ("buying", "browsing")
            for index in range(6)
        ]
        folds = stratified_folds(samples, 3)
        self.assertEqual([len(fold) for fold in folds], [4, 4, 4])
        for fold in folds:
            self.assertTrue(any(value.startswith("buying-") for value in fold))
            self.assertTrue(any(value.startswith("browsing-") for value in fold))

    def test_technical_metrics_uses_official_formula(self) -> None:
        sessions = [
            {"hit": True, "reciprocal_rank": 1.0, "first_hit_turn": 1},
            {"hit": False, "reciprocal_rank": 0.0, "first_hit_turn": None},
        ]
        result = technical_metrics(sessions)
        self.assertEqual(result["hit_rate_at_10"], 0.5)
        self.assertEqual(result["mrr"], 0.5)
        self.assertEqual(result["mttc"], 6.0)
        self.assertEqual(result["recommended_technical_score"], 0.5)


if __name__ == "__main__":
    unittest.main()
