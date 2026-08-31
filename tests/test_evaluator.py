from __future__ import annotations

import unittest
from pathlib import Path
import json
import tempfile

from evaluator.local_evaluator import catalog_index, evaluate, metric_summary, normalize_recommendations


class EchoTargetAgent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        self.session_id = session_id

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        asin = "A"
        if "B" in user_message:
            asin = "B"
        return {"message": "ok", "ask_attribute": None, "recommendations": [{"parent_asin": asin}]}


class DiagnosticMissAgent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        self.session_id = session_id

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {"message": "ok", "ask_attribute": None, "recommendations": [{"parent_asin": "B"}]}

    def debug_session(self, session_id: str) -> dict:
        return {
            "retrieval_history": [{
                "turn": 1,
                "intent": "buying",
                "active_slots": {"category": "shoe"},
                "query_terms": ["shoe"],
                "sparse_pool": ["B", "A"],
                "ranked_pool": ["B", "A"],
                "ask_attribute": None,
            }]
        }


class BatchedQuestionAgent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        self.history: list[dict] = []

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        ask = "other" if turn <= 2 else None
        self.history.append({
            "turn": turn,
            "intent": "browsing",
            "active_slots": {},
            "query_terms": ["shoe"],
            "sparse_pool": ["B"],
            "ranked_pool": ["B"],
            "ask_attribute": ask,
        })
        return {"message": "question", "ask_attribute": ask, "recommendations": [{"parent_asin": "B"}]}

    def debug_session(self, session_id: str) -> dict:
        return {"retrieval_history": self.history}


class EvaluatorTest(unittest.TestCase):
    def test_normalization_preserves_first_valid_unique_order(self) -> None:
        payload = [
            {"parent_asin": "A"}, {"parent_asin": "bad"}, {"parent_asin": "A"},
            "B", {"parent_asin": "C"},
        ]
        self.assertEqual(normalize_recommendations(payload, {"A", "B", "C"}), ["A", "B", "C"])

    def test_metric_summary_assigns_turn_11_to_miss(self) -> None:
        sessions = [
            {"hit": True, "reciprocal_rank": .5, "first_hit_turn": 2},
            {"hit": False, "reciprocal_rank": 0.0, "first_hit_turn": None},
        ]
        self.assertEqual(metric_summary(sessions), {
            "sample_count": 2,
            "hit_rate_at_10": .5,
            "mrr": .25,
            "mttc": 6.5,
        })

    def test_evaluate_derives_hidden_fields_when_public_set_omits_them(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.jsonl"
            catalog_rows = [
                {
                    "parent_asin": "A",
                    "title": "Blue running shoe",
                    "features": ["cotton"],
                    "details": {"department": "womens"},
                    "description": ["walking shoe"],
                    "categories": ["Clothing", "Shoes"],
                    "store": "Example",
                    "average_rating": 4.2,
                    "rating_number": 10,
                    "price": 49.0,
                },
                {
                    "parent_asin": "B",
                    "title": "Black winter boot",
                    "features": ["leather"],
                    "details": {"department": "womens"},
                    "description": ["winter boot"],
                    "categories": ["Clothing", "Boots"],
                    "store": "Example",
                    "average_rating": 4.4,
                    "rating_number": 12,
                    "price": 89.0,
                },
            ]
            catalog_path.write_text("".join(json.dumps(row) + "\n" for row in catalog_rows), encoding="utf-8")
            catalog_ids, categories, products = catalog_index(catalog_path)
            samples = [{
                "sample_id": "public_v2_0001",
                "scenario_type": "buying",
                "user_profile": {"summary": "x"},
                "ground_truth": {"parent_asin": "A"},
            }]
            result = evaluate(EchoTargetAgent(), samples, catalog_ids, categories, products)
            self.assertEqual(result["hit_rate_at_10"], 1.0)

    def test_diagnostics_distinguish_ranking_from_retrieval_misses(self) -> None:
        products = {
            "A": {"parent_asin": "A", "title": "shoe"},
            "B": {"parent_asin": "B", "title": "boot"},
        }
        samples = [{
            "sample_id": "diagnostic",
            "scenario_type": "buying",
            "user_profile": {},
            "ground_truth": {"parent_asin": "A"},
        }]
        result = evaluate(
            DiagnosticMissAgent(), samples, {"A", "B"}, {"A": ["Shoes"]}, products,
            include_diagnostics=True,
        )
        self.assertEqual(result["diagnostic_summary"]["failure_modes"], {"ranking": 1})
        self.assertEqual(result["diagnostic_summary"]["recall_pool_hit_rate"], 1.0)

    def test_diagnostics_measure_batched_constraint_disclosures(self) -> None:
        products = {
            "A": {
                "parent_asin": "A",
                "title": "Blue cotton running shoe",
                "features": ["machine washable", "rubber sole"],
                "categories": ["Clothing", "Shoes"],
            },
            "B": {"parent_asin": "B", "title": "boot", "categories": ["Clothing", "Boots"]},
        }
        samples = [{
            "sample_id": "batched",
            "scenario_type": "browsing",
            "user_profile": {},
            "ground_truth": {"parent_asin": "A"},
        }]
        result = evaluate(
            BatchedQuestionAgent(), samples, {"A", "B"},
            {"A": ["Shoes"], "B": ["Boots"]}, products,
            include_diagnostics=True,
        )
        turns = result["sessions"][0]["diagnostics"]["turns"]
        self.assertEqual(turns[1]["newly_disclosed_count"], 2)
        self.assertEqual(turns[2]["newly_disclosed_count"], 2)
        self.assertEqual(turns[2]["remaining_undisclosed_constraints"], 0)
        self.assertEqual(result["diagnostic_summary"]["constraints_disclosed_by_questions"], 4)


if __name__ == "__main__":
    unittest.main()
