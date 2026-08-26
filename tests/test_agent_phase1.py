from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.agent import Agent, extract_slots


class Phase1AgentTest(unittest.TestCase):
    def test_extracts_normalized_slots(self) -> None:
        slots = extract_slots("women's red dresses, size medium, under $50")
        self.assertEqual(slots["color"], "red")
        self.assertEqual(slots["gender"], "women")
        self.assertEqual(slots["size"], "M")
        self.assertEqual(slots["category"], "dress")
        self.assertEqual(slots["max_price"], 50.0)

    def test_negated_color_is_not_a_positive_slot(self) -> None:
        slots = extract_slots("a jacket without red, above $20")
        self.assertNotIn("color", slots)
        self.assertEqual(slots["min_price"], 20.0)
        self.assertIn("red", slots["negated_terms"])

    def test_state_merges_and_overrides_slots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.jsonl"
            catalog.write_text(
                "\n".join(
                    json.dumps({"parent_asin": asin, "title": title})
                    for asin, title in (("A", "red dress"), ("B", "blue dress"))
                )
                + "\n",
                encoding="utf-8",
            )
            agent = Agent(catalog)
            agent.reset("session", {"summary": "profile"})
            agent.respond("session", "a red dress under $50", 1, 10)
            agent.respond("session", "Actually, make it blue", 2, 10)

            state = agent.sessions["session"]
            self.assertEqual(state.accumulated_slots["color"], "blue")
            self.assertEqual(state.accumulated_slots["max_price"], 50.0)
            self.assertEqual([item["role"] for item in state.dialog_history], ["user", "agent", "user", "agent"])

    def test_reset_isolated_and_copies_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.jsonl"
            catalog.write_text(json.dumps({"parent_asin": "A", "title": "dress"}) + "\n", encoding="utf-8")
            profile = {"tags": ["comfort"]}
            agent = Agent(catalog)
            agent.reset("one", profile)
            profile["tags"].append("changed")
            agent.respond("one", "red dress", 1, 10)
            agent.reset("two", {"summary": "other"})

            self.assertEqual(agent.sessions["one"].user_profile, {"tags": ["comfort"]})
            self.assertEqual(agent.sessions["two"].dialog_history, [])
            self.assertEqual(agent.sessions["two"].accumulated_slots, {})

    def test_price_constraint_filters_candidates_and_requests_attribute(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.jsonl"
            catalog.write_text(
                "\n".join(
                    json.dumps({"parent_asin": asin, "title": "jacket", "price": price})
                    for asin, price in (("cheap", 20.0), ("expensive", 100.0))
                )
                + "\n",
                encoding="utf-8",
            )
            agent = Agent(catalog)
            agent.reset("session", {})
            response = agent.respond("session", "jacket under $30", 1, 10)

            self.assertEqual([item["parent_asin"] for item in response["recommendations"]], ["cheap"])
            self.assertEqual(response["ask_attribute"], "material")

    def test_declined_attribute_is_not_repeated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.jsonl"
            catalog.write_text(json.dumps({"parent_asin": "A", "title": "jacket"}) + "\n", encoding="utf-8")
            agent = Agent(catalog)
            agent.reset("session", {})
            first = agent.respond("session", "jacket", 1, 10)
            second = agent.respond(
                "session",
                f"I don't have a preference for {first['ask_attribute']}.",
                2,
                10,
            )

            self.assertEqual(first["ask_attribute"], "material")
            self.assertEqual(second["ask_attribute"], "feature")


if __name__ == "__main__":
    unittest.main()
