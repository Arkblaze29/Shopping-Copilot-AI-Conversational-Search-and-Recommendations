from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.agent import Agent, extract_slots
from starter.semantics import extract_product_facets


class SemanticAgentTest(unittest.TestCase):
    def _agent(self, directory: str, rows: list[dict]) -> Agent:
        catalog = Path(directory) / "catalog.jsonl"
        catalog.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return Agent(catalog)

    def test_category_aliases_do_not_corrupt_plural_words(self) -> None:
        self.assertEqual(extract_slots("women's shoes")["category"], "shoe")
        self.assertEqual(extract_slots("silk blouses")["category"], "blouse")
        self.assertEqual(extract_slots("warm hoodies")["category"], "hoodie")

    def test_specific_category_aliases_are_preferred(self) -> None:
        self.assertEqual(extract_slots("tank tops")["category"], "tank top")
        self.assertEqual(extract_slots("fashion sneakers")["category"], "fashion sneaker")
        self.assertEqual(extract_slots("loafers and slip-ons")["category"], "slip-on")

    def test_percentage_material_is_extractable(self) -> None:
        self.assertEqual(extract_slots("running shoes, 100% synthetic")["material"], "synthetic")

    def test_around_price_is_a_target_not_a_maximum(self) -> None:
        slots = extract_slots("budget around $50")
        self.assertEqual(slots["target_price"], 50.0)
        self.assertNotIn("max_price", slots)

    def test_product_facets_preserve_category_and_semantics(self) -> None:
        product = {
            "categories": ["Clothing, Shoes & Jewelry", "Men", "Shoes", "Hiking Boots"],
            "details": {"Department": "mens", "Manufacturer": "Trail Works"},
            "features": ["Available in size XL"],
        }
        facets = extract_product_facets(product, "brown leather waterproof hiking boots")
        self.assertEqual(facets.department, "men")
        self.assertEqual(facets.product_type, "boot")
        self.assertEqual(facets.subtype, "hiking boots")
        self.assertEqual(facets.brand, "trail works")
        self.assertIn("leather", facets.materials)
        self.assertIn("XL", facets.sizes)
        self.assertIn("hiking", facets.use_cases)
        self.assertIn("waterproof", facets.styles)

    def test_negation_removes_an_existing_slot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.jsonl"
            catalog.write_text(json.dumps({"parent_asin": "A", "title": "red jacket"}) + "\n", encoding="utf-8")
            agent = Agent(catalog)
            agent.reset("session", {})
            agent.respond("session", "red jacket", 1, 10)
            agent.respond("session", "actually, not red", 2, 10)
            self.assertNotIn("color", agent.sessions["session"].accumulated_slots)
            self.assertIn("red", agent.sessions["session"].negated_terms)

    def test_override_preserves_independent_constraints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.jsonl"
            catalog.write_text(json.dumps({"parent_asin": "A", "title": "blue cotton jacket", "price": 40}) + "\n", encoding="utf-8")
            agent = Agent(catalog)
            agent.reset("session", {})
            agent.respond("session", "red jacket under $50", 1, 10)
            agent.respond("session", "blue instead of red", 2, 10)
            slots = agent.sessions["session"].accumulated_slots
            self.assertEqual(slots["color"], "blue")
            self.assertEqual(slots["category"], "jacket")
            self.assertEqual(slots["max_price"], 50.0)

    def test_browsing_starts_with_high_value_clarification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.jsonl"
            catalog.write_text(json.dumps({"parent_asin": "A", "title": "jacket"}) + "\n", encoding="utf-8")
            agent = Agent(catalog)
            agent.reset("session", {})
            response = agent.respond("session", "I'm looking for a jacket, but I'm still exploring.", 1, 10)
            self.assertEqual(agent.sessions["session"].current_intent, "browsing")
            self.assertEqual(response["ask_attribute"], "material")

    def test_lower_bound_price_does_not_create_maximum(self) -> None:
        slots = extract_slots("jacket above $20")
        self.assertEqual(slots["min_price"], 20.0)
        self.assertNotIn("max_price", slots)
        self.assertNotIn("target_price", slots)

    def test_explicit_price_range_sets_both_bounds(self) -> None:
        slots = extract_slots("between $20 and $50")
        self.assertEqual(slots["min_price"], 20.0)
        self.assertEqual(slots["max_price"], 50.0)
        self.assertNotIn("target_price", slots)

    def test_target_price_prevents_redundant_budget_question(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self._agent(directory, [{"parent_asin": "A", "title": "jacket", "price": 50}])
            agent.reset("session", {})
            response = agent.respond("session", "jacket around $50", 1, 10)
            self.assertNotEqual(response["ask_attribute"], "budget")

    def test_attribute_qualified_negations_are_not_positive_slots(self) -> None:
        size_slots = extract_slots("not size medium")
        color_slots = extract_slots("no color red")
        self.assertNotIn("size", size_slots)
        self.assertEqual(size_slots["negated_slots"], {"size": {"M"}})
        self.assertNotIn("color", color_slots)
        self.assertEqual(color_slots["negated_slots"], {"color": {"red"}})

    def test_readding_value_clears_stale_negation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self._agent(directory, [{"parent_asin": "A", "title": "red jacket"}])
            agent.reset("session", {})
            agent.respond("session", "not red", 1, 10)
            agent.respond("session", "red after all", 2, 10)
            state = agent.sessions["session"]
            self.assertEqual(state.accumulated_slots["color"], "red")
            self.assertNotIn("red", state.negated_terms)

    def test_global_override_clears_soft_and_preserves_hard_slots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self._agent(directory, [{"parent_asin": "A", "title": "red leather waterproof jacket"}])
            agent.reset("session", {})
            agent.respond("session", "I need a red leather jacket", 1, 10)
            agent.respond("session", "I prefer casual", 2, 10)
            agent.respond(
                "session",
                "Actually, ignore my earlier preference. What I need is waterproof.",
                3,
                10,
            )
            slots = agent.sessions["session"].accumulated_slots
            self.assertEqual(slots["color"], "red")
            self.assertEqual(slots["material"], "leather")
            self.assertNotEqual(slots.get("style"), "casual")

    def test_brand_and_feature_clarifications_persist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self._agent(
                directory,
                [{"parent_asin": "A", "title": "Nike jacket", "features": ["machine washable"], "store": "Nike"}],
            )
            agent.reset("session", {})
            state = agent.sessions["session"]
            state.last_asked_attribute = "brand"
            agent.respond("session", "Nike", 2, 10)
            self.assertEqual(state.accumulated_slots["brand"], "nike")
            state.last_asked_attribute = "feature"
            agent.respond("session", "For that, what matters is: machine washable.", 3, 10)
            self.assertEqual(state.accumulated_slots["feature"], "machine washable")
            agent.respond("session", "show me more", 4, 10)
            self.assertEqual(state.accumulated_slots["brand"], "nike")
            self.assertEqual(state.accumulated_slots["feature"], "machine washable")

    def test_unrecognized_subject_terms_persist_across_declined_answers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self._agent(
                directory,
                [
                    {"parent_asin": "A", "title": "Women's mules and clogs"},
                    {"parent_asin": "B", "title": "Women's running shoes"},
                ],
            )
            agent.reset("session", {})
            agent.respond("session", "I'm looking for Shoes Mules & Clogs, but I'm still exploring.", 1, 10)
            agent.respond("session", "I don't have an additional preference for use_case.", 2, 10)
            trace = agent.debug_session("session")["retrieval_history"][-1]
            self.assertEqual(trace["subject_terms"], ["shoes", "mules", "clogs"])
            self.assertIn("mules", trace["query_terms"])
            self.assertNotIn("don", trace["query_terms"])

    def test_override_message_is_not_bound_to_previous_clarification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self._agent(directory, [{"parent_asin": "A", "title": "leather boots"}])
            agent.reset("session", {})
            agent.respond("session", "I'm looking for boots, but I'm still exploring.", 1, 10)
            state = agent.sessions["session"]
            state.last_asked_attribute = "brand"
            agent.respond(
                "session",
                "Actually, ignore my earlier preference. What I need is: leather.",
                3,
                10,
            )
            self.assertNotIn("brand", state.accumulated_slots)
            self.assertEqual(state.accumulated_slots["material"], "leather")

    def test_global_override_reopens_clarification_attributes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self._agent(directory, [{"parent_asin": "A", "title": "leather boots"}])
            agent.reset("session", {})
            state = agent.sessions["session"]
            state.asked_attributes.extend(["material", "feature"])
            state.declined_attributes.add("color")
            agent.respond(
                "session",
                "Actually, ignore my earlier preference. What I need is: leather.",
                3,
                10,
            )
            self.assertNotIn("material", state.asked_attributes)
            self.assertEqual(state.asked_attributes.count("feature"), 1)
            self.assertEqual(state.last_asked_attribute, "feature")
            self.assertNotIn("color", state.declined_attributes)

    def test_global_override_reopens_previously_shown_products(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self._agent(
                directory,
                [
                    {"parent_asin": str(index), "title": "leather walking shoe"}
                    for index in range(4)
                ],
            )
            agent.reset("session", {})
            first = agent.respond("session", "walking shoes", 1, 2)
            override = agent.respond(
                "session",
                "Actually, ignore my earlier preference. What I need is: leather.",
                2,
                2,
            )
            first_ids = {item["parent_asin"] for item in first["recommendations"]}
            override_ids = {item["parent_asin"] for item in override["recommendations"]}
            self.assertTrue(first_ids & override_ids)

    def test_declined_attribute_is_neutral_not_negated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self._agent(directory, [{"parent_asin": "A", "title": "cotton jacket"}])
            agent.reset("session", {})
            state = agent.sessions["session"]
            state.last_asked_attribute = "material"
            agent.respond(
                "session",
                "I don't have a preference for material; please use your judgment.",
                2,
                10,
            )
            self.assertIn("material", state.declined_attributes)
            self.assertNotIn("material", state.accumulated_slots)
            self.assertFalse(state.negated_slots)

    def test_feature_clarification_does_not_create_incidental_slots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self._agent(directory, [{"parent_asin": "A", "title": "pendant necklace"}])
            agent.reset("session", {})
            state = agent.sessions["session"]
            state.last_asked_attribute = "feature"
            agent.respond(
                "session",
                "For that, what matters is: a moon symbol representing phases in the life of women.",
                2,
                10,
            )
            self.assertIn("life of women", state.accumulated_slots["feature"])
            self.assertNotIn("gender", state.accumulated_slots)

    def test_feature_coverage_handles_catalog_list_boundaries(self) -> None:
        self.assertEqual(
            Agent._term_coverage("imported; rubber sole", "Imported Rubber sole"),
            1.0,
        )

    def test_continuation_rotates_previously_shown_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self._agent(
                directory,
                [
                    {"parent_asin": str(index), "title": "walking shoe"}
                    for index in range(4)
                ],
            )
            agent.reset("session", {})
            first = agent.respond("session", "walking shoes", 1, 2)
            second = agent.respond("session", "show me more", 2, 2)
            first_ids = {item["parent_asin"] for item in first["recommendations"]}
            second_ids = {item["parent_asin"] for item in second["recommendations"]}
            self.assertFalse(first_ids & second_ids)


if __name__ == "__main__":
    unittest.main()
