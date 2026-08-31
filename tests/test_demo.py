from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import demo


class FakeAgent:
    def __init__(self) -> None:
        self.reset_calls: list[tuple[str, dict]] = []
        self.respond_calls: list[tuple[str, str, int, int]] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.reset_calls.append((session_id, user_profile))

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        self.respond_calls.append((session_id, user_message, turn, top_k))
        return {
            "message": "I found some options. Do you have a preference for material?",
            "ask_attribute": "material",
            "recommendations": [{"parent_asin": "A"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    def debug_session(self, session_id: str) -> dict[str, object]:
        return {
            "intent": "buying",
            "active_slots": {"category": "boots", "color": "black"},
            "negated_slots": {"material": ["suede"]},
        }


def scripted_input(*messages: str):
    values = iter(messages)
    return lambda _prompt: next(values)


class DemoTest(unittest.TestCase):
    def test_load_product_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.jsonl"
            catalog.write_text(
                json.dumps({"parent_asin": "A", "title": "Black boots", "price": "79.95"}) + "\n",
                encoding="utf-8",
            )
            products = demo.load_product_summaries(catalog)
            self.assertEqual(products["A"].title, "Black boots")
            self.assertEqual(products["A"].price, 79.95)

    def test_missing_catalog_returns_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.jsonl"
            output = StringIO()
            errors = StringIO()
            with redirect_stdout(output), redirect_stderr(errors):
                result = demo.main(["--catalog", str(missing)])
            self.assertEqual(result, 2)
            self.assertIn("Catalog not found", errors.getvalue())
            self.assertIn("experiments.build_index", errors.getvalue())

    def test_two_turns_reuse_session_and_increment_turn(self) -> None:
        agent = FakeAgent()
        output: list[str] = []
        demo.run_interactive(
            agent,
            {"A": demo.ProductSummary("Black boots", 79.95)},
            input_fn=scripted_input("black boots", "leather", ""),
            output_fn=output.append,
            session_id_iterator=iter(("session-one",)),
        )
        self.assertEqual(len(agent.reset_calls), 1)
        self.assertEqual(
            agent.respond_calls,
            [
                ("session-one", "black boots", 1, 10),
                ("session-one", "leather", 2, 10),
            ],
        )

    def test_new_command_resets_session_and_turn(self) -> None:
        agent = FakeAgent()
        output: list[str] = []
        demo.run_interactive(
            agent,
            {"A": demo.ProductSummary("Black boots", None)},
            input_fn=scripted_input("first request", "/new", "second request", "exit"),
            output_fn=output.append,
            session_id_iterator=iter(("session-one", "session-two")),
        )
        self.assertEqual([call[0] for call in agent.reset_calls], ["session-one", "session-two"])
        self.assertEqual(agent.respond_calls[0][2], 1)
        self.assertEqual(agent.respond_calls[1][0], "session-two")
        self.assertEqual(agent.respond_calls[1][2], 1)

    def test_exit_does_not_call_agent(self) -> None:
        agent = FakeAgent()
        demo.run_interactive(
            agent,
            {"A": demo.ProductSummary("Black boots", None)},
            input_fn=scripted_input("quit"),
            output_fn=lambda _line: None,
            session_id_iterator=iter(("session-one",)),
        )
        self.assertEqual(agent.respond_calls, [])

    def test_output_contains_target_blind_state_and_product_title(self) -> None:
        agent = FakeAgent()
        output: list[str] = []
        demo.run_interactive(
            agent,
            {"A": demo.ProductSummary("Black boots", 79.95)},
            input_fn=scripted_input("black boots", ""),
            output_fn=output.append,
            session_id_iterator=iter(("session-one",)),
        )
        rendered = "\n".join(output)
        self.assertIn("ASK_ATTRIBUTE: material", rendered)
        self.assertIn("INTENT: buying", rendered)
        self.assertIn("category=boots", rendered)
        self.assertIn("material=suede", rendered)
        self.assertIn("Black boots | A | $79.95", rendered)
        self.assertNotIn("hidden target", rendered.lower())
        self.assertNotIn("target_asin", rendered.lower())


if __name__ == "__main__":
    unittest.main()
