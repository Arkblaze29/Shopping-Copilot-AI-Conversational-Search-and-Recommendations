from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from starter.agent import Agent


MAX_TURNS = 10
DISPLAY_LIMIT = 5
EXIT_COMMANDS = {"quit", "exit"}
NEW_SESSION_COMMAND = "/new"
NEUTRAL_PROFILE = {
    "purchase_frequency": "unknown",
    "average_prior_rating": None,
    "rating_style": "unknown",
    "preference_tags": [],
    "summary": "Interactive demonstration session",
}


@dataclass(frozen=True)
class ProductSummary:
    title: str
    price: float | None


def load_product_summaries(catalog_path: str | Path) -> dict[str, ProductSummary]:
    """Load display-only product information from the frozen catalog."""
    path = Path(catalog_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Catalog not found at {path}. Download and extract it to data/catalog.jsonl."
        )

    products: dict[str, ProductSummary] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                product = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid catalog JSON on line {line_number}: {error.msg}") from error
            asin = str(product.get("parent_asin", "")).strip()
            if not asin:
                raise ValueError(f"Catalog row {line_number} has no parent_asin")
            title = str(product.get("title") or "Untitled product").strip()
            raw_price = product.get("price")
            try:
                price = float(raw_price) if raw_price not in (None, "") else None
            except (TypeError, ValueError):
                price = None
            products[asin] = ProductSummary(title=title, price=price)
    if not products:
        raise ValueError("The catalog is empty")
    return products


def _session_ids() -> Iterator[str]:
    while True:
        yield f"demo-{uuid.uuid4().hex[:12]}"


def _format_value(value: object) -> str:
    if isinstance(value, dict):
        return ", ".join(
            f"{key}:{_format_value(item)}" for key, item in sorted(value.items())
        ) or "none"
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value) or "none"
    return str(value)


def _format_mapping(values: object) -> str:
    if not isinstance(values, dict) or not values:
        return "none"
    return " | ".join(
        f"{key}={_format_value(value)}" for key, value in sorted(values.items())
    )


def render_response(
    response: dict,
    diagnostics: dict[str, object],
    products: dict[str, ProductSummary],
    *,
    output_fn: Callable[[str], None] = print,
) -> None:
    """Render only response data and target-blind session diagnostics."""
    output_fn("")
    output_fn(f"AGENT: {response.get('message', '')}")
    output_fn(f"ASK_ATTRIBUTE: {response.get('ask_attribute') or 'none'}")
    output_fn(f"INTENT: {diagnostics.get('intent') or 'unknown'}")
    output_fn(f"ACTIVE PREFERENCES: {_format_mapping(diagnostics.get('active_slots'))}")
    output_fn(f"NEGATED PREFERENCES: {_format_mapping(diagnostics.get('negated_slots'))}")
    output_fn("")
    output_fn("TOP RECOMMENDATIONS:")
    recommendations = response.get("recommendations")
    if not isinstance(recommendations, list) or not recommendations:
        output_fn("  No matching products found.")
        return
    for rank, recommendation in enumerate(recommendations[:DISPLAY_LIMIT], 1):
        asin = str(recommendation.get("parent_asin", ""))
        product = products.get(asin, ProductSummary("Unknown product", None))
        price = f" | ${product.price:.2f}" if product.price is not None else ""
        output_fn(f"  {rank}. {product.title} | {asin}{price}")


def run_interactive(
    agent: Agent,
    products: dict[str, ProductSummary],
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    session_id_iterator: Iterator[str] | None = None,
) -> int:
    """Run a human-driven conversation without changing the agent API."""
    ids = session_id_iterator or _session_ids()

    def reset_session() -> str:
        session_id = next(ids)
        agent.reset(session_id, dict(NEUTRAL_PROFILE))
        return session_id

    session_id = reset_session()
    turn = 1
    output_fn("Constraint-Aware Shopping Copilot - Interactive Demo")
    output_fn("Type one shopper message per turn. Use /new for a fresh scenario.")
    output_fn("Type quit, exit, or press Enter on an empty line to stop.")

    while turn <= MAX_TURNS:
        output_fn("")
        output_fn("-" * 72)
        output_fn(f"TURN {turn} OF {MAX_TURNS}")
        try:
            user_message = input_fn("YOU: ").strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("")
            output_fn("Demo ended.")
            return 0

        command = user_message.lower()
        if not user_message or command in EXIT_COMMANDS:
            output_fn("Demo ended.")
            return 0
        if command == NEW_SESSION_COMMAND:
            session_id = reset_session()
            turn = 1
            output_fn("Started a fresh demo session.")
            continue

        response = agent.respond(session_id, user_message, turn, top_k=10)
        diagnostics = agent.debug_session(session_id)
        render_response(response, diagnostics, products, output_fn=output_fn)
        turn += 1

    output_fn("")
    output_fn("The 10-turn competition limit has been reached. Start demo.py again for a new run.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive terminal demo for the shopping agent")
    parser.add_argument("--catalog", default="data/catalog.jsonl", help="Path to the frozen JSONL catalog")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    catalog_path = Path(args.catalog)
    print(f"Loading product display data from {catalog_path} ...", flush=True)
    try:
        products = load_product_summaries(catalog_path)
        print("Starting the agent. The optional prebuilt index makes this step faster.", flush=True)
        agent = Agent(catalog_path)
    except (FileNotFoundError, OSError, ValueError, sqlite3.Error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        print(
            "After the catalog is installed, you may run "
            "'python -m experiments.build_index' to speed up future starts.",
            file=sys.stderr,
        )
        return 2
    return run_interactive(agent, products)


if __name__ == "__main__":
    raise SystemExit(main())
