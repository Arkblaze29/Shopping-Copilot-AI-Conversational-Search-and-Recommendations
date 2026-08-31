from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from time import perf_counter

from starter.agent import Agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the reproducible catalog-derived SQLite index")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--output", default="data/catalog_index.sqlite")
    args = parser.parse_args()

    catalog = Path(args.catalog)
    output = Path(args.output)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()

    started = perf_counter()
    agent = Agent(catalog, use_prebuilt_index=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    destination = sqlite3.connect(temporary)
    agent.connection.backup(destination)
    destination.close()
    temporary.replace(output)
    elapsed = perf_counter() - started
    print(f"Built {output} in {elapsed:.3f}s ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
