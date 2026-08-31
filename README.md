# Constraint-Aware Shopping Copilot

An offline conversational shopping agent for the TechJam Conversational E-Commerce Search Challenge. It turns an evolving shopper message into product constraints, retrieves from the frozen 50,000-item Amazon catalog, ranks exact `parent_asin` candidates, and asks at most one useful follow-up attribute per turn.

**Public development result:** **0.845444 TechnicalScore**, **0.995 Hit Rate@10**, **0.621480 MRR**, and **2.925 MTTC** across the 200 released sessions. The official agent uses **zero runtime tokens**, no external API, and no network access after the catalog is available.

## Why this approach

The challenge rewards both finding the exact purchased product and reaching it quickly. A generic chat flow can waste turns, while basic keyword search can forget preferences or mishandle a change of mind. This project uses an explainable, deterministic decision pipeline:

```text
Shopper message
  -> slot extraction and session state
  -> buying/browsing intent routing
  -> SQLite FTS5 candidate retrieval
  -> facet- and constraint-aware reranking
  -> ranked recommendations + one optional clarification question
```

The system is target-blind at runtime: it never receives the hidden target or evaluator labels while responding.

## Key design decisions

- **Intent-aware retrieval:** Buying requests emphasize explicit category, price, size, material, colour, brand, style, and feature constraints. Browsing requests retain broader subject terms and catalog-derived vocabulary expansion.
- **Stateful conversation:** Each session retains preferences across turns, supports negations, accumulates clarification answers, and clears soft preferences after an explicit intent override.
- **One attribute at a time:** The response always uses one API-supported `ask_attribute` or `null`. The accepted `other_second` policy asks a high-value typed question first, then permits a structured open follow-up only when useful.
- **Deterministic reranking:** Sparse retrieval is combined with category, price, facet, negation, feature-coverage, and unseen-product signals. Recommendations are inspectable and repeatable.
- **Offline by design:** The submission uses Python’s standard library and SQLite FTS5. There are no API keys, model downloads, runtime network calls, or token costs.

## Results

The weak supplied BM25 starter achieved a TechnicalScore of `0.125`. The final agent was evaluated with the official local evaluator on the same public set.

| Metric | Starter baseline | Final agent |
| --- | ---: | ---: |
| TechnicalScore | 0.125000 | **0.845444** |
| Hit Rate@10 | 0.125000 | **0.995000** |
| MRR | 0.068034 | **0.621480** |
| MTTC | 9.810 | **2.925** |
| Runtime tokens | N/A | **0** |

| Scenario | Hit Rate@10 | MRR | MTTC |
| --- | ---: | ---: | ---: |
| Buying | 0.9875 | 0.606815 | 2.400 |
| Browsing | 1.0000 | 0.627703 | 2.675 |
| Intent Override | 1.0000 | 0.630317 | 4.967 |
| Boundary | 1.0000 | 0.662500 | 3.000 |

Detailed optimization evidence is in [docs/deterministic_optimization_results.md](docs/deterministic_optimization_results.md).

## Quick start

### Requirements

- Python **3.10 or later**
- The frozen competition catalog, `data/catalog.jsonl` (50,000 rows)

The official agent has **no third-party Python dependencies**. The catalog is read-only; do not edit it or add mock ASINs.

### 1. Get the catalog

Download `catalog.jsonl.gz` from the [participant-kit release](https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participant-kit). Put it in `data/`, then extract it to `data/catalog.jsonl`.

On macOS/Linux or Git Bash:

```bash
gzip -dk data/catalog.jsonl.gz
```

On Windows, extract the `.gz` file with 7-Zip or another archive tool. Confirm that `data/catalog.jsonl` exists afterwards.

### 2. Build the optional local search index

```bash
python -m experiments.build_index
```

This creates `data/catalog_index.sqlite`, a catalog-derived SQLite FTS5 index. It makes startup faster, is validated against the catalog before use, and can be deleted safely: the agent rebuilds the index in memory when it is absent.

### 3. Run the interactive demo

```bash
python demo.py
```

Type one shopper message per turn. Each message is sent as the next turn in the same session, so the demo shows the real stateful dialogue rather than isolated searches. After every turn it prints:

- the agent's response and its single supported `ask_attribute` (or `none`);
- the current Buying/Browsing intent and compact target-blind active/negated preferences;
- five readable product rows containing rank, title, ASIN, and price when available.

The wrapper requests `top_k=10` from the unchanged official API. It displays only the first five rows to keep a terminal recording easy to read; the full Top 10 is still produced and the evaluator's ranking behavior is unchanged. The exact follow-up attribute can vary because it is chosen from the current conversation state. Use `/new` to reset the session for a new scenario, or `quit`, `exit`, or an empty line to stop.

Recommended video prompts:

```text
I need black leather ankle boots under $100.
I'm exploring ideas for a summer wedding guest outfit.
I'm looking for a red leather handbag.
Actually, ignore my earlier preference. I need a waterproof backpack instead.
```

Use `/new` before each independent scenario. Running `python -m experiments.build_index` first is optional but reduces demo startup time.

For a three-minute recording, show three short decision moments:

1. **Vague occasion request:** demonstrate broad browsing and the agent's one follow-up question.
2. **Precise purchase request:** demonstrate constraint-first retrieval without an unnecessary question.
3. **Intent change:** send a new request that contradicts an earlier preference and show the updated state and recommendations.

Example interaction (the wording of the agent response and question is data-dependent):

```text
YOU: I need black leather ankle boots under $100.
AGENT: <recommendation response>
ASK_ATTRIBUTE: <one attribute or none>
INTENT: Buying
ACTIVE: category=boots, colour=black, material=leather, price_max=100
NEGATED: none
TOP PRODUCTS:
  1. <title> | <ASIN> | $<price>
  ...
```

When moving to the next video moment, type `/new`; this clears the conversation ledger while keeping the catalog and agent loaded.

### 4. Run the tests

```bash
python -m unittest discover -s tests -q
```

### 5. Reproduce the public evaluation

```bash
python -m evaluator.local_evaluator
```

The evaluator writes `results.json`. For target-aware local diagnostics only, use:

```bash
python -m evaluator.local_evaluator --diagnostics --output results_diagnostics.json
```

Diagnostics are evaluator-only; the agent itself never receives target information.

## Submission interface

The entry point is [`starter/agent.py`](starter/agent.py), which exports the required `Agent` class:

```python
from starter.agent import Agent

agent = Agent("data/catalog.jsonl")
agent.reset(session_id, user_profile)
response = agent.respond(session_id, user_message, turn, top_k=10)
```

`respond(...)` returns a message, one allowed `ask_attribute` (or `null`), ordered recommendations, and zero token usage. It complies with the published [Agent API contract](docs/agent_api_contract.json) and preserves the required 10-turn limit.

## Repository layout

```text
demo.py             interactive, target-blind terminal demonstration
starter/
  agent.py          conversation flow, retrieval, ranking, and API entry point
  config.py         reproducible retrieval/ranking/dialogue parameters
  state.py          per-session preference and override state
  semantics.py      catalog facets, aliases, and vocabulary expansion
evaluator/          supplied local scorer and simulator
experiments/
  build_index.py    creates the optional catalog-derived SQLite index
  optimize.py       fixed-fold policy and parameter experiments
tests/              API, state, ranking, evaluator, and optimizer tests
docs/               API contract, rules, and measured optimization evidence
data/               frozen catalog and public development sessions
```

## Evaluation and explainability

The project treats tuning as an evidence problem, not a collection of opaque rules. It records target-blind session diagnostics including active constraints, query terms, candidate pools, ranking choices, question choices, timing, and cache usage. The configuration lives in one place so each result is reproducible.

Several alternatives were tested on fixed scenario-stratified splits. For example, asking two unconditional open questions, feature-first questioning, and stronger tiered subject bonuses all reduced the public technical score. The accepted `other_second` dialogue policy and sparse/facet weights were selected because they improved the full-set result, not because they looked more sophisticated.

A local dense semantic-retrieval prototype was evaluated and deliberately not shipped: it produced plausible substitutes but harmed exact-ASIN recovery, increased turns, and raised latency.

## Limitations and future work

- The public score is based on 200 development sessions; the organizer retains 800 private sessions, so public-set overfitting remains a risk.
- The system uses deterministic lexical and catalog-facet evidence. It does not provide the broader natural-language understanding of a well-calibrated local language model.
- The optional SQLite index speeds startup but requires additional disk space; the documented fallback rebuilds it from the catalog.
- With more time, semantic retrieval would be used only as a tightly gated recall rescue for weak sparse queries, and only shipped after it improves every relevant scenario.

## Tools, data, and cost disclosure

- **Language/runtime:** Python 3.10+ and the Python standard library (`sqlite3` FTS5).
- **Development/evaluation:** supplied TechJam evaluator and frozen public sessions.
- **Data:** frozen 50,000-product `Clothing_Shoes_and_Jewelry` catalog derived from Amazon Reviews 2023. See [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md).
- **Models/APIs:** none in the official agent.
- **Network requirement:** none after the catalog is present.
- **Runtime token cost:** 0.
- **Estimated API cost:** $0.

## Competition compliance

The catalog remains read-only. The repository contains no private evaluation data, credentials, or required external service. The official evaluator is not modified by the agent, and the output follows the competition’s published [submission rules](docs/submission_rules.md).
