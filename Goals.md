# TechJam Conversational Shopping Agent Goals

## Objective

Build a deterministic, offline-capable conversational shopping agent that identifies the exact target product from the frozen 50,000-item Amazon catalog within the evaluator's ten-turn limit.

The agent should combine structured catalog understanding, sparse retrieval, semantic facet matching, intent-aware ranking, and adaptive clarification while preserving API compliance, reproducibility, and acceptable latency.

The current milestone targets are a checkpoint rather than a final ceiling. Once they are achieved reliably, later development sessions should set higher targets based on measured results.

---

## I. Structured Catalog Understanding and Hybrid Retrieval

### Catalog model

- Parse category paths into normalized department, product type, and subtype facets.
- Extract normalized material, color, size, style, use-case, brand, feature, and price evidence from titles, categories, features, details, stores, and descriptions.
- Build all required indexes and semantic facets in memory from the read-only catalog.

### Retrieval pipeline

Use a multi-route retrieval pipeline:

```text
Intent Routing
    -> Sparse and Structured Candidate Retrieval
    -> Semantic Facet Matching
    -> Deterministic Reranking
    -> Optional Dense or LLM Enhancement
```

- Retain SQLite FTS5 as the reproducible lexical retrieval path.
- Use structured metadata constraints and semantic facet compatibility during reranking.
- Add dense embeddings only after an isolated evaluation demonstrates useful gains, especially for Browsing.
- Treat LLM reranking as an optional enhancement, never as the only working path.
- Maintain a deterministic offline fallback suitable for an evaluation environment without network access.

Any feature requiring an API key, hosted model, external service, or new credential must be discussed with and explicitly approved by the repository owner before implementation. Credentials must never be committed.

---

## II. Buying and Browsing Intent Routing

### Buying track

- Detect explicit product types, price bounds, sizes, colors, materials, gender, brands, and purchase signals.
- Prioritize high-precision filtering and exact metadata evidence.
- Preserve explicit hard constraints throughout the session unless the user changes or negates them.

### Browsing track

- Detect exploratory, situational, occasion-based, and use-case-driven requests.
- Use semantic expansion and broader candidate recall to bridge vocabulary gaps.
- Return a useful range of products rather than near-duplicate recommendations.
- Ask targeted clarification questions when additional information is likely to reduce the candidate space.

The active route may change as the conversation evolves.

---

## III. Multi-Turn Dialog and State Management

- Maintain turn-level provenance for each active search slot.
- Accumulate compatible constraints across turns.
- Apply negations and slot-specific overrides correctly.
- Preserve unrelated constraints when a preference changes.
- Handle global intent overrides by removing stale soft preferences while retaining independent hard requirements.
- Record attributes already asked or explicitly declined.
- Select clarification attributes using candidate value distribution and expected candidate reduction.
- Respect the ten-turn cap and minimize unnecessary conversation.

The provided anonymized aggregate profile may be retained safely, but cross-session identity or long-term personal memory must not be assumed. Profile-based ranking remains optional and must not override explicit current-session intent.

---

## IV. Runtime Adaptation

Runtime adaptation means selecting the appropriate retrieval and clarification strategy from observable session state. It does not require self-modifying code.

The agent should adapt using:

- current Buying or Browsing intent;
- accumulated and overridden constraints;
- candidate-set quality and size;
- attributes already asked or declined;
- remaining turns;
- retrieval confidence and failure signals.

When the current strategy is weak, the agent may broaden retrieval, use semantic expansion, ask a higher-value clarification, or switch routing strategy.

---

## V. Evaluation and Development Gates

The official evaluator remains the authoritative measurement source. Every material retrieval, ranking, extraction, or dialogue-policy change should be tested in isolation where practical.

### Current measured baseline

```text
Technical score:  0.784767
Hit Rate@10:      0.945
MRR:              0.544224
MTTC:             3.550
```

### Current milestone targets

```text
Technical score:  >= 0.50
Hit Rate@10:      >= 0.62
MRR:              >= 0.40
MTTC:             <= 7.20
```

These targets are the next milestone. After reaching them reproducibly, increase the targets in later development sessions rather than treating them as the final objective.

The current deterministic checkpoint exceeds these targets. New targets should be
set with the repository owner before the next optimization phase.

### Required evaluation practice

- Report overall and per-scenario metrics for Buying, Browsing, Intent Override, and Boundary sessions.
- Track initialization time, evaluator runtime, and reported token usage.
- Preserve deterministic results across repeated runs when no external model is enabled.
- Do not accept an aggregate improvement without identifying meaningful scenario regressions.
- Maintain a working offline fallback whenever optional dense or LLM components are tested.

---

## VI. Hard Constraints

- Catalog access is strictly read-only.
- The evaluator API contract must remain unchanged.
- Recommendations must contain valid catalog `parent_asin` values ordered best to worst.
- Only the first ten valid unique recommendations are scored.
- Sessions must complete within ten turns.
- The default submission must run without privileged infrastructure or mandatory network access.
- API keys and secrets must be supplied externally and never committed.

---

## Definition of Success

The current phase succeeds when the milestone targets are achieved reproducibly without violating the API contract, offline fallback requirement, ten-turn limit, or read-only catalog constraint.

The broader project succeeds when the agent continues improving beyond those targets while demonstrating strong performance across all four scenario types, especially semantic Browsing and Intent Override behavior.
