# Deterministic Optimization Results

This checkpoint keeps API reranking and user-profile ranking disabled. The agent
uses SQLite FTS5 retrieval, structured session state, semantic facets, and a
deterministic local reranker.

## Reproduction

Run the standard evaluator:

```bash
python3 -m evaluator.local_evaluator
```

Run target-aware local diagnostics without exposing the target to the agent:

```bash
python3 -m evaluator.local_evaluator \
  --diagnostics \
  --output results_diagnostics.json
```

The diagnostic output records whether the target entered the sparse recall pool,
its sparse and reranked positions, active slots, query terms, and clarification
choices. Diagnostic fields are evaluator-only and do not change the agent API.

## Accepted checkpoint

| Metric | Start of pass | Accepted result | Change |
| --- | ---: | ---: | ---: |
| Technical score | 0.494488 | 0.784767 | +0.290279 |
| Hit Rate@10 | 0.615000 | 0.945000 | +0.330000 |
| MRR | 0.407294 | 0.544224 | +0.136930 |
| MTTC | 7.760000 | 3.550000 | -4.210000 |
| Recall-pool hit rate | 0.845000 | 0.990000 | +0.145000 |

### Scenario metrics

| Scenario | Hit Rate@10 | MRR | MTTC |
| --- | ---: | ---: | ---: |
| Boundary | 1.000000 | 0.606508 | 3.900000 |
| Browsing | 0.987500 | 0.575149 | 2.975000 |
| Buying | 0.887500 | 0.468839 | 3.550000 |
| Intent Override | 0.966667 | 0.642024 | 4.966667 |

## Changes retained

- Preserved unrecognized product/category subject terms across turns.
- Removed declined-preference and retry boilerplate from retrieval queries.
- Prevented intent-override messages from becoming clarification answers.
- Scoped clarification answers to the attribute that was asked.
- Reopened clarification choices after a global intent override.
- Ranked clarification attributes by expected value with candidate diversity as
  a secondary signal.
- Added token-coverage scoring for multi-part feature values whose catalog text
  crosses list or punctuation boundaries.
- Added field-aware subtype, manufacturer/brand, and normalized size facets.
- Treats a continued session as implicit rejection of already-shown products and
  rotates unseen candidates ahead of repeats.
- Clears shown-product rejection history when the user globally overrides intent.
- Keeps declined clarification attributes neutral while preventing repeated questions.
- Expanded the internal sparse pool to 500 while retaining a fixed top-100 BM25
  score gradient. The evaluator still contains 200 sessions and only Top 10 is returned.

## Rejected experiment

Asking for `feature` before `material` reduced the technical score from 0.606205
to 0.604169 and weakened Intent Override performance. The material-first policy
was retained.

A stronger tiered weight formula with subject-term bonuses reduced the score from
the semantic-model checkpoint of 0.641629 to 0.639685 and lowered Intent Override
Hit Rate. The proven sparse/facet weights were retained.

## Runtime sample

On the current development machine, catalog initialization took approximately
11.43 seconds and one measured first response took approximately 0.124 seconds.
These are local development measurements, not guaranteed deployment limits.

## Remaining deterministic targets

- Improve ordering for the nine remaining diagnostic ranking misses.
- Improve structured/category recall for the two remaining retrieval misses.
- Measure memory usage with an available platform profiler.
- Keep dense retrieval deferred until a separate, optional experiment is
  explicitly approved.
