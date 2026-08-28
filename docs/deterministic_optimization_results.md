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
| Technical score | 0.784767 | 0.845444 | +0.060677 |
| Hit Rate@10 | 0.945000 | 0.995000 | +0.050000 |
| MRR | 0.544224 | 0.621480 | +0.077256 |
| MTTC | 3.550000 | 2.925000 | -0.625000 |

### Scenario metrics

| Scenario | Hit Rate@10 | MRR | MTTC |
| --- | ---: | ---: | ---: |
| Boundary | 1.000000 | 0.662500 | 3.000000 |
| Browsing | 1.000000 | 0.627703 | 2.675000 |
| Buying | 0.987500 | 0.606815 | 2.400000 |
| Intent Override | 1.000000 | 0.630317 | 4.966667 |

## Changes retained

- Added selectable clarification policies and measured them rather than assuming
  that two unconditional `other` questions were optimal.
- Selected a typed-first, batched-second policy. If a batched reply contains two
  values and the target is still unresolved, one final batch is allowed. This
  recovered all Boundary sessions with repeated hidden constraints.
- Parses multi-value clarification replies and classifies values returned for
  `other` into usable material, color, size, style, use-case, budget, or feature slots.
- Moved ranking, retrieval, and clarification weights into `AgentConfig`.
- Converged the sparse ranking weight from `0.60` through a coordinate and narrow
  sweep to the accepted value `0.75`.
- Reduced the recall pool from 500 to 300 and the active retrieval lanes from
  three to one after the full evaluator retained 0.990 Hit Rate@10.
- Added a bounded cross-session retrieval cache and runtime/query counters.
- Added a reproducible prebuilt SQLite catalog index with schema/catalog validation
  and an automatic build-from-catalog fallback.
- Prevented dimension phrases such as `up to 8-inch` from being interpreted as
  price constraints. This recovered one general parser-driven ranking miss.

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
- Retains a fixed top-100 BM25 score gradient while retrieving a 300-item sparse
  pool. The evaluator still contains 200 sessions and only Top 10 is returned.

## Rejected experiment

Asking for `feature` before `material` reduced the technical score from 0.606205
to 0.604169 and weakened Intent Override performance. The material-first policy
was retained.

A stronger tiered weight formula with subject-term bonuses reduced the score from
the semantic-model checkpoint of 0.641629 to 0.639685 and lowered Intent Override
Hit Rate. The proven sparse/facet weights were retained.

## Runtime sample

Building the 50,000-product derived index took 15.77 seconds in the current Linux
environment; building from the catalog during evaluator startup took 14.94 seconds.
Loading the artifact into memory took 0.23–0.30 seconds. The accepted full run
performed 501 FTS queries for 584 responses and completed evaluation in 33.98
seconds. Peak resident memory was 978588 KiB (about 956 MiB), including the
in-memory SQLite copy and evaluator catalog structures. Wall-clock timings vary
under CPU contention, so metrics and query counts remain the reproducible signals.

The available alternate `new_set.jsonl` retained 0.960 Hit Rate@10 and scored
0.826838. Relative to the committed parser baseline, one Intent Override session
converted two turns earlier at a lower target rank; aggregate technical score moved
by -0.000149 while Hit Rate was unchanged.

## Clarification policy evidence

On the fixed 20-session scenario-balanced screen:

| Policy | Technical score | Hit@10 | MRR | MTTC |
| --- | ---: | ---: | ---: | ---: |
| Current typed | 0.859167 | 1.000 | 0.677222 | 3.200 |
| Other first | 0.810559 | 1.000 | 0.505198 | 3.050 |
| Other second | 0.860542 | 1.000 | 0.675139 | 3.100 |
| Confidence gated | 0.730310 | 0.900 | 0.451032 | 3.750 |
| Two unconditional batches | 0.743310 | 0.900 | 0.491032 | 3.700 |

The evaluator does expose up to two unknown constraints per `other` response, but
asking two unconditional batches lowered ranking quality and created ten-turn
misses. Typed-first plus conditional batching generalized better and produced the
accepted full-set checkpoint above.

## Remaining deterministic targets

- Keep the remaining `public_0020` ambiguity/ranking miss diagnosed unless a
  general change improves both the public and alternate sets.
- Re-run on any recovered historical diverse datasets before submission.
- Re-measure peak memory on the intended submission host.
- Confirm whether package rules allow the ignored 202 MiB prebuilt index as a
  separately disclosed artifact; otherwise use the build-from-catalog fallback.
- Keep dense retrieval deferred until a separate, optional experiment is
  explicitly approved.
