# TechJam Agent Optimization Handoff

## Current status

The implementation is runnable and verified on 28 August 2026.

- `42` unit tests pass.
- Python compilation passes for the agent, evaluator, and experiment tools.
- Real-catalog Boundary smoke evaluation passes all 10 sessions:
  - Hit Rate@10: `1.000`
  - MRR: `0.662500`
  - MTTC: `3.000`
  - Technical score: `0.858750`
- No runtime LLM or network dependency was added.
- The public Agent API remains `reset(...)` and `respond(...)`.

The accepted full 200-session result is:

| Metric | Previous checkpoint | Current checkpoint |
| --- | ---: | ---: |
| Technical score | 0.784767 | **0.845444** |
| Hit Rate@10 | 0.945 | **0.995** |
| MRR | 0.544224 | **0.621480** |
| MTTC | 3.550 | **2.925** |

Scenario Hit Rate@10: Boundary `1.000`, Browsing `1.000`, Buying `0.9875`, Intent Override `1.000`.

## What was implemented

- Added `AgentConfig` with explicit retrieval, ranking, clarification, and runtime parameters.
- Added six selectable question policies: `current`, `other_first`, `other_second`, `typed_only`, `confidence_gated`, and `two_batch`.
- Made the measured winner, `other_second`, the default. It asks one high-value typed question, then batches with `other`; a second batch is allowed only after the first batch returned two values and the target remains unresolved.
- Added multi-value clarification parsing and accumulation. Values returned for `other` are classified into usable slots.
- Added a continuation guard so `show me more` is not stored as a clarification value.
- Added retrieval caching, FTS query counters, response timings, disclosure diagnostics, and scenario-focused evaluator runs.
- Added a reproducible prebuilt SQLite catalog index. It is validated, copied into memory at startup, and falls back to rebuilding from `catalog.jsonl`.
- Prevented dimensions such as `up to 8-inch` and `at least 18mm` from being parsed as price constraints. This converted `public_0042` without changing the retrieval configuration.
- Added scenario-stratified policy, runtime, coarse, coordinate, and sparse-weight experiment tooling.
- Converged defaults to:
  - clarification policy: `other_second`
  - sparse weight: `0.75`
  - recall pool: `300`
  - retrieval lanes: `1`

The unconditional two-`other` hypothesis was tested and rejected. On the fixed 20-session screen it scored `0.743310`, versus `0.860542` for typed-first/other-second before the Boundary refinement. It exposed information quickly but damaged ranking and caused full-turn misses.

## Important files

- `starter/config.py`: accepted defaults and all tunable values.
- `starter/agent.py`: parsing, policies, retrieval cache/index loading, and deterministic ranking.
- `evaluator/local_evaluator.py`: runtime and information-gain diagnostics plus policy/config/scenario CLI options.
- `experiments/optimize.py`: stratified policy and weight search.
- `experiments/build_index.py`: prebuilt index generator.
- `docs/deterministic_optimization_results.md`: experiment evidence and accepted metrics.

`data/catalog_index.sqlite` currently exists locally and is approximately 202 MiB. It is ignored by Git to prevent accidental repository upload. Build it after obtaining the catalog. Keep it out of Git and include it only as a separately disclosed submission artifact if package size rules permit.

## Reproduction commands

Use the available Python runtime, or ordinary Python 3.10+:

```bash
python -m experiments.build_index
python -m unittest discover -s tests -q
python -m evaluator.local_evaluator
python -m evaluator.local_evaluator --diagnostics --output results_diagnostics.json
```

Focused experiments:

```bash
python -m experiments.optimize --stage policies --sample-limit 20
python -m experiments.optimize --stage runtime --sample-limit 20
python -m experiments.optimize --stage coordinate --sample-limit 20
python -m experiments.optimize --stage sparse --sample-limit 20
```

Experiment output files are ignored by Git. Always validate a screen winner on all 200 sessions before changing defaults.

## Remaining work

1. Run one final full default evaluation after final package assembly and confirm it reproduces `0.845444`.
2. Restore any historical diverse datasets if they are recovered. The available `new_set.jsonl` was evaluated at technical score `0.826838`, Hit Rate@10 `0.960`, MRR `0.633792`, and MTTC `3.165`.
3. Keep `public_0020` as a diagnosed ambiguity/ranking miss unless a general fix improves full-set and diverse-set metrics. The target remains in recall, but the disclosed generic novelty-shirt constraints do not distinguish it within ten Top-10 pages.
4. Re-measure peak memory on the intended submission host. The current Linux run peaked at `978588` KiB (about 956 MiB) while loading the SQLite artifact into memory and retaining evaluator catalog structures.
5. Confirm the submission package size rules. Keep the generated index out of Git; include it separately only if allowed, otherwise use the documented build-from-catalog fallback.
6. Defer local/LLM integration to a later phase, as requested.

## Caveats

- With the prebuilt index, initialization measured `0.23`–`0.30` seconds and the full 200-session evaluation measured about `34` seconds. Without it, initialization measured `14.94` seconds on the current Linux environment.
- Wall-clock evaluation time varies with CPU contention. Use technical metrics and query counts for deterministic comparisons; rerun runtime measurements on the intended submission machine.
- The official evaluator code computes efficiency as `clip((11 - MTTC) / 10, 0, 1)`.
- The prebuilt index is catalog-derived and contains no hidden labels, but its inclusion and reproducibility instructions should be disclosed in the submission.
- Do not tune only against the three known misses. Protect full-set and diverse-set Hit@10 before accepting further weight changes.
