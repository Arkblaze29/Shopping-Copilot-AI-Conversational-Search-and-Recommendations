# TechJam Agent Optimization Handoff

## Current status

The implementation is runnable and verified on 28 August 2026.

- `41` unit tests pass.
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
| Technical score | 0.784767 | **0.840644** |
| Hit Rate@10 | 0.945 | **0.990** |
| MRR | 0.544224 | **0.616480** |
| MTTC | 3.550 | **2.965** |

Scenario Hit Rate@10: Boundary `1.000`, Browsing `1.000`, Buying `0.975`, Intent Override `1.000`.

## What was implemented

- Added `AgentConfig` with explicit retrieval, ranking, clarification, and runtime parameters.
- Added six selectable question policies: `current`, `other_first`, `other_second`, `typed_only`, `confidence_gated`, and `two_batch`.
- Made the measured winner, `other_second`, the default. It asks one high-value typed question, then batches with `other`; a second batch is allowed only after the first batch returned two values and the target remains unresolved.
- Added multi-value clarification parsing and accumulation. Values returned for `other` are classified into usable slots.
- Added a continuation guard so `show me more` is not stored as a clarification value.
- Added retrieval caching, FTS query counters, response timings, disclosure diagnostics, and scenario-focused evaluator runs.
- Added a reproducible prebuilt SQLite catalog index. It is validated, copied into memory at startup, and falls back to rebuilding from `catalog.jsonl`.
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

`data/catalog_index.sqlite` currently exists locally and is approximately 212 MB. It is ignored by Git to prevent accidental repository upload. Build it after obtaining the catalog, or include it separately in the final submission package if package rules permit.

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

1. Run one final full default evaluation after checkout/package assembly and confirm it reproduces `0.840644`.
2. Restore the historical diverse datasets and evaluate the accepted defaults for generalization. They are not in the current workspace.
3. Diagnose the two remaining public-set misses. The previous `0.69` diagnostic run had three ranking misses (`public_0020`, `public_0042`, `public_0052`); increasing sparse weight to `0.75` converted one of them, but the final two IDs were not re-diagnosed.
4. Measure peak memory outside the sandbox. Loading the 212 MB SQLite artifact into memory showed roughly 0.9 GB working-set usage during long local runs.
5. Decide how the generated index will be packaged. The agent remains correct without it, but startup rebuilding took about 79.7 seconds on this machine; loading the artifact normally took under three seconds.
6. Review the working-tree diff and commit the implementation. All current modified/untracked source files belong to this optimization pass.
7. Defer local/LLM integration to a later phase, as requested.

## Caveats

- Wall-clock evaluation time varied heavily because of sandbox CPU contention. Use technical metrics and query counts for deterministic comparisons; rerun runtime measurements on the intended submission machine.
- The official evaluator code computes efficiency as `clip((11 - MTTC) / 10, 0, 1)`.
- The prebuilt index is catalog-derived and contains no hidden labels, but its inclusion and reproducibility instructions should be disclosed in the submission.
- Do not tune only against the three known misses. Protect full-set and diverse-set Hit@10 before accepting further weight changes.
