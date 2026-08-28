from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from time import perf_counter

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl, metric_summary
from starter.agent import Agent
from starter.config import AgentConfig


POLICIES = (
    "current", "other_first", "other_second", "typed_only",
    "confidence_gated", "two_batch",
)
FACET_FIELDS = (
    "category_weight", "gender_weight", "material_weight", "color_weight",
    "size_weight", "style_weight", "use_case_weight", "brand_weight", "feature_weight",
)


def technical_metrics(sessions: list[dict]) -> dict:
    result = metric_summary(sessions)
    efficiency = max(0.0, min(1.0, (11.0 - float(result["mttc"])) / 10.0))
    result["efficiency"] = round(efficiency, 6)
    result["recommended_technical_score"] = round(
        0.50 * result["hit_rate_at_10"] + 0.30 * result["mrr"] + 0.20 * efficiency,
        6,
    )
    return result


def stratified_folds(samples: list[dict], fold_count: int) -> list[set[str]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        grouped[str(sample["scenario_type"])].append(sample)
    folds = [set() for _ in range(fold_count)]
    for scenario in sorted(grouped):
        ordered = sorted(grouped[scenario], key=lambda item: str(item["sample_id"]))
        for index, sample in enumerate(ordered):
            folds[index % fold_count].add(str(sample["sample_id"]))
    return folds


def stratified_limit(samples: list[dict], limit: int) -> list[dict]:
    if limit <= 0 or limit >= len(samples):
        return list(samples)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        grouped[str(sample["scenario_type"])].append(sample)
    selected: list[dict] = []
    scenarios = sorted(grouped)
    while len(selected) < limit:
        progressed = False
        for scenario in scenarios:
            if grouped[scenario]:
                selected.append(grouped[scenario].pop(0))
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
    return selected


def scaled_facets(config: AgentConfig, scale: float) -> AgentConfig:
    return replace(config, **{
        field: round(float(getattr(config, field)) * scale, 6)
        for field in FACET_FIELDS
    })


def coarse_configs(base: AgentConfig) -> list[tuple[str, AgentConfig]]:
    candidates: list[tuple[str, AgentConfig]] = []
    for sparse, facet_scale, mismatch in product(
        (0.50, 0.60, 0.70), (0.85, 1.0, 1.15), (0.25, 0.35, 0.45)
    ):
        config = scaled_facets(replace(
            base,
            clarification_policy="current",
            sparse_weight=sparse,
            mismatch_penalty_multiplier=mismatch,
        ), facet_scale)
        candidates.append((f"coarse-s{sparse:.2f}-f{facet_scale:.2f}-m{mismatch:.2f}", config))
    return candidates


def coordinate_configs(name: str, base: AgentConfig) -> list[tuple[str, AgentConfig]]:
    fields = (
        "sparse_weight", "category_weight", "material_weight", "color_weight",
        "feature_weight", "mismatch_penalty_multiplier", "semantic_hit_bonus",
        "price_match_bonus", "unseen_product_bonus", "candidate_diversity_weight",
    )
    candidates = [(name, base)]
    for field in fields:
        value = float(getattr(base, field))
        for scale in (0.85, 1.15):
            candidates.append((f"{name}-{field}-{scale:.2f}", replace(base, **{field: round(value * scale, 6)})))
    return candidates


def policy_configs(base: AgentConfig, prefix: str = "policy") -> list[tuple[str, AgentConfig]]:
    return [(f"{prefix}-{policy}", replace(base, clarification_policy=policy)) for policy in POLICIES]


def runtime_configs(base: AgentConfig) -> list[tuple[str, AgentConfig]]:
    return [
        (
            f"runtime-lanes{lanes}-pool{pool}",
            replace(base, max_retrieval_lanes=lanes, recall_pool_size=pool),
        )
        for lanes, pool in product((1, 2, 3), (300, 500))
    ]


def sparse_configs(base: AgentConfig) -> list[tuple[str, AgentConfig]]:
    return [
        (f"sparse-{value:.2f}", replace(base, sparse_weight=value))
        for value in (0.60, 0.63, 0.66, 0.69, 0.72, 0.75, 0.78)
    ]


def result_key(record: dict) -> tuple:
    overall = record["overall"]
    return (
        overall["recommended_technical_score"],
        overall["hit_rate_at_10"],
        overall["mrr"],
        -overall["mttc"],
        -record["runtime"]["evaluation_seconds"],
    )


def evaluate_candidates(
    agent: Agent,
    candidates: list[tuple[str, AgentConfig]],
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    folds: list[set[str]],
) -> tuple[list[dict], dict[str, list[dict]]]:
    records: list[dict] = []
    sessions_by_name: dict[str, list[dict]] = {}
    previous_retrieval_signature: tuple | None = None
    for index, (name, config) in enumerate(candidates, 1):
        retrieval_signature = (
            config.fts_weights,
            config.max_retrieval_lanes,
            config.recall_pool_size,
        )
        agent.configure(
            config,
            clear_cache=retrieval_signature != previous_retrieval_signature,
        )
        previous_retrieval_signature = retrieval_signature
        started = perf_counter()
        raw = evaluate(agent, samples, catalog_ids, categories, products)
        elapsed = perf_counter() - started
        sessions = raw["sessions"]
        sessions_by_name[name] = sessions
        fold_metrics = [
            technical_metrics([item for item in sessions if str(item["sample_id"]) in fold])
            for fold in folds
        ]
        record = {
            "name": name,
            "config": asdict(config),
            "overall": technical_metrics(sessions),
            "folds": fold_metrics,
            "mean_fold_score": round(statistics.fmean(item["recommended_technical_score"] for item in fold_metrics), 6),
            "worst_fold_score": min(item["recommended_technical_score"] for item in fold_metrics),
            "scenario_metrics": raw["scenario_metrics"],
            "runtime": {**raw["runtime"], "evaluation_seconds": round(elapsed, 6)},
            "agent_runtime": dict(agent.runtime_stats),
        }
        records.append(record)
        print(
            f"[{index}/{len(candidates)}] {name}: "
            f"score={record['overall']['recommended_technical_score']:.6f} "
            f"hit={record['overall']['hit_rate_at_10']:.3f} "
            f"mrr={record['overall']['mrr']:.6f} "
            f"mttc={record['overall']['mttc']:.3f} time={elapsed:.2f}s",
            flush=True,
        )
    records.sort(key=result_key, reverse=True)
    return records, sessions_by_name


def cross_validation_selection(records: list[dict], sessions_by_name: dict[str, list[dict]], folds: list[set[str]]) -> list[dict]:
    selections: list[dict] = []
    all_ids = set().union(*folds)
    for fold_index, holdout in enumerate(folds):
        train_ids = all_ids - holdout
        ranked: list[tuple[tuple, str, dict]] = []
        for record in records:
            sessions = sessions_by_name[record["name"]]
            train = technical_metrics([item for item in sessions if str(item["sample_id"]) in train_ids])
            key = (
                train["recommended_technical_score"], train["hit_rate_at_10"],
                train["mrr"], -train["mttc"],
            )
            ranked.append((key, record["name"], train))
        ranked.sort(reverse=True)
        _, selected_name, train_metrics = ranked[0]
        validation = technical_metrics([
            item for item in sessions_by_name[selected_name]
            if str(item["sample_id"]) in holdout
        ])
        selections.append({
            "holdout_fold": fold_index,
            "selected": selected_name,
            "training": train_metrics,
            "validation": validation,
        })
    return selections


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic policy and ranking-weight optimizer")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="experiment_results.json")
    parser.add_argument("--stage", choices=("policies", "runtime", "sparse", "coarse", "coordinate", "all"), default="policies")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--sample-limit", type=int, default=0, help="Stratified quick-screen size; zero uses all sessions")
    parser.add_argument("--top-configs", type=int, default=3)
    parser.add_argument("--base-config-json", help="JSON object or file with AgentConfig overrides")
    parser.add_argument("--policies", help="Comma-separated policy subset for the policies stage")
    parser.add_argument("--scenario", help="Optional scenario_type filter")
    parser.add_argument("--include-sample-ids", help="Comma-separated samples always included after limiting")
    args = parser.parse_args()

    all_samples = load_jsonl(args.dataset)
    if args.scenario:
        all_samples = [
            sample for sample in all_samples
            if str(sample.get("scenario_type")) == args.scenario
        ]
        if not all_samples:
            raise ValueError(f"No samples found for scenario {args.scenario!r}")
    samples = stratified_limit(all_samples, args.sample_limit)
    if args.include_sample_ids:
        required = {value.strip() for value in args.include_sample_ids.split(",") if value.strip()}
        known = {str(sample["sample_id"]) for sample in samples}
        samples.extend(
            sample for sample in all_samples
            if str(sample["sample_id"]) in required and str(sample["sample_id"]) not in known
        )
        missing = required - {str(sample["sample_id"]) for sample in samples}
        if missing:
            raise ValueError(f"Unknown sample ids: {', '.join(sorted(missing))}")
    folds = stratified_folds(samples, args.folds)
    catalog_ids, categories, products = catalog_index(args.catalog)
    initialization_started = perf_counter()
    agent = Agent(args.catalog)
    initialization_seconds = perf_counter() - initialization_started
    base = AgentConfig()
    if args.base_config_json:
        candidate = Path(args.base_config_json)
        payload = candidate.read_text(encoding="utf-8") if candidate.exists() else args.base_config_json
        overrides = json.loads(payload)
        if not isinstance(overrides, dict):
            raise ValueError("--base-config-json must contain a JSON object")
        base = base.with_overrides(**overrides)

    if args.stage == "policies":
        requested = tuple(value.strip() for value in args.policies.split(",")) if args.policies else POLICIES
        unknown = sorted(set(requested) - set(POLICIES))
        if unknown:
            raise ValueError(f"Unknown policies: {', '.join(unknown)}")
        candidates = [(f"policy-{policy}", replace(base, clarification_policy=policy)) for policy in requested]
    elif args.stage == "runtime":
        candidates = runtime_configs(base)
    elif args.stage == "sparse":
        candidates = sparse_configs(base)
    elif args.stage == "coarse":
        candidates = coarse_configs(base)
    elif args.stage == "coordinate":
        candidates = coordinate_configs("baseline", base)
    else:
        coarse_records, _ = evaluate_candidates(
            agent, coarse_configs(base), samples, catalog_ids, categories, products, folds
        )
        top = coarse_records[: max(1, args.top_configs)]
        candidates = []
        for record in top:
            config = AgentConfig(**record["config"])
            candidates.extend(policy_configs(config, record["name"]))

    records, sessions_by_name = evaluate_candidates(
        agent, candidates, samples, catalog_ids, categories, products, folds
    )
    best = records[0]
    if args.stage == "all":
        coordinate = coordinate_configs(best["name"], AgentConfig(**best["config"]))
        coordinate_records, coordinate_sessions = evaluate_candidates(
            agent, coordinate, samples, catalog_ids, categories, products, folds
        )
        records.extend(coordinate_records)
        sessions_by_name.update(coordinate_sessions)
        records.sort(key=result_key, reverse=True)
        best = records[0]

    report = {
        "dataset": args.dataset,
        "sample_count": len(samples),
        "fold_count": len(folds),
        "initialization_seconds": round(initialization_seconds, 6),
        "best": best,
        "cross_validation_selection": cross_validation_selection(records, sessions_by_name, folds),
        "ranked_candidates": records,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best": best, "output": args.output}, indent=2), flush=True)


if __name__ == "__main__":
    main()
