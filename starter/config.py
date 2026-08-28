from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Literal


ClarificationPolicy = Literal[
    "current",
    "other_first",
    "other_second",
    "typed_only",
    "confidence_gated",
    "two_batch",
]


@dataclass(frozen=True)
class AgentConfig:
    """Deterministic retrieval, ranking, and dialogue parameters.

    Keeping these values outside the agent logic makes experiments reproducible
    and prevents a winning configuration from becoming a collection of magic
    constants spread across the implementation.
    """

    clarification_policy: ClarificationPolicy = "other_second"

    # SQLite FTS5 column weights: parent_asin, title, categories, features,
    # details, store, description.
    fts_asin_weight: float = 0.0
    fts_title_weight: float = 6.0
    fts_category_weight: float = 4.0
    fts_feature_weight: float = 2.5
    fts_detail_weight: float = 2.5
    fts_store_weight: float = 1.5
    fts_description_weight: float = 1.0

    sparse_weight: float = 0.75
    category_weight: float = 0.24
    gender_weight: float = 0.16
    material_weight: float = 0.18
    color_weight: float = 0.18
    size_weight: float = 0.14
    style_weight: float = 0.12
    use_case_weight: float = 0.12
    brand_weight: float = 0.14
    feature_weight: float = 0.12
    exact_match_multiplier: float = 1.0
    partial_match_multiplier: float = 0.50
    mismatch_penalty_multiplier: float = 0.35
    feature_mismatch_multiplier: float = 0.15
    negation_penalty: float = 0.25
    multi_match_bonus: float = 0.01
    multi_match_bonus_cap: float = 0.04
    missing_facet_penalty: float = 0.03
    price_match_bonus: float = 0.14
    target_price_bonus: float = 0.16
    price_mismatch_penalty: float = 0.30
    unknown_price_penalty: float = 0.05
    semantic_hit_bonus: float = 0.03
    semantic_bonus_cap: float = 0.15
    unseen_product_bonus: float = 10.0

    clarification_feature_value: float = 0.80
    clarification_material_value: float = 0.85
    clarification_color_value: float = 0.65
    clarification_style_value: float = 0.45
    clarification_use_case_value: float = 0.40
    clarification_size_value: float = 0.30
    clarification_budget_value: float = 0.25
    clarification_brand_value: float = 0.10
    clarification_category_value: float = 0.10
    candidate_diversity_weight: float = 0.15
    confidence_candidate_threshold: int = 50
    confidence_constraint_threshold: int = 2
    two_batch_question_limit: int = 2

    recall_pool_size: int = 300
    sparse_rank_window: int = 100
    max_retrieval_lanes: int = 1
    retrieval_cache_size: int = 512

    @property
    def fts_weights(self) -> tuple[float, ...]:
        return (
            self.fts_asin_weight,
            self.fts_title_weight,
            self.fts_category_weight,
            self.fts_feature_weight,
            self.fts_detail_weight,
            self.fts_store_weight,
            self.fts_description_weight,
        )

    @property
    def facet_weights(self) -> dict[str, float]:
        return {
            "category": self.category_weight,
            "gender": self.gender_weight,
            "material": self.material_weight,
            "color": self.color_weight,
            "size": self.size_weight,
            "style": self.style_weight,
            "use_case": self.use_case_weight,
            "brand": self.brand_weight,
            "feature": self.feature_weight,
        }

    @property
    def clarification_values(self) -> dict[str, float]:
        return {
            "feature": self.clarification_feature_value,
            "material": self.clarification_material_value,
            "color": self.clarification_color_value,
            "style": self.clarification_style_value,
            "use_case": self.clarification_use_case_value,
            "size": self.clarification_size_value,
            "budget": self.clarification_budget_value,
            "brand": self.clarification_brand_value,
            "category": self.clarification_category_value,
        }

    def with_overrides(self, **values: object) -> "AgentConfig":
        valid = {item.name for item in fields(self)}
        unknown = sorted(set(values) - valid)
        if unknown:
            raise ValueError(f"Unknown AgentConfig fields: {', '.join(unknown)}")
        return replace(self, **values)
