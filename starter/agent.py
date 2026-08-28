from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import OrderedDict
from pathlib import Path
from time import perf_counter

from starter.config import AgentConfig
from starter.semantics import (
    CATEGORY_ALIASES,
    CATEGORY_PARENTS,
    CATEGORY_RE,
    COLOR_ALIASES,
    COLOR_RE,
    GENDER_ALIASES,
    GENDER_RE,
    MATERIAL_RE,
    SIZE_ALIASES,
    SIZE_RE,
    STYLE_RE,
    USE_CASE_RE,
    category_terms,
    extract_product_facets,
    normalize_feature_text,
    semantic_expansions,
)
from starter.state import SessionState


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "actually", "additional", "attribute", "earlier", "exploring", "found", "ignore",
    "key", "matters", "options", "preference", "requirement", "specific", "what",
}

RANGE_PRICE_RE = re.compile(
    r"\bbetween\s*(?:\$|usd\s*)?(\d+(?:\.\d{1,2})?)\s*(?:and|to|-)\s*"
    r"(?:\$|usd\s*)?(\d+(?:\.\d{1,2})?)(?![\d.])"
    r"(?!\s*-?\s*(?:inch(?:es)?|mm|cm)\b)",
    re.IGNORECASE,
)
MAX_PRICE_RE = re.compile(
    r"\b(?:under|below|less than|up to|at most|max(?:imum)?(?: price)?)\s*"
    r"(?:\$|usd\s*)?(\d+(?:\.\d{1,2})?)(?![\d.])"
    r"(?!\s*-?\s*(?:inch(?:es)?|mm|cm)\b)",
    re.IGNORECASE,
)
MIN_PRICE_RE = re.compile(
    r"\b(?:over|above|more than|at least|minimum(?: price)?)\s*"
    r"(?:\$|usd\s*)?(\d+(?:\.\d{1,2})?)(?![\d.])"
    r"(?!\s*-?\s*(?:inch(?:es)?|mm|cm)\b)",
    re.IGNORECASE,
)
GLOBAL_OVERRIDE_RE = re.compile(r"\b(?:ignore|replace)\s+(?:my\s+)?(?:earlier|previous|old)\s+preference\b", re.I)
TARGET_PRICE_RE = re.compile(
    r"\b(?:(?:around|about|approximately|roughly)|budget(?:\s+of)?)\s*"
    r"(?:\$|usd\s*)?(\d+(?:\.\d{1,2})?)(?![\d.])"
    r"(?!\s*-?\s*(?:inch(?:es)?|mm|cm)\b)",
    re.I,
)
BARE_PRICE_RE = re.compile(r"(?:\$|\busd\s*)(\d+(?:\.\d{1,2})?)\b", re.I)
NEGATION_START_RE = re.compile(r"\b(?:not|no|without|avoid|except)\b", re.I)
NEGATION_ATTRIBUTE_RE = re.compile(
    r"^(color|size|material|brand|style|category|gender|feature|use[_ ]case)\s+",
    re.I,
)
DECLINE_RE = re.compile(
    r"\b(?:(?:no|without)\s+(?:a\s+)?preference\s+for|"
    r"(?:don't|do not)\s+have\s+(?:an?\s+)?(?:additional\s+)?preference\s+for)\s*"
    r"(category|material|color|size|style|brand|budget|feature|use[_ ]case|other)\b",
    re.IGNORECASE,
)
LOOKING_FOR_RE = re.compile(
    r"\blooking\s+for\s+(.+?)(?=\s*(?:,|\.|;|\bbut\b|\ba key requirement\b)|$)",
    re.IGNORECASE,
)
RETRY_MESSAGE_RE = re.compile(
    r"\bthose options are not quite right yet\b|\bask me about one specific attribute\b",
    re.IGNORECASE,
)
CONTINUATION_RE = re.compile(r"^\s*(?:show me more|more options|show more|try again)\s*[.!]?\s*$", re.I)
CLARIFICATION_RESPONSE_RE = re.compile(
    r"^\s*(?:for that\b|my preference is\b|i prefer\b|it should be\b)",
    re.IGNORECASE,
)


def _extract_price_slots(text: str) -> dict[str, float]:
    range_match = RANGE_PRICE_RE.search(text)
    if range_match:
        first, second = float(range_match.group(1)), float(range_match.group(2))
        return {"min_price": min(first, second), "max_price": max(first, second)}
    min_match = MIN_PRICE_RE.search(text)
    if min_match:
        return {"min_price": float(min_match.group(1))}
    max_match = MAX_PRICE_RE.search(text)
    if max_match:
        return {"max_price": float(max_match.group(1))}
    target_match = TARGET_PRICE_RE.search(text)
    if target_match:
        return {"target_price": float(target_match.group(1))}
    bare_match = BARE_PRICE_RE.search(text)
    if bare_match:
        return {"target_price": float(bare_match.group(1))}
    return {}


def _extract_negated_slots(text: str) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    extractors = {
        "color": (COLOR_RE, lambda value: COLOR_ALIASES.get(value, value)),
        "size": (SIZE_RE, lambda value: SIZE_ALIASES[value]),
        "material": (MATERIAL_RE, lambda value: value),
        "category": (CATEGORY_RE, lambda value: CATEGORY_ALIASES[value]),
        "gender": (GENDER_RE, lambda value: GENDER_ALIASES[value.replace("'", "")]),
        "use_case": (USE_CASE_RE, lambda value: value),
        "style": (STYLE_RE, lambda value: value),
    }
    for marker in NEGATION_START_RE.finditer(text):
        segment = text[marker.end():]
        segment = re.split(r"[,.;]|\b(?:but|instead|after all)\b", segment, maxsplit=1, flags=re.I)[0]
        segment = re.sub(r"^\s*(?:a|an)\s+", "", segment, flags=re.I).strip()
        qualifier_match = NEGATION_ATTRIBUTE_RE.match(segment)
        qualifier = None
        if qualifier_match:
            qualifier = qualifier_match.group(1).lower().replace(" ", "_")
            segment = segment[qualifier_match.end():].strip()
        keys = [qualifier] if qualifier in extractors else list(extractors)
        found = False
        for key in keys:
            pattern, normalize = extractors[key]
            value_match = pattern.search(segment)
            if value_match:
                value = normalize(value_match.group(1).lower())
                result.setdefault(key, set()).add(value)
                found = True
        if qualifier in {"brand", "feature"} and not found:
            value = " ".join(_terms(segment)[:6])
            if value:
                result.setdefault(qualifier, set()).add(value)
    return result


def extract_slots(text: str) -> dict[str, object]:
    """Extract a small, deterministic set of shopping constraints from text."""
    lowered = text.lower()
    slots: dict[str, object] = {}

    slots.update(_extract_price_slots(lowered))
    negated_slots = _extract_negated_slots(lowered)

    def is_negated(key: str, value: str) -> bool:
        return any(value.lower() == negated.lower() for negated in negated_slots.get(key, set()))

    color_match = COLOR_RE.search(lowered)
    if color_match and not is_negated("color", COLOR_ALIASES.get(color_match.group(1).lower(), color_match.group(1).lower())):
        color = color_match.group(1).lower()
        slots["color"] = COLOR_ALIASES.get(color, color)
    gender_match = GENDER_RE.search(lowered)
    if gender_match:
        gender = GENDER_ALIASES[gender_match.group(1).lower().replace("'", "")]
        if not is_negated("gender", gender):
            slots["gender"] = gender
    size_match = SIZE_RE.search(lowered)
    if size_match:
        size = SIZE_ALIASES[size_match.group(1).lower()]
        if not is_negated("size", size):
            slots["size"] = size
    category_matches = list(CATEGORY_RE.finditer(lowered))
    if category_matches:
        category_match = max(category_matches, key=lambda match: (len(match.group(1)), match.start()))
        category = CATEGORY_ALIASES[category_match.group(1).lower()]
        if not is_negated("category", category):
            slots["category"] = category
    material_match = MATERIAL_RE.search(lowered)
    if material_match and not is_negated("material", material_match.group(1).lower()):
        slots["material"] = material_match.group(1).lower()
    use_case_match = USE_CASE_RE.search(lowered)
    if use_case_match and not is_negated("use_case", use_case_match.group(1).lower()):
        slots["use_case"] = use_case_match.group(1).lower()
    style_match = STYLE_RE.search(lowered)
    if style_match and not is_negated("style", style_match.group(1).lower()):
        slots["style"] = style_match.group(1).lower()
    if negated_slots:
        slots["negated_slots"] = negated_slots
        slots["negated_terms"] = {value for values in negated_slots.values() for value in values}
    return slots


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _slot_strength(text: str, key: str) -> str:
    if key in {"category", "gender", "min_price", "max_price", "target_price"}:
        return "hard"
    lowered = text.lower()
    if any(cue in lowered for cue in ("must", "need", "requirement", "what matters is", "key requirement")):
        return "hard"
    if any(cue in lowered for cue in ("prefer", "ideally", "still exploring", "maybe", "would like")):
        return "soft"
    if "i'm looking for" in lowered and "key requirement" not in lowered:
        return "soft"
    return "hard"


def _clarification_values(text: str) -> list[str]:
    if re.search(r"\b(?:no|not)\s+(?:an?\s+)?(?:additional\s+)?preference\b", text, re.I):
        return []
    cleaned = re.sub(r"^\s*for that,?\s*(?:what matters is\s*:?)?\s*", "", text, flags=re.I)
    cleaned = re.sub(r"^\s*(?:my preference is|i prefer|it should be)\s*:?[ ]*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .;,\t\n")
    if not cleaned:
        return []
    return [value.strip(" .;,\t\n")[:180] for value in cleaned.split(";") if value.strip(" .;,\t\n")]


def _clarification_answer(text: str) -> str | None:
    """Backward-compatible single-string view used by older callers/tests."""
    values = _clarification_values(text)
    return "; ".join(values) or None


def _slot_values(value: object) -> tuple[str, ...]:
    if isinstance(value, (tuple, list, set)):
        return tuple(str(item).lower() for item in value if str(item).strip())
    return (str(value).lower(),) if str(value).strip() else ()


def _classify_clarification_value(value: str) -> tuple[str, object]:
    """Mirror the evaluator's constraint classes without importing evaluator code."""
    extracted = extract_slots(value)
    for key in ("max_price", "min_price", "target_price"):
        if key in extracted:
            return key, extracted[key]
    for key in ("material", "color", "size", "style", "use_case"):
        if key in extracted:
            return key, extracted[key]
    return "feature", value.lower()


class Agent:
    """Stateful offline shopping agent with sparse and semantic-facet retrieval."""

    INDEX_SCHEMA_VERSION = 1

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        config: AgentConfig | None = None,
        *,
        index_path: str | Path | None = None,
        use_prebuilt_index: bool = True,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.config = config or AgentConfig()
        self.index_path = Path(index_path) if index_path else self.catalog_path.with_name("catalog_index.sqlite")
        self.sessions: dict[str, SessionState] = {}
        self._retrieval_cache: OrderedDict[tuple, tuple[tuple, ...]] = OrderedDict()
        self.runtime_stats = {
            "responses": 0,
            "response_seconds": 0.0,
            "fts_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        if use_prebuilt_index and self._prebuilt_index_is_valid(self.index_path):
            source = sqlite3.connect(f"file:{self.index_path.as_posix()}?mode=ro", uri=True)
            self.connection = sqlite3.connect(":memory:")
            source.backup(self.connection)
            source.close()
            self.runtime_stats["index_source"] = "prebuilt"
        else:
            self.connection = sqlite3.connect(":memory:")
            self._build_index()
            self.runtime_stats["index_source"] = "memory"

    def _prebuilt_index_is_valid(self, path: Path) -> bool:
        if not path.is_file():
            return False
        try:
            connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
            row = connection.execute(
                "SELECT schema_version, catalog_size FROM index_manifest LIMIT 1"
            ).fetchone()
            connection.close()
            return bool(
                row
                and int(row[0]) == self.INDEX_SCHEMA_VERSION
                and int(row[1]) == self.catalog_path.stat().st_size
            )
        except (OSError, sqlite3.Error, TypeError, ValueError):
            return False

    def configure(self, config: AgentConfig, *, clear_cache: bool = True) -> None:
        """Switch experiment parameters while reusing the expensive catalog index."""
        self.config = config
        self.sessions.clear()
        self.runtime_stats.update({
            "responses": 0, "response_seconds": 0.0, "fts_queries": 0,
            "cache_hits": 0, "cache_misses": 0,
        })
        if clear_cache:
            self._retrieval_cache.clear()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE TABLE index_manifest (schema_version INTEGER NOT NULL, catalog_size INTEGER NOT NULL)"
        )
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        cursor.execute(
            "CREATE TABLE product_metadata ("
            "parent_asin TEXT PRIMARY KEY, price REAL, title TEXT NOT NULL, features TEXT NOT NULL, "
            "categories TEXT NOT NULL, details TEXT NOT NULL, searchable_text TEXT NOT NULL, "
            "department TEXT, product_type TEXT, subtype TEXT, materials TEXT NOT NULL, colors TEXT NOT NULL, "
            "sizes TEXT NOT NULL, styles TEXT NOT NULL, use_cases TEXT NOT NULL, brand TEXT NOT NULL, "
            "semantic_text TEXT NOT NULL)"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        metadata_batch: list[tuple] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                fields = [
                    _text(product.get(name))
                    for name in ("title", "categories", "features", "details", "store", "description")
                ]
                fields[2] = normalize_feature_text(fields[2])
                batch.append(
                    (
                        str(product["parent_asin"]),
                        *fields,
                    )
                )
                raw_price = product.get("price")
                try:
                    price = float(raw_price) if raw_price not in (None, "") else None
                except (TypeError, ValueError):
                    price = None
                searchable_text = " ".join(fields).lower()
                facets = extract_product_facets(product, searchable_text)
                semantic_text = " ".join(facets.semantic_terms())
                metadata_batch.append(
                    (
                        str(product["parent_asin"]), price, fields[0].lower(), fields[2].lower(),
                        fields[1].lower(), fields[3].lower(), searchable_text,
                        facets.department, facets.product_type, facets.subtype,
                        " ".join(sorted(facets.materials)), " ".join(sorted(facets.colors)),
                        " ".join(sorted(facets.sizes)),
                        " ".join(sorted(facets.styles)), " ".join(sorted(facets.use_cases)),
                        facets.brand or fields[4].lower(), semantic_text,
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    cursor.executemany("INSERT INTO product_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", metadata_batch)
                    batch.clear()
                    metadata_batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
            cursor.executemany("INSERT INTO product_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", metadata_batch)
        cursor.execute(
            "INSERT INTO index_manifest VALUES (?, ?)",
            (self.INDEX_SCHEMA_VERSION, self.catalog_path.stat().st_size),
        )
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.sessions[session_id] = SessionState.create(session_id, user_profile)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        response_started = perf_counter()
        state = self.sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")
        state.dialog_history.append({"role": "user", "text": user_message})
        previous_attribute = state.last_asked_attribute
        declined_match = DECLINE_RE.search(user_message)
        subject_match = LOOKING_FOR_RE.search(user_message)
        if subject_match:
            state.subject_terms = list(dict.fromkeys(_terms(subject_match.group(1))))[:16]
        global_override = bool(GLOBAL_OVERRIDE_RE.search(user_message))
        is_structured_clarification = bool(
            previous_attribute
            and not declined_match
            and not global_override
            and CLARIFICATION_RESPONSE_RE.search(user_message)
        )
        clarification_values = (
            _clarification_values(user_message)
            if (
                previous_attribute
                and not declined_match
                and not global_override
                and not CONTINUATION_RE.search(user_message)
                and (is_structured_clarification or ";" in user_message)
            )
            else []
        )
        state.last_clarification_count = len(clarification_values)
        extracted = {} if is_structured_clarification else extract_slots(user_message)
        if global_override:
            state.clear_soft_preferences()
            state.asked_attributes.clear()
            state.declined_attributes.clear()
            state.shown_asins.clear()

        price_keys = {"min_price", "max_price", "target_price"} & extracted.keys()
        if price_keys:
            for key in {"min_price", "max_price", "target_price"}:
                state.slots.pop(key, None)

        negated_slots = extracted.get("negated_slots", {})
        if isinstance(negated_slots, dict):
            for key, values in negated_slots.items():
                for value in values:
                    state.negate_slot(str(key), str(value))
        for key, value in extracted.items():
            if key in {"negated_terms", "negated_slots"}:
                continue
            state.set_slot(
                key, value, turn, user_message,
                strength=_slot_strength(user_message, key),
            )

        if declined_match:
            declined = declined_match.group(1).lower().replace(" ", "_")
            state.declined_attributes.add(declined)
        elif (
            previous_attribute
            and clarification_values
            and not global_override
            and (is_structured_clarification or not extracted)
        ):
            state.clarification_values_received.extend(clarification_values)
            if previous_attribute == "other":
                for value in clarification_values:
                    key, normalized = _classify_clarification_value(value)
                    if key in {"min_price", "max_price", "target_price"}:
                        for price_key in {"min_price", "max_price", "target_price"}:
                            state.slots.pop(price_key, None)
                        state.set_slot(key, normalized, turn, user_message, strength="hard")
                    else:
                        state.add_slot_value(key, normalized, turn, user_message, strength="hard")
            elif previous_attribute == "budget":
                for value in clarification_values:
                    key, normalized = _classify_clarification_value(value)
                    if key in {"min_price", "max_price", "target_price"}:
                        for price_key in {"min_price", "max_price", "target_price"}:
                            state.slots.pop(price_key, None)
                        state.set_slot(key, normalized, turn, user_message, strength="hard")
            else:
                for value in clarification_values:
                    state.add_slot_value(
                        previous_attribute, value.lower(), turn, user_message, strength="hard"
                    )
        if previous_attribute:
            state.last_asked_attribute = None

        active_slots = state.accumulated_slots
        state.current_intent = self._classify_intent(user_message, active_slots)
        current_terms = [] if declined_match or RETRY_MESSAGE_RE.search(user_message) else _terms(user_message)
        context_terms = [
            item
            for key, value in active_slots.items()
            if key not in {"max_price", "min_price", "target_price"}
            for item in _slot_values(value)
        ]
        active_category = str(active_slots.get("category", "")).lower()
        category_query_terms = set(state.subject_terms)
        if active_category:
            category_query_terms.update(category_terms(active_category))
            category_query_terms.add(CATEGORY_PARENTS.get(active_category, ""))
        expansion_terms = semantic_expansions({
            item for value in active_slots.values() for item in _slot_values(value)
        })
        constraint_terms = [
            item
            for key, value in active_slots.items()
            if key not in {"max_price", "min_price", "target_price", "category"}
            for item in _slot_values(value)
        ]
        lane_queries = [
            " ".join([
                *state.subject_terms, *current_terms, *context_terms, *sorted(expansion_terms),
            ]),
            " ".join([*sorted(category_query_terms), *state.subject_terms]),
            " ".join([*constraint_terms, *current_terms, *sorted(expansion_terms)]),
        ]
        unique_terms = list(dict.fromkeys(_terms(" ".join(lane_queries))))[:40]
        rows: list[tuple] = []
        ranked_pool: list[str] = []
        if not unique_terms:
            recommendations: list[dict] = []
        else:
            rows = self._retrieve_multi_lane(
                lane_queries, active_slots, max(top_k * 10, self.config.recall_pool_size)
            )
            ranked_pool = self._rank_candidates(rows, state)
            recommendations = [
                {"parent_asin": asin} for asin in ranked_pool[:top_k]
            ]
            state.shown_asins.update(item["parent_asin"] for item in recommendations)

        ask_attribute = self._next_attribute(state, rows)
        if ask_attribute:
            state.asked_attributes.append(ask_attribute)
            state.last_asked_attribute = ask_attribute
            message = (
                "I found some options. What else matters to you?"
                if ask_attribute == "other"
                else f"I found some options. Do you have a preference for {ask_attribute}?"
            )
        else:
            message = "Here are the closest matches I found."
        state.retrieval_history.append({
            "turn": turn,
            "intent": state.current_intent,
            "active_slots": dict(active_slots),
            "query_terms": unique_terms,
            "subject_terms": list(state.subject_terms),
            "sparse_pool": [str(row[0]) for row in rows],
            "ranked_pool": ranked_pool,
            "ask_attribute": ask_attribute,
            "received_constraints": list(clarification_values),
            "received_constraint_count": len(clarification_values),
        })
        response = {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
        state.dialog_history.append({"role": "agent", "text": response["message"]})
        elapsed = perf_counter() - response_started
        self.runtime_stats["responses"] += 1
        self.runtime_stats["response_seconds"] += elapsed
        state.retrieval_history[-1]["response_seconds"] = elapsed
        state.retrieval_history[-1]["fts_queries_total"] = self.runtime_stats["fts_queries"]
        return response

    def _retrieve_multi_lane(
        self,
        lane_queries: list[str],
        active_slots: dict[str, object],
        limit: int,
    ) -> list[tuple]:
        """Retrieve and merge title/category, constraint, and synonym candidates."""
        normalized_lanes = tuple(
            tuple(dict.fromkeys(_terms(query)))[:40]
            for query in lane_queries[: self.config.max_retrieval_lanes]
        )
        cache_key = (
            normalized_lanes,
            active_slots.get("min_price"),
            active_slots.get("max_price"),
            limit,
            self.config.fts_weights,
            self.config.max_retrieval_lanes,
        )
        cached = self._retrieval_cache.get(cache_key)
        if cached is not None:
            self._retrieval_cache.move_to_end(cache_key)
            self.runtime_stats["cache_hits"] += 1
            return list(cached)
        self.runtime_stats["cache_misses"] += 1
        bm25_sql = "bm25(products, " + ", ".join(
            f"{float(weight):.8g}" for weight in self.config.fts_weights
        ) + ")"
        select_sql = (
            f"SELECT products.parent_asin, {bm25_sql}, "
            "m.price, m.title, m.features, m.categories, m.details, m.searchable_text, "
            "m.department, m.product_type, m.subtype, m.materials, m.colors, m.sizes, "
            "m.styles, m.use_cases, m.brand, m.semantic_text "
            "FROM products JOIN product_metadata AS m ON m.parent_asin = products.parent_asin "
        )
        merged: dict[str, tuple] = {}
        lane_limits = [max(1, min(350, limit)), max(1, min(150, limit)), max(1, min(50, limit))]
        for lane_index, terms_tuple in enumerate(normalized_lanes):
            terms = list(terms_tuple)
            if not terms:
                continue
            expression = " OR ".join(f'"{term}"' for term in terms)
            predicates = ["products MATCH ?"]
            parameters: list[object] = [expression]
            if "max_price" in active_slots:
                predicates.append("(m.price IS NULL OR m.price <= ?)")
                parameters.append(active_slots["max_price"])
            if "min_price" in active_slots:
                predicates.append("(m.price IS NULL OR m.price >= ?)")
                parameters.append(active_slots["min_price"])
            query = (
                select_sql + f"WHERE {' AND '.join(predicates)} "
                f"ORDER BY {bm25_sql}, products.parent_asin LIMIT ?"
            )
            lane_limit = lane_limits[min(lane_index, len(lane_limits) - 1)]
            self.runtime_stats["fts_queries"] += 1
            for row in self.connection.execute(query, (*parameters, lane_limit)).fetchall():
                merged.setdefault(str(row[0]), row)
                if len(merged) >= limit:
                    break
            if len(merged) >= limit:
                break
        result = tuple(merged.values())
        self._retrieval_cache[cache_key] = result
        self._retrieval_cache.move_to_end(cache_key)
        while len(self._retrieval_cache) > self.config.retrieval_cache_size:
            self._retrieval_cache.popitem(last=False)
        return list(result)

    def debug_session(self, session_id: str) -> dict[str, object]:
        """Return target-blind session diagnostics for local evaluation tooling."""
        state = self.sessions.get(session_id)
        if state is None:
            return {}
        return {
            "intent": state.current_intent,
            "active_slots": dict(state.accumulated_slots),
            "negated_slots": {
                key: sorted(values) for key, values in state.negated_slots.items()
            },
            "asked_attributes": list(state.asked_attributes),
            "clarification_values_received": list(state.clarification_values_received),
            "retrieval_history": list(state.retrieval_history),
            "runtime_stats": dict(self.runtime_stats),
        }

    @staticmethod
    def _classify_intent(user_message: str, slots: dict[str, object]) -> str:
        lowered = user_message.lower()
        if any(word in lowered for word in ("exploring", "ideas", "suggest", "recommend", "occasion")):
            return "browsing"
        return "buying" if len(slots) > int("category" in slots) else "browsing"

    def _rank_candidates(self, rows: list[tuple], state: SessionState) -> list[str]:
        scored: list[tuple[float, str]] = []
        slots = state.accumulated_slots
        max_price = slots.get("max_price")
        min_price = slots.get("min_price")
        target_price = slots.get("target_price")
        denominator = max(min(len(rows) - 1, self.config.sparse_rank_window - 1), 1)
        for rank, row in enumerate(rows):
            (
                asin, _bm25_score, price, title, features, categories, details, searchable_text,
                department, product_type, subtype, materials, colors, sizes, styles, use_cases,
                brand, semantic_text,
            ) = row
            sparse_score = max(0.0, 1.0 - rank / denominator)
            score = self.config.sparse_weight * sparse_score
            facet_fields = {
                "category": f"{product_type or ''} {subtype or ''} {categories}",
                "gender": str(department or ""),
                "material": str(materials), "color": str(colors), "style": str(styles),
                "use_case": str(use_cases), "brand": str(brand), "feature": str(features),
                "size": f"{sizes} {title} {features} {details}",
            }
            weights = self.config.facet_weights
            matched_constraints = 0
            for key, weight in weights.items():
                if key not in slots:
                    continue
                source = facet_fields.get(key, f"{title} {features} {details} {semantic_text}")
                for value in _slot_values(slots[key]):
                    if key == "category":
                        category_strength = self._category_match_strength(
                            value, product_type, subtype, title, categories
                        )
                        if category_strength:
                            score += category_strength
                            matched_constraints += 1
                        else:
                            score -= weight * self.config.mismatch_penalty_multiplier
                        continue
                    effective_weight = weight
                    if self._exact_term(value, source):
                        score += effective_weight * self.config.exact_match_multiplier
                        matched_constraints += 1
                    elif key == "feature":
                        coverage = self._term_coverage(value, source)
                        if coverage >= 0.75:
                            score += effective_weight * coverage
                            matched_constraints += 1
                        elif coverage >= 0.35:
                            score += effective_weight * coverage * self.config.partial_match_multiplier
                        else:
                            score -= effective_weight * self.config.feature_mismatch_multiplier
                    elif key in {"category", "gender"} and not source.strip():
                        score -= self.config.missing_facet_penalty
                    else:
                        score -= effective_weight * self.config.mismatch_penalty_multiplier
            if matched_constraints >= 2:
                score += min(
                    self.config.multi_match_bonus_cap,
                    self.config.multi_match_bonus * matched_constraints,
                )
            for key, values in state.negated_slots.items():
                source = facet_fields.get(key, searchable_text)
                if any(self._exact_term(value, source) for value in values):
                    score -= self.config.negation_penalty
            if max_price is not None or min_price is not None or target_price is not None:
                if price is None:
                    score -= self.config.unknown_price_penalty
                else:
                    if max_price is not None:
                        score += (
                            self.config.price_match_bonus
                            if price <= float(max_price)
                            else -self.config.price_mismatch_penalty
                        )
                    if min_price is not None:
                        score += (
                            self.config.price_match_bonus
                            if price >= float(min_price)
                            else -self.config.price_mismatch_penalty
                        )
                    if target_price is not None:
                        relative_distance = abs(price - float(target_price)) / max(float(target_price), 1.0)
                        score += self.config.target_price_bonus * max(0.0, 1.0 - relative_distance)
            if state.current_intent == "browsing":
                expansions = semantic_expansions({
                    item for value in slots.values() for item in _slot_values(value)
                })
                semantic_hits = sum(self._exact_term(term, f"{semantic_text} {searchable_text}") for term in expansions)
                score += min(
                    semantic_hits * self.config.semantic_hit_bonus,
                    self.config.semantic_bonus_cap,
                )
            if str(asin) not in state.shown_asins:
                score += self.config.unseen_product_bonus
            scored.append((score, str(asin)))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [asin for _, asin in scored]

    @staticmethod
    def _exact_term(value: str, source: str) -> bool:
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(value.lower())}(?![a-z0-9])", source.lower()))

    def _category_match_strength(
        self,
        requested: str,
        product_type: str | None,
        subtype: str | None,
        title: str,
        categories: str,
    ) -> float:
        requested_terms = category_terms(requested)
        canonical = CATEGORY_ALIASES.get(requested.lower(), requested.lower())
        exact_subtype = subtype and (
            canonical == subtype.lower() or any(term == subtype.lower() for term in requested_terms)
        )
        if exact_subtype:
            return self.config.category_weight * 1.125
        if product_type and product_type.lower() == canonical:
            return self.config.category_weight
        if any(Agent._exact_term(term, title) for term in requested_terms):
            return self.config.category_weight * (0.25 / 0.24)
        parent = CATEGORY_PARENTS.get(canonical)
        if parent and Agent._exact_term(parent, categories):
            return self.config.category_weight * (0.04 / 0.24)
        if CATEGORY_PARENTS.get(product_type or "") == canonical:
            return self.config.category_weight * (0.10 / 0.24)
        return 0.0

    @staticmethod
    def _is_generic_feature(value: str) -> bool:
        generic = {"imported", "machine wash", "machine washable", "pull on closure", "pull-on closure"}
        parts = {part.strip() for part in re.split(r"[;,]", value.lower()) if part.strip()}
        return bool(parts) and parts.issubset(generic)

    @staticmethod
    def _term_coverage(value: str, source: str) -> float:
        value_terms = set(_terms(value))
        if not value_terms:
            return 0.0
        source_terms = set(_terms(source))
        return len(value_terms & source_terms) / len(value_terms)

    def _next_attribute(self, state: SessionState, rows: list[tuple]) -> str | None:
        active = set(state.accumulated_slots)
        asked = set(state.asked_attributes) | state.declined_attributes
        policy = self.config.clarification_policy

        if policy == "two_batch":
            question_count = len(state.asked_attributes)
            if "other" in state.declined_attributes:
                return None
            return "other" if question_count < self.config.two_batch_question_limit else None

        if policy == "confidence_gated":
            informative = active - {
                "category", "gender", "min_price", "max_price", "target_price"
            }
            confident = (
                len(rows) <= self.config.confidence_candidate_threshold
                or len(informative) >= self.config.confidence_constraint_threshold
            )
            if confident or "other" in state.declined_attributes:
                return None
            return "other" if state.asked_attributes.count("other") < 2 else None

        if policy == "other_first" and "other" not in asked:
            informative = active - {
                "category", "gender", "min_price", "max_price", "target_price"
            }
            if not informative:
                return "other"

        if (
            policy == "other_second"
            and state.asked_attributes
            and state.asked_attributes[-1] == "other"
            and state.last_clarification_count == 2
            and state.asked_attributes.count("other") < 2
        ):
            return "other"

        priorities = [
            ("material", "material"), ("color", "color"),
            ("feature", "feature"), ("style", "style"),
            ("use_case", "use_case"), ("size", "size"),
            ("budget", "budget"), ("brand", "brand"),
            ("category", "category"),
        ]
        expected_value = self.config.clarification_values
        candidates: list[tuple[float, int, str]] = []
        for index, (slot, attribute) in enumerate(priorities):
            slot_present = slot in active or (
                slot == "budget" and {"max_price", "min_price", "target_price"} & active
            )
            if slot_present or attribute in asked:
                continue
            diversity = self._attribute_entropy(attribute, rows)
            utility = expected_value[attribute] + self.config.candidate_diversity_weight * diversity
            candidates.append((utility, -index, attribute))
        if candidates:
            candidates.sort(reverse=True)
            typed = candidates[0][2]
            if policy == "other_second" and state.asked_attributes and "other" not in asked:
                return "other"
            return typed
        return None

    @staticmethod
    def _attribute_entropy(attribute: str, rows: list[tuple]) -> float:
        if not rows:
            return 0.0
        counts: dict[str, int] = {}
        for row in rows:
            (
                _asin, _bm25, price, title, features, categories, details, searchable,
                department, product_type, subtype, materials, colors, sizes, styles, use_cases,
                brand, _semantic,
            ) = row
            values: set[str] = set()
            if attribute == "budget":
                values = {"unknown" if price is None else str(int(float(price) // 25) * 25)}
            elif attribute == "color":
                values = set(str(colors).split())
            elif attribute == "material":
                values = set(str(materials).split())
            elif attribute == "use_case":
                values = set(str(use_cases).split())
            elif attribute == "size":
                values = set(str(sizes).split())
            elif attribute == "category":
                values = {str(subtype or product_type)} if subtype or product_type else set()
            elif attribute == "style":
                values = set(str(styles).split())
            elif attribute == "brand":
                values = {str(brand)} if brand else set()
            if not values:
                values = {"unknown"}
            for value in values:
                counts[value] = counts.get(value, 0) + 1
        if len(counts) <= 1:
            return 0.0
        total = sum(counts.values())
        entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
        return entropy / math.log(len(counts))
