from __future__ import annotations

import re
from dataclasses import dataclass, field


COLORS = {
    "black", "white", "blue", "red", "pink", "green", "brown", "gray",
    "grey", "purple", "yellow", "orange", "navy", "beige", "gold", "silver",
}
COLOR_ALIASES = {"grey": "gray", "navy": "blue"}
MATERIALS = {
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk",
    "rayon", "fabric", "linen", "denim", "fleece", "satin", "cashmere",
    "synthetic", "elastane", "modal", "rubber", "suede", "canvas",
}
USE_CASES = {
    "hiking", "running", "gym", "winter", "outdoor", "work", "casual",
    "formal", "wedding", "beach", "travel", "sports", "party",
}
STYLES = {
    "slim fit", "relaxed fit", "short sleeve", "long sleeve", "crew neck",
    "v-neck", "casual", "formal", "vintage", "elegant", "lightweight",
    "breathable", "water resistant", "waterproof", "thermal", "insulated",
}
SEMANTIC_EXPANSIONS = {
    "winter": ("thermal", "insulated", "fleece", "warm", "wool"),
    "hiking": ("trail", "outdoor", "waterproof", "durable"),
    "running": ("athletic", "performance", "breathable", "lightweight"),
    "wedding": ("formal", "elegant", "dressy", "occasion"),
    "beach": ("summer", "vacation", "lightweight", "breathable"),
    "work": ("business", "office", "professional", "formal"),
}
GENDER_ALIASES = {
    "women": "women", "woman": "women", "womens": "women", "female": "women",
    "men": "men", "man": "men", "mens": "men", "male": "men", "unisex": "unisex",
}
SIZE_ALIASES = {
    "xs": "XS", "extra small": "XS", "small": "S", "s": "S",
    "medium": "M", "med": "M", "m": "M", "large": "L", "lg": "L", "l": "L",
    "extra large": "XL", "xl": "XL", "xxl": "XXL", "2xl": "XXL",
}
CATEGORY_ALIASES = {
    "dress": "dress", "dresses": "dress", "shirt": "shirt", "shirts": "shirt",
    "t-shirt": "t-shirt", "t-shirts": "t-shirt", "tshirt": "t-shirt", "tshirts": "t-shirt",
    "tee": "t-shirt", "tees": "t-shirt", "tank": "tank top", "tanks": "tank top",
    "tank top": "tank top", "tank tops": "tank top", "top": "top", "tops": "top",
    "blouse": "blouse", "blouses": "blouse", "button-down": "button-down",
    "button-downs": "button-down", "shacket": "shacket", "shackets": "shacket",
    "jacket": "jacket", "jackets": "jacket", "coat": "coat", "coats": "coat",
    "pants": "pants", "jeans": "jeans", "skirt": "skirt", "skirts": "skirt",
    "shoe": "shoe", "shoes": "shoe", "running shoe": "running shoe", "running shoes": "running shoe",
    "trail shoe": "trail shoe", "trail shoes": "trail shoe", "boot": "boot", "boots": "boot",
    "sneaker": "sneaker", "sneakers": "sneaker", "fashion sneaker": "fashion sneaker",
    "fashion sneakers": "fashion sneaker", "loafer": "loafer", "loafers": "loafer",
    "slip-on": "slip-on", "slip-ons": "slip-on",
    "sandal": "sandal", "sandals": "sandal", "sneaker": "sneaker", "sneakers": "sneaker",
    "sweater": "sweater", "sweaters": "sweater", "hoodie": "hoodie", "hoodies": "hoodie",
    "shorts": "shorts", "earring": "earring", "earrings": "earring",
    "necklace": "necklace", "necklaces": "necklace", "bracelet": "bracelet",
    "bracelets": "bracelet", "watch": "watch", "watches": "watch",
    "costume": "costume", "costumes": "costume", "sock": "sock", "socks": "sock",
    "dress sock": "dress sock", "dress socks": "dress sock", "calf sock": "calf sock",
    "calf socks": "calf sock", "wallet": "wallet", "wallets": "wallet",
    "card case": "card case", "card cases": "card case", "card holder": "card holder",
    "card holders": "card holder",
}

CATEGORY_PARENTS = {
    "t-shirt": "tops", "tank top": "tops", "blouse": "tops",
    "button-down": "tops", "shacket": "tops", "top": "tops",
    "running shoe": "shoes", "trail shoe": "shoes", "boot": "shoes", "sneaker": "shoes",
    "fashion sneaker": "shoes", "loafer": "shoes", "slip-on": "shoes",
    "dress sock": "socks", "calf sock": "socks", "sock": "socks",
    "wallet": "wallets", "card case": "wallets", "card holder": "wallets",
}


def _alternation(values: set[str] | dict[str, str]) -> str:
    return "|".join(re.escape(value) for value in sorted(values, key=len, reverse=True))


COLOR_RE = re.compile(rf"\b({_alternation(COLORS)})\b", re.I)
MATERIAL_RE = re.compile(rf"\b({_alternation(MATERIALS)})\b", re.I)
USE_CASE_RE = re.compile(rf"\b({_alternation(USE_CASES)})\b", re.I)
STYLE_RE = re.compile(rf"\b({_alternation(STYLES)})\b", re.I)
CATEGORY_RE = re.compile(rf"\b({_alternation(CATEGORY_ALIASES)})\b", re.I)
GENDER_RE = re.compile(r"\b(women'?s?|female|men'?s?|male|unisex)\b", re.I)
SIZE_RE = re.compile(
    r"(?<!['\w])(?:size\s*)?(extra\s+small|extra\s+large|medium|small|large|xxl|2xl|xl|xs|lg|med|[smlx])\b",
    re.I,
)


@dataclass(frozen=True)
class ProductFacets:
    category_path: tuple[str, ...] = ()
    department: str | None = None
    product_type: str | None = None
    subtype: str | None = None
    materials: frozenset[str] = field(default_factory=frozenset)
    colors: frozenset[str] = field(default_factory=frozenset)
    sizes: frozenset[str] = field(default_factory=frozenset)
    styles: frozenset[str] = field(default_factory=frozenset)
    use_cases: frozenset[str] = field(default_factory=frozenset)
    brand: str | None = None

    def semantic_terms(self) -> tuple[str, ...]:
        values = [self.product_type, self.subtype, self.department, self.brand]
        values.extend(self.materials | self.colors | self.sizes | self.styles | self.use_cases)
        return tuple(value for value in values if value)


def extract_product_facets(product: dict, searchable_text: str) -> ProductFacets:
    categories = tuple(str(value).strip().lower() for value in product.get("categories") or [] if str(value).strip())
    details = product.get("details") if isinstance(product.get("details"), dict) else {}
    normalized_details = {str(key).strip().lower(): str(value).strip() for key, value in details.items()}
    title_text = str(product.get("title") or "")
    feature_value = product.get("features")
    feature_text = " ".join(str(value) for value in feature_value) if isinstance(feature_value, list) else str(feature_value or "")
    details_text = " ".join(f"{key} {value}" for key, value in details.items())
    trusted_text = " ".join((title_text, " ".join(categories), feature_text, details_text)).lower()
    facet_text = searchable_text.lower() or trusted_text

    department_text = normalized_details.get("department", "").lower().replace("'", "")
    department_match = GENDER_RE.search(department_text or f"{' '.join(categories)} {title_text}")
    department = None
    if department_match:
        department = GENDER_ALIASES[department_match.group(1).lower().replace("'", "")]
    category_source = " ".join(categories + (title_text.lower(),))
    category_matches = list(CATEGORY_RE.finditer(category_source))
    product_type = None
    if category_matches:
        best_match = max(category_matches, key=lambda match: (len(match.group(1)), match.start()))
        product_type = CATEGORY_ALIASES[best_match.group(1).lower()]
    subtype = categories[-1].split(",")[-1].strip() if categories else None
    materials = frozenset(match.group(1).lower() for match in MATERIAL_RE.finditer(facet_text))
    colors = frozenset(COLOR_ALIASES.get(match.group(1).lower(), match.group(1).lower()) for match in COLOR_RE.finditer(facet_text))
    sizes = frozenset(
        SIZE_ALIASES[raw]
        for match in SIZE_RE.finditer(" ".join((title_text, feature_text, details_text)))
        if (raw := match.group(1).lower()) in SIZE_ALIASES
    )
    styles = frozenset(match.group(1).lower() for match in STYLE_RE.finditer(facet_text))
    use_cases = frozenset(match.group(1).lower() for match in USE_CASE_RE.finditer(facet_text))
    brand = normalized_details.get("brand") or normalized_details.get("manufacturer") or str(product.get("store") or "").strip()
    return ProductFacets(
        category_path=categories,
        department=department,
        product_type=product_type,
        subtype=subtype,
        materials=materials,
        colors=colors,
        sizes=sizes,
        styles=styles,
        use_cases=use_cases,
        brand=brand.lower() or None,
    )


def semantic_expansions(terms: set[str]) -> set[str]:
    expanded: set[str] = set()
    for term in terms:
        expanded.update(SEMANTIC_EXPANSIONS.get(term, ()))
    return expanded


def category_terms(category: str) -> set[str]:
    """Return canonical, parent, and user-facing aliases for a category."""
    canonical = CATEGORY_ALIASES.get(category.lower(), category.lower())
    terms = {canonical, CATEGORY_PARENTS.get(canonical, "")}
    terms.update(alias for alias, value in CATEGORY_ALIASES.items() if value == canonical)
    return {term for term in terms if term}


def normalize_feature_text(value: str) -> str:
    """Normalize common catalog feature formatting without discarding detail."""
    normalized = re.sub(r"(?i)(?<=\d)\s*%\s*", "% ", value)
    normalized = re.sub(r"(?i)pull\s+on", "pull-on", normalized)
    normalized = re.sub(r"(?i)button\s+down", "button-down", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized
