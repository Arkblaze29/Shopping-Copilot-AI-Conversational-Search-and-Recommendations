# TikTok TechJam 2026: Shopping Copilot Architecture & Implementation Plan

> **Target Audience:** OpenAI Codex / Cursor / Claude Code / AI Coding Assistants (Plan & Execution Mode)  
> **Repository Scope:** Rebuilding `Agent` (`agent.py`) for high-precision, low-latency, stateful e-commerce search and conversational recommendation.

---

## 1. Problem Statement & System Overview

The objective of the **TikTok TechJam - Shopping Copilot** challenge is to build a next-generation conversational shopping agent that bridges the gap between ambiguous user dialogue and a static product catalog (50,000 Amazon items under `Clothing_Shoes_and_Jewelry`).

### Core Pillars
1. **Core Architecture: Intent Routing & Hybrid Pipeline**
   - **Dual-Track Routing:** Detect "Buying" (high-precision filter lock) vs. "Browsing" (diverse dense retrieval across scenarios).
   - **Hybrid Retrieval Stream:** Combine keyword (BM25), category tree filters, and dense vector similarity $\rightarrow$ LLM Semantic Ranking.
2. **Self-Evolution: Dynamic Context Programming**
   - **Runtime Adaptation:** Distill context across dialogue history, maintaining short-term session state and long-term user preferences.
   - **Adaptive Orchestration:** Dynamic workflow re-orchestration to refine guidance logic turn-by-turn.
3. **Evaluation Metrics**
   - **Coverage (Hit Rate@K):** Catalog recall during retrieval.
   - **Precision (MRR / Top-K Hit Rate):** Accuracy in pushing exact target item to rank #1.
   - **Efficiency (MTTC - Mean Turns to Conversion):** Penalty for unnecessary turns; must convert within a **hard cap of 10 turns**.

---

## 2. Hard Scope & Constraint Matrix

| Category | Requirement / Constraint | Technical Implication |
| :--- | :--- | :--- |
| **Execution Environment** | **In-memory only** | No heavy external vector DBs (Milvus, Qdrant docker). Use `FAISS` in-memory index or NumPy/SciPy matrix alongside `sqlite3` FTS5. |
| **Catalog Access** | **Strictly Read-Only** (50k items) | Precompute embeddings and metadata lookup tables at initialization (`_build_index`). No catalog mutations. |
| **Interaction Limits** | **10 Turns Max** | Zero score if exceeded. Must aggressively narrow search space and minimize dialogue rounds. |
| **Data Format** | Text catalogs, structured metadata, text dialogs | Text-only pipelines (no image/multimodal features required). |
| **Evaluation Determinism**| API Contract Enforcement | Agent must adhere strictly to `respond(session_id, user_message, turn, top_k)` output schema. |

---

## 3. Analysis of Baseline Implementation (`agent.py`)

The provided starter code is a **stateless, keyword-only search engine** with severe limitations:

```python
# BASELINE CODE REFERENCE
from __future__ import annotations
import json
import re
import sqlite3
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
             "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
             "that", "the", "this", "to", "want", "with", "would", "you", "looking"}

def _text(value: object) -> str:
    if value is None: return ""
    if isinstance(value, dict): return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list): return " ".join(str(item) for item in value)
    return str(value)

def _terms(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text) if len(token) > 1 and token.lower() not in STOPWORDS]

class Agent:
    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: set[str] = set()
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                batch.append((
                    str(product["parent_asin"]), _text(product.get("title")),
                    _text(product.get("categories")), _text(product.get("features")),
                    _text(product.get("details")), _text(product.get("store")),
                    _text(product.get("description")),
                ))
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions.add(session_id)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        if session_id not in self._sessions: raise RuntimeError("reset must be called before respond")
        unique_terms = list(dict.fromkeys(_terms(user_message)))[:40]
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        if not expression: recommendations = []
        else:
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
                (expression, top_k),
            ).fetchall()
            recommendations = [{"parent_asin": str(row[0])} for row in rows]
        return {
            "message": "Here are the closest matches I found.",
            "ask_attribute": None,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
```

### Critical Deficiencies in Baseline
1. **Statelessness (Fails MTTC & Context Distillation):** `reset()` stores `session_id` in a `set()`, dropping `user_profile`. `respond()` processes only `user_message` of the current turn, ignoring turn history.
2. **Pure BM25 (Fails Recall & Semantic Matching):** Keyword match fails on synonyms (e.g., "warm winter jacket" won't match "thermal parka" unless exact tokens exist).
3. **No Intent Routing:** Standard `OR` search is run for both high-intent buying queries and broad browsing queries.
4. **No Slot/Filter Extraction:** Fails to parse numerical constraints (e.g., "under $50", "size M", "red").
5. **Passive Dialogue:** Always returns `"Here are the closest matches I found."` without asking clarifying questions to prune search space, maximizing turns needed to convert (destroying MTTC metric).

---

## 4. Architectural Target Blueprint

To achieve top performance, the refactored `Agent` will follow a modular pipeline architecture:

```
                          ┌──────────────────────────┐
                          │  user_message & history  │
                          └─────────────┬────────────┘
                                        │
                         ┌──────────────▼──────────────┐
                         │   1. Context Distiller &    │
                         │      State Store            │
                         └──────────────┬──────────────┘
                                        │
                         ┌──────────────▼──────────────┐
                         │   2. Dual-Track Router &    │
                         │      Slot Extractor         │
                         └──────┬──────────────┬───────┘
                                │              │
                  ┌─────────────▼─┐          ┌─▼──────────────┐
                  │ BUYING TRACK  │          │ BROWSING TRACK │
                  │ Hard Metadata │          │  Dense Vector  │
                  │  Constraint   │          │  Embedding S.  │
                  └─────────────┬─┘          └─┬──────────────┘
                                │              │
                                └──────┬───────┘
                                       │
                         ┌─────────────▼──────────────┐
                         │   3. In-Memory Hybrid      │
                         │      Retrieval (BM25+Dense)│
                         └─────────────┬──────────────┘
                                       │ Top-50 Candidates
                         ┌─────────────▼──────────────┐
                         │   4. LLM Semantic Ranker   │
                         │      & Action Planner      │
                         └─────────────┬──────────────┘
                                       │ Top-K Ranked
                         ┌─────────────▼──────────────┐
                         │  5. Dynamic Response &     │
                         │     Ask Attribute Engine   │
                         └────────────────────────────┘
```

---

## 5. Detailed Component Specifications

### Component 1: Dynamic Session Memory (`SessionStore`)
* **State Representation:**
  ```python
  class SessionState:
      session_id: str
      user_profile: dict
      dialog_history: list[dict]  # [{"role": "user"/"agent", "text": "..."}]
      accumulated_slots: dict     # {"color": "blue", "max_price": 50.0, "category": "dress", "gender": "women"}
      negated_terms: set[str]     # {"cheap", "leather"}
      current_intent: str         # "buying" | "browsing"
  ```
* **Logic:** On each `respond()` invocation:
  1. Record the incoming message.
  2. Parse new explicit/implicit constraints and merge with `accumulated_slots`.
  3. Context decay/override: If user specifies conflicting preferences (e.g., "actually, make it red"), overwrite the slot.

### Component 2: Dual-Track Router & Rule/LLM Slot Extractor
* **Intent Detection:**
  * **Buying Track (High Precision):** Triggered when input contains exact attributes, price limits, specific product types, or purchase signals (e.g., "looking for a red silk blouse under $40 size medium").
  * **Browsing Track (High Recall):** Triggered on open-ended/exploratory queries (e.g., "what should I wear to a beach wedding in Hawaii?").
* **Slot Extractor:**
  * Regex + NLP extraction for: `max_price`, `min_price`, `size`, `color`, `brand`, `gender`, `category`.
  * Convert extracted slots into SQL predicates for hard filtering:
    `WHERE price <= :max_price AND categories LIKE :category`.

### Component 3: In-Memory Hybrid Retrieval Engine
* **Sparse Index:** SQLite FTS5 (BM25) over `title`, `features`, `categories`, `description`.
* **Dense Index:**
  * Pre-calculate embeddings for 50k items during `_build_index` using `sentence-transformers/all-MiniLM-L6-v2` (or fast local ONNX model).
  * Store vectors in an in-memory `faiss.IndexFlatIP` (normalized cosine similarity) or NumPy array matrix for fast vector dot-product.
* **Hybrid Score Fusion (RRF or Linear Score Combination):**
  $$\text{Score}(d) = \alpha \cdot \text{Norm}(\text{Score}_{\text{dense}}) + (1 - \alpha) \cdot \text{Norm}(\text{Score}_{\text{sparse}})$$
  * Apply hard candidate exclusion based on extracted slot metadata prior to vector/BM25 scoring.

### Component 4: LLM Reranker & Active Conversation Engine
* **Reranking:**
  * Extract top 30-50 candidate products from hybrid retrieval.
  * Formulate structured prompt for LLM containing: `user_profile`, `accumulated_slots`, `dialog_history`, and candidate snippets.
  * LLM outputs JSON array of ranked product IDs + optimal single attribute to clarify.
* **Turn Minimization (MTTC Optimization):**
  * If candidates are too broad ($N > 15$), set `ask_attribute` to the most discriminatory attribute (e.g., "sleeve length", "budget", "occasion") to drastically reduce search entropy on the next turn.

---

## 6. Refactoring Roadmap for Codex (Plan Mode Execution)

When instructing Codex / Cursor in **Plan Mode**, follow this modular multi-stage execution plan:

### Phase 1: Infrastructure & Session Memory Foundations
1. Refactor `Agent.__init__` to instantiate a session dictionary `self.sessions: dict[str, SessionState]`.
2. Update `reset(session_id, user_profile)` to initialize `SessionState`.
3. Build a regex/heuristic utility `extract_slots(text)` to capture price, color, size, and gender.

### Phase 2: In-Memory Vector & Metadata Store Integration
1. Extend SQLite schema to include numerical `price` and categorical columns alongside FTS5.
2. Integrate `sentence-transformers` (or `torch`/`onnxruntime` equivalent) into `_build_index`.
3. Compute matrix of embeddings for product titles + key features; store in memory.
4. Implement hybrid scoring function `_hybrid_search(query_text, slots, top_k)`.

### Phase 3: Dual-Track Router & LLM Reranker Integration
1. Implement `classify_intent(user_message, history)` method.
2. Construct structured LLM prompt template for reranking top 30 retrieved items against `SessionState`.
3. Parse LLM JSON output to update top candidates and extract targeted `ask_attribute`.

### Phase 4: Output Formatting & Evaluator Compliance Verification
1. Ensure `respond()` strictly returns:
   ```json
   {
     "message": "...",
     "ask_attribute": "...",
     "recommendations": [{"parent_asin": "..."}, ...],
     "usage": {"prompt_tokens": 0, "completion_tokens": 0}
   }
   ```
2. Verify execution time per turn stays within acceptable limits (< 500ms local).

---

## 7. Directive Prompts for Codex Execution

When asking your AI coding assistant to implement specific modules, pass the prompts below:

### Prompt A: Session Memory & Slot Extraction
> "Act as a Python engineer. Refactor the `Agent` class in `agent.py` to support stateful session tracking. Create a `SessionState` dataclass storing chat history, user profiles, and extracted search slots (max_price, color, size, category). Implement `extract_slots(text: str) -> dict` using robust regex patterns for e-commerce attributes."

### Prompt B: In-Memory Hybrid Vector Search
> "Implement an in-memory hybrid search mechanism inside `Agent`. Combine SQLite FTS5 BM25 search with an in-memory `sentence-transformers` vector index using `all-MiniLM-L6-v2`. Create a method `_retrieved_candidates(query, slots, top_n=50)` that filters items by metadata slots first, then merges sparse and dense normalized scores using weighted linear fusion."

### Prompt C: LLM Reranker & Response Optimization
> "Write the LLM reranking and active clarification module for `Agent.respond`. Send the candidate product list and accumulated session memory to an LLM prompt. The LLM must return a strict JSON schema containing the reordered product list and a single target attribute to ask the user if intent is ambiguous, optimizing Mean Turns to Conversion (MTTC)."

---

*This document serves as the single source of truth for refactoring the TikTok TechJam Shopping Copilot codebase.*
