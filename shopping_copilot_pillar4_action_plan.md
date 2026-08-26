# Technical Action Plan: Pillar IV – Evaluation Matrix & Optimization Strategy
## Challenge: Shopping Copilot – AI Conversational Search and Recommendations

---

### Executive Summary
This document details the step-by-step engineering plan to optimize performance against **Pillar IV: Evaluation Matrix (Product & Efficiency Metrics)** in the Shopping Copilot hackathon. The strategy is designed to systematically maximize **Hit Rate@K** (Coverage), **MRR / Top-K Hit Rate** (Precision), and **MTTC** (Efficiency) within the strictly enforced constraints (10 max turns, read-only 50k Amazon dataset, strictly in-memory processing).

---

### Metric Targets & Core Mechanics

| Metric | Target | Focus Area | Engineering Solution |
| :--- | :--- | :--- | :--- |
| **Coverage (Hit Rate@K)** | **> 90%** @ K=50 | Retrieval Recall | Dual-track hybrid search (BM25 + in-memory Dense Embeddings via `Faiss-CPU`/NumPy). |
| **Precision (MRR)** | **> 0.65** | Ranking Accuracy | Context-aware LLM / Cross-Encoder Reranker using dynamic session state & slot context. |
| **Efficiency (MTTC)** | **2.5 – 3.5 Turns** | Dialog Speed | Candidate-density cutoff trigger & proactive slot-filling clarification logic. |

---

### Detailed Implementation Roadmap

#### Step 1: Baseline Benchmarking & Evaluation Pipeline Setup
1. **Repository Setup & Execution:**
   - Initialize the environment with the official participant kit and verified SHA256 catalog checksum.
   - Run the weak BM25 starter agent against the 200 public development sessions.
2. **Automated Evaluation Loop:**
   - Create a local automated test runner around the official evaluator (`evaluator.py`).
   - Log standard benchmark metrics: `Hit Rate@10`, `MRR`, `MTTC`, `Efficiency`, and composite `TechnicalScore`.
3. **Failure Mode Analysis:**
   - Profile the public dev set interactions to classify failure root causes:
     - *Lexical gaps* (e.g., synonyms, broad style descriptions).
     - *Over-generality* (e.g., broad queries like "running shoes").
     - *Slot loss / drift* (e.g., forgetting previous user constraints).

---

#### Step 2: High-Recall Hybrid Retrieval Architecture (Hit Rate@K)
1. **Sparse Retrieval Track (BM25):**
   - Pre-process catalog titles, descriptions, and category hierarchies.
   - Fine-tune BM25 parameters ($k_1$, $b$) for retail search dynamics.
2. **Dense In-Memory Retrieval Track:**
   - Pre-compute sentence embeddings for all 50,000 products in the `Clothing_Shoes_and_Jewelry` catalog using `all-MiniLM-L6-v2` or `bge-small-en-v1.5`.
   - Store vectors in an in-memory `Faiss-CPU` index (`IndexFlatIP` with normalized vectors for cosine similarity).
3. **Hybrid Fusion Layer:**
   - Apply Reciprocal Rank Fusion (RRF) or normalized score weighting to combine Sparse and Dense candidate lists:
     $$	ext{Score}(d) = lpha \cdot 	ext{RRF}_{	ext{Sparse}}(d) + (1 - lpha) \cdot 	ext{RRF}_{	ext{Dense}}(d)$$
   - Dynamically adjust $lpha$: higher weight on BM25 for strict "Buying" intent (exact brands/models); higher weight on Dense for "Browsing" intent (scenario matching).

---

#### Step 3: Dialog Strategy & State Machine (MTTC Efficiency)
1. **Intent Classifier (Buying vs. Browsing):**
   - Classify incoming turn queries using lightweight zero-shot prompt or rule-based slot extraction.
   - *Buying Intent:* Execute hard constraint filtering (e.g., brand, price, color) and narrow retrieval.
   - *Browsing Intent:* Run broad semantic search across cross-category embeddings.
2. **Dynamic State Machine (Slot Filling & Override):**
   - Maintain an in-memory session slot dict:
     ```json
     {
       "category": "Shoes",
       "sub_category": "Running",
       "brand": "Nike",
       "color": "Black",
       "price_max": 120.0,
       "size": "10"
     }
     ```
   - Implement slot rewrite logic to handle user mid-session pivots (e.g., *"Actually, change color to white"*).
3. **Over-Generality Cutoff & Proactive Guidance:**
   - Compute candidate pool entropy/variance after Turn 1 retrieval.
   - **Cutoff Trigger:** If Top-20 candidates span $> 3$ sub-categories, trigger a single proactive clarification question instead of listing random products.
   - *Example Prompt:* *"Are you looking for men's or women's running shoes, and do you have a color preference?"*
   - Hard limit: Allow at most **1 clarification turn** per session to keep MTTC under 3.5 turns.

---

#### Step 4: Semantic Reranking & Context Distillation (MRR Precision)
1. **Personalized Context Distillation:**
   - Condense full conversation history into a concise prompt representation highlighting explicit likes, dislikes, and hard negatives.
2. **LLM Reranking Stage:**
   - Pass Top-30 hybrid candidates into an LLM ranker (or local Cross-Encoder).
   - Require structured output format (JSON array of ASINs with confidence scores).
3. **Re-Ranking Constraints:**
   - Penalize candidate items that violate explicit negative constraints (e.g., user stated *"no high heels"*).

---

#### Step 5: Trade-off Balancing & Final Guardrails
1. **Efficiency vs. Accuracy Tuning:**
   - Grid search parameter thresholds for clarification triggers vs. immediate product recommendations.
2. **Generalization & Anti-Overfitting Check:**
   - Perform 5-fold cross-validation on the 200 public development sessions.
   - Ensure scoring logic relies on general domain rules rather than dataset-specific memorization (preparing for the 800 hidden evaluation sessions).

---

### Deliverables & Submission Checklist

- [ ] **Python Agent Module:** Fully modular agent implementation meeting official interface specifications.
- [ ] **Local Evaluation Results:** Baseline vs. Final `TechnicalScore` improvement summary.
- [ ] **Devpost Project Description:** Comprehensive explanation of architecture, tools, APIs, and datasets.
- [ ] **GitHub Repository:** Well-structured code with installation, reproduction steps, limitations, and team contributions.
- [ ] **Demo Video (YouTube):** End-to-end walkthrough showing agent execution, multi-turn handling, and evaluation output.
