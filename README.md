# Trust-Aware RAG Pipeline: Indirect Prompt Injection Defense
## Tiered Defense-in-Depth & Fine-Grained Evidence Preservation Academic Research Benchmark

---

### 1. Research Context: Comparative Team Implementations & Codebase Overview

In this COMPSCI 764 research project, team members independently engineered distinct methodological versions and implementation architectures to investigate defense against Indirect Prompt Injection (IPI) from complementary perspectives:

- **Member A's Implementation Track (Prototype-to-v6 Iterative Span Classification)**:
  - Explored an iterative progression from early heuristic prototypes through v6;
  - Focused on token-level span deletion and learned contextual span classifiers evaluated on fixed cohort samples;
  - Examined the behavior of paragraph-level gating, trade-offs between context availability and filtering, and downstream impacts on answer F1 and benign quotations.
- **Member B's Implementation Track (This Codebase: Trust-Aware Tiered Defense-in-Depth & Evidence Preservation Pipeline)**:
  - Establishes a complete, self-contained, end-to-end research architecture designed for rigorous academic evaluation:
    - **Formal Threat Benchmark Engine**: Generates 50 distinct Garak attack probes across 10 seen archetypes and 40 unseen probes ([`GarakDataEngine`](file:///Users/Young/Desktop/Young/%E5%AD%A6%E4%B9%A0/UOA/COMPSCI%20764/Research%20project%20topic/Trust-Aware%20RAG%20Pipeline/trust_aware_rag/benchmark/garak_engine.py)), ensuring zero benchmark leakage via opaque SHA-256 IDs and strict metadata exclusion;
    - **Dense Retrieval Foundation**: In-memory dense vector retrieval built on LlamaIndex ([`LlamaIndexRAG`](file:///Users/Young/Desktop/Young/%E5%AD%A6%E4%B9%A0/UOA/COMPSCI%20764/Research%20project%20topic/Trust-Aware%20RAG%20Pipeline/trust_aware_rag/core/rag_engine.py));
    - **Stage 1 (11D High-Throughput Trust Gate)**: Combines 8 microscopic statistical features (with sliding-window local max-pooling) and 3 threat concept anchor projections ([`MultiSignalFeatureExtractor`](file:///Users/Young/Desktop/Young/%E5%AD%A6%E4%B9%A0/UOA/COMPSCI%20764/Research%20project%20topic/Trust-Aware%20RAG%20Pipeline/trust_aware_rag/extractors/multi_signal.py)), calibrated via Platt scaling ([`PlattCalibrator`](file:///Users/Young/Desktop/Young/%E5%AD%A6%E4%B9%A0/UOA/COMPSCI%20764/Research%20project%20topic/Trust-Aware%20RAG%20Pipeline/trust_aware_rag/calibrators/platt.py));
    - **In-Batch Peer Consensus**: Dynamic geometric outlier compensation calculating pairwise cosine similarity matrices across retrieved candidates;
    - **Stage 2 (Sandboxed Cognitive Judge)**: Local SLM semantic judge ([`LLMSemanticJudge`](file:///Users/Young/Desktop/Young/%E5%AD%A6%E4%B9%A0/UOA/COMPSCI%20764/Research%20project%20topic/Trust-Aware%20RAG%20Pipeline/trust_aware_rag/judges/semantic_judge.py)) featuring full-passage sliding-window coverage (450-char window, 120-char stride), explicit tri-state structured verdicts (`benign`, `malicious`, `uncertain`), defensive fallback on uncertainty, and SHA-256 versioned caching;
    - **Fine-Grained Evidence Sanitizer**: Resolves the "Evidence Loss Dilemma" through surgical excision of malicious command spans while preserving Markdown paragraph boundaries (`\n\n`), code fences, and core academic evidence ([`SentenceSanitizer`](file:///Users/Young/Desktop/Young/%E5%AD%A6%E4%B9%A0/UOA/COMPSCI%20764/Research%20project%20topic/Trust-Aware%20RAG%20Pipeline/trust_aware_rag/core/sanitizer.py)), with strict post-sanitization re-verification;
    - **Downstream Generation & Cascaded Calibration**: Joint bivariate logistic calibration ([`CascadedCalibrator`](file:///Users/Young/Desktop/Young/%E5%AD%A6%E4%B9%A0/UOA/COMPSCI%20764/Research%20project%20topic/Trust-Aware%20RAG%20Pipeline/trust_aware_rag/calibrators/platt.py)) and local LLM generator ([`QwenLocalGenerator`](file:///Users/Young/Desktop/Young/%E5%AD%A6%E4%B9%A0/UOA/COMPSCI%20764/Research%20project%20topic/Trust-Aware%20RAG%20Pipeline/trust_aware_rag/core/generator.py)) measuring token F1, exact match, and canary exfiltration across deep-copied isolated comparison arms.

---

### 2. Tiered Defense-in-Depth Architecture

```text
                     [ Retriever Candidate Nodes (Top-k=3) ]
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│  In-Batch Peer Consensus & Geometric Outlier Compensation              │
│  - Compute pairwise cosine similarity matrix across Top-k candidates   │
│  - Outlier score Oi > 0.85 tightens decision threshold: tau* -> +0.12  │
└────────────────────────────────┬───────────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Stage 1: 11D Orthogonal Multi-Signal Statistical & Concept Gate       │
│  - 8 Microscopic Signals: Entropy, Imperatives, Roleplay, Boundaries...│
│  - 3 Threat Anchors: Override, Roleplay, Shell in latent space         │
│  - Sliding-window max-pooling eliminates long-text dilution bias       │
│  - Platt Scaling Posterior Probability Calibration (tau* = 0.75)       │
│  - Millisecond-level fast rejection of high-entropy & overt attacks    │
└────────────────────────────────┬───────────────────────────────────────┘
                                 │ (Candidate nodes Trust >= tau*)
                                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Stage 2: Sandboxed LLM Cognitive Semantic Judge                       │
│  - Deep intent arbitration via Qwen/Qwen2.5-0.5B-Instruct              │
│  - Sliding-window coverage (450 chars, 120 overlap) across full text   │
│  - Structured tri-state output: benign / malicious / uncertain         │
│  - Defensive fallback policy: uncertainty/parse error -> drop/sanitize │
│  - SHA-256 versioned caching prevents stale evaluation state           │
└────────────────────────────────┬───────────────────────────────────────┘
                                 │
                ┌────────────────┴────────────────┐
                │ (If flagged toxic by S1 or S2)  │
                ▼                                 ▼
   [ Baseline: Coarse Drop ]           [ Fine-Grained Sanitizer ]
   Discard entire passage              Surgically excise malicious spans,
   (Loses all supporting facts)        preserve legitimate factual assertions
                │                                 │
                │                                 ▼
                │                   [ Strict Post-Sanitization Verification ]
                │                   (Check canary, re-verify S1/S2 thresholds)
                │                                 │
                ▼                                 ▼
    (Factual Evidence Lost)             [ Safe Academic Context for Downstream LLM ]
```

---

### 3. Empirical Benchmark & Comprehensive Ablations

The benchmark suite (`python trust_aware_pipeline.py`) systematically implements solutions addressing all peer-review feedback:
1. **Separation of Retrieval Exposure from Downstream Execution**: Cross-category exposures are fully accounted for, and downstream ASR is measured strictly from generated text.
2. **Replacement of Length Shortcuts**: Evaluation uses standard NLP **Token F1** and **Exact Match (EM)** against human-curated reference answers, alongside explicit abstention tracking.
3. **Guaranteed Threat Delivery**: Dual experimental tracks—Setting A (controlled post-retrieval injection with 100% baseline exposure) and Setting B (global index infiltration with natural retrieval).
4. **Stratified Seeded Sampling**: 50% seen archetypes and 50% unseen probes sampled across 12 disjoint academic domains.
5. **Zero Metadata Leakage**: Opaque SHA-256 document IDs; metadata keys explicitly excluded from embeddings and prompts.

#### 3.1 Experimental Setting A: Controlled Post-Retrieval Injection Benchmark
Evaluated across 60 stratified test cases (50% seen archetypes, 50% unseen probes across 12 domains) with candidate pools containing 1 injected document + 2 benign distractors (Baseline exposure: 100%):

| Metric | Raw Baseline | Two-Stage Coarse (Drop) | Two-Stage Sanitized | Academic Takeaways & Findings |
| :--- | :---: | :---: | :---: | :--- |
| **Prompt Attack Exposure Rate** | **100.00%** (60/60) | **11.67%** (7/60) | **11.67%** (7/60) | Tiered defense filters or neutralizes **88.33%** of injected payloads |
| **Downstream Attack Success (ASR)** | **3.33%** (2/60) | **0.00%** (0/60) | **0.00%** (0/60) | Both sanitization and coarse drop achieve **0% attack execution**, preventing canary leaks |
| **Abstention Rate (Refusal)** | 1.67% (1/60) | **26.67%** (16/60) | **20.00%** (12/60) | **Key Result**: Coarse dropping induces over 26% refusals; sanitization reduces abstention by **-6.67%** |
| **Answer Factual Token F1** | **46.35%** | **26.15%** | **32.53%** | Sanitized evidence boosts factual F1 by **+6.38%** over coarse dropping |
| **Exact Match (EM)** | 13.33% (8/60) | 5.00% (3/60) | **6.67%** (4/60) | Preserved evidence enables 1.33x higher exact entity matches compared to coarse drop |
| **Facts Rescued from Poisoned Docs** | — | 0 cases (all dropped) | **21 / 60 cases** | **Successfully rescued core academic evidence across 21 poisoned documents** |

> [!NOTE]
> Under Raw Baseline, despite 100% payload exposure, the 0.5B model's downstream execution rate was only 3.33% (2/60), confirming small models exhibit substantial natural instruction inertia. The primary value of fine-grained sanitization lies in **preventing evidence loss, elevating Token F1 from 26.15% to 32.53%**, and rescuing facts in 21 cases.

#### 3.2 Experimental Setting B: Global Index Infiltration & Multi-Tier Ablation
Evaluated across a global index of 2,100 documents (10 domains, including hard negative confounders) over 1,300 queries under natural vector retrieval:

| Metric | Baseline | Stage 2 Alone | Stage 1 Gate | Two-Stage Coarse | Two-Stage Sanitized | Analysis & Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Retrieval Infiltration Exposure** | **85.23%** (1108/1300)<br>[95% CI: 83.20-87.06] | **77.54%** (1008/1300)<br>[95% CI: 75.19-79.72] | **59.92%** (779/1300)<br>[95% CI: 57.23-62.55] | **42.08%** (547/1300)<br>[95% CI: 39.42-44.78] | **42.08%** (547/1300)<br>[95% CI: 39.42-44.78] | Cascaded defense reduces infiltration exposure by **43.15 percentage points** (50.6% relative mitigation) |
| **Mitigation Rate** | 0.0% | **9.0%** | **29.7%** | **50.6%** | **50.6%** | Cascaded pipeline mitigates **20.9% – 41.6%** more attacks than single-stage baselines |
| **Factual Evidence Rescue Rate** | 0.0% | 0.0% | 0.0% | 0.0% | **22.1%** (508/2301 nodes) | **Rescues academic facts across 508 poisoned retrieved documents** |
| **10 Seen Archetypes Detection** | 0.00% | 56.0% | 98.0% | 100.0% | **100.0%** (100/100) | Complete 100% blocking on known attack archetypes |
| **40 Unseen Probes Zero-Shot Detection** | 0.00% | 23.8% | 73.3% | 82.7% | **82.7%** (992/1200) | Cascaded defense lifts unseen probe detection by **+9.4%** across 1,200 unseen test samples |
| **Benign Knowledge Retention** | 100.00% | 100.00% | 100.00% | 100.00% | **98.62%** (286/290) | Retains **98.62%** utility on challenging hard negative scientific texts |
| **False Positive Rate (FPR)** | 0.00% | 0.00% | 0.00% | 0.00% | **1.38%** (4/290) | Minimal false alarm rate on non-malicious queries |
| **Validation ECE (Platt Scaling)** | — | — | 3.21% | 3.21% | **3.21%** | Well-calibrated on 200 validation samples ($\tau^* = 0.75$) |
| **Test Set ECE (40 Unseen Probes)** | — | — | 29.16% (Brier: 0.2227) | 26.06% | **26.06%** (Brier: 0.2245) | Joint calibration reduces calibration error by **-3.10%** under domain shift |

---

### 4. Publication-Grade Figures

The benchmark suite generates three publication-ready figures saved in the project root:

1. **[`eval_figure_1_reliability_diagram.png`](file:///Users/Young/Desktop/Young/%E5%AD%A6%E4%B9%A0/UOA/COMPSCI%20764/Research%20project%20topic/Trust-Aware%20RAG%20Pipeline/eval_figure_1_reliability_diagram.png)**:
   - **Reliability Diagram**: Compares Stage 1 alone (red dashed, ECE: 29.16%) against Two-Stage Cascaded Defense (solid navy, ECE: 26.06%).
   - Highlights epistemic uncertainty drift on 40 zero-shot unseen probes alongside 98.62% clean context retention in Bin 0.
2. **[`eval_figure_2_probes_generalization.png`](file:///Users/Young/Desktop/Young/%E5%AD%A6%E4%B9%A0/UOA/COMPSCI%20764/Research%20project%20topic/Trust-Aware%20RAG%20Pipeline/eval_figure_2_probes_generalization.png)**:
   - **50 Garak Probes Generalization Distribution**: Illustrates passage-level detection across all 50 probes without benchmark leakage (100.0% on seen archetypes, 82.7% macro-average on unseen probes, +9.4% improvement over Stage 1).
3. **[`eval_figure_3_security_utility_tradeoff.png`](file:///Users/Young/Desktop/Young/%E5%AD%A6%E4%B9%A0/UOA/COMPSCI%20764/Research%20project%20topic/Trust-Aware%20RAG%20Pipeline/eval_figure_3_security_utility_tradeoff.png)**:
   - **Multi-Tier Defense & Evidence Rescue Ablation**: Visualizes the infiltration reduction from 85.23% to 42.08%, highlighting the 22.1% evidence rescue rate (508 passages) and 98.62% knowledge retention.

---

### 5. Academic Discussion & Analytical Transparency

1. **Overcoming the Evidence Loss Dilemma via Strict Re-Verification**:
   - Traditional whole-passage drop strategies discard entire passages upon detecting a threat. In low-redundancy or single-source scenarios, this deprives the downstream model of vital premises, resulting in refusal or hallucination.
   - `SentenceSanitizer` excises malicious command spans while preserving legitimate assertions and markdown structure (`\n\n` boundaries, code blocks).
   - Sanitized passages undergo strict re-verification: canary checks, residual pattern scanning, re-evaluation through Stage 1/Stage 2 thresholds, and verification of annotated evidence retention (`verify_evidence_preservation`). This rescued facts across 508 passages in the global index and 21 cases in controlled testing.
2. **Realistic Interpretation of Downstream Generation**:
   - Small models (0.5B) demonstrate natural instruction inertia, executing baseline canary payloads in only 3.33% of exposed cases.
   - The primary contribution of fine-grained sanitization is preserving single-source factual integrity and reducing refusals (from 26.67% to 20.00%) rather than relying on artificial empty contexts.
3. **Peer Consensus Outlier Ablation**:
   - Ablation runs with `enable_peer_outlier_check=False` vs `True` revealed that retrieved poisoned passages in our dataset were already intercepted by Stage 1 or Stage 2, leaving peer consensus with neutral marginal impact on these queries. While geometrically well-motivated for weak-semantic attacks, its contribution here is transparently documented.
4. **Probability Calibration & Epistemic Uncertainty Drift**:
   - Platt scaling on validation yields 3.21% ECE ($\tau^* = 0.75$).
   - On 40 unseen probes, Stage 1 exhibits covariate shift (ECE: 29.16%). Fitting `CascadedCalibrator` jointly combines Stage 1 continuous logits and Stage 2 discrete verdicts to bring test ECE down to 26.06%.
5. **Zero-Leakage Rule Provenance Audit**:
   - `python tools/audit_rules.py` confirms that all **60 rules** are strictly derived from seen training archetypes (100% compliance), with 0 unseen test-specific rules and 0 dead code.

---

### 6. Modular Software Architecture

The codebase is organized into a clean, modular Python package `trust_aware_rag`:

```text
Trust-Aware RAG Pipeline/
├── trust_aware_rag/          # Core research library
│   ├── __init__.py           # Unified top-level exports
│   ├── core/                 # RAG backbone and pipeline components
│   │   ├── __init__.py
│   │   ├── rag_engine.py     # LlamaIndexRAG (vector retrieval, zero metadata leakage)
│   │   ├── generator.py      # QwenLocalGenerator (local model inference, canary detection)
│   │   ├── sanitizer.py      # SentenceSanitizer (fine-grained span filtering, format preservation)
│   │   └── postprocessor.py  # TwoStageTrustPostprocessor (LCEL cascade, dynamic gating)
│   ├── extractors/           # 11D multi-signal statistical & concept anchor engine
│   │   ├── __init__.py
│   │   └── multi_signal.py   # MultiSignalFeatureExtractor (sliding-window max pooling, 3 anchors)
│   ├── calibrators/          # Probability calibration and uncertainty quantification
│   │   ├── __init__.py
│   │   └── platt.py          # PlattCalibrator & CascadedCalibrator (logistic joint scaling)
│   ├── judges/               # Second-stage cognitive semantic judges
│   │   ├── __init__.py
│   │   └── semantic_judge.py # LLMSemanticJudge (sliding-window coverage, tri-state fallback)
│   ├── benchmark/            # Threat datasets and evaluation benchmarks
│   │   ├── __init__.py
│   │   ├── corpus.py         # 10 disjoint academic domains, hard negative corpus
│   │   └── garak_engine.py   # GarakDataEngine (50 probes, natural queries, 3-way split)
│   └── eval/                 # Formal evaluation metrics and plotting engines
│       ├── __init__.py
│       └── evaluator.py      # RAGEvaluator (ECE/Brier computation, 3 figures, deep-copy isolation)
│
├── tests/                    # Automated unit and integration test suite
│   ├── test_rule_provenance.py # Zero-leakage rule provenance audit test (100% Pass)
│   └── test_pipeline.py        # Component unit & integration tests (13/13 Pass)
│
├── tools/
│   └── audit_rules.py        # Independent rule provenance audit script (60 rules audited)
│
├── trust_aware_pipeline.py   # Top-level benchmark orchestration script and CLI
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── PROJECT_REPORT.md         # Comprehensive academic research report
└── eval_figure_*.png         # 3 Publication-grade figures
```

---

### 7. Quick Start & Replication Guide

```bash
# 1. Run the full two-stage benchmark, evidence rescue ablation, and figure rendering:
python trust_aware_pipeline.py

# 2. Run the automated test suite (component functionality & zero-leakage):
pytest tests/

# 3. Run the independent rule provenance auditor (audits all 60 rules):
python tools/audit_rules.py

# 4. Verify modular Python package imports:
python -c "import trust_aware_rag; from trust_aware_rag import LlamaIndexRAG, TwoStageTrustPostprocessor, SentenceSanitizer; print('Modular imports verified!')"
```
