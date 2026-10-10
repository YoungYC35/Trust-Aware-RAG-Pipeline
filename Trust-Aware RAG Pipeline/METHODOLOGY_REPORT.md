### Evidence-Preserving Cascaded Defence for Retrieval-Augmented Generation

#### III Methodology

**A. Research question and experimental scope**
Retrieval-Augmented Generation (RAG) systems are highly vulnerable to indirect prompt injection when retrieving unverified external context. We ask whether a cascaded, tiered RAG post-processing pipeline can effectively mitigate these attacks while retaining factual utility better than monolithic detection methods. Instead of relying solely on binary whole-passage rejection, our hypothesis is that a Two-Stage Trust Postprocessor—combining a calibrated fast heuristic gate with a meticulous sliding-window Semantic LLM Judge—can optimize the security-utility trade-off.

Our experiments investigate defensive routing, localized span repair, and tri-state ambiguity handling as responses to observed false positive rates and evidence loss. By dynamically cascading the detection models, we aim to intercept malicious instructions before prompt construction while explicitly preserving clean, educational sub-spans within poisoned documents.

**B. Data partitions and threat model**
The evaluation spans two rigorous settings. Setting A utilizes a controlled 60-case injection subset to track precise token-level evidence preservation and answer-level Attack Success Rate (ASR). Setting B expands to a large-scale evaluation on 2,100 documents and 1,300 queries to assess global false positive rates and utility retention. To evaluate zero-shot generalization, we utilize an independent 50-probe standalone dataset, strictly partitioned into 10 seen archetypes and 40 unseen NVIDIA Garak probes. 

We enforce a strict zero-leakage guarantee across training and test sets, mathematically verified by a 60-rule automated auditing framework. The threat model focuses on post-retrieval corruption. We define attack success by the downstream execution of injected instructions within the generated output, rather than merely measuring survival exposure in the retrieved prompt.

**C. Cascaded scoring and evidence-preserving repair**
The defence is implemented as a `TwoStageTrustPostprocessor`. Stage 1 acts as a rapid routing gate, calibrated via a `CascadedCalibrator` utilizing Platt scaling to align predictive trust scores. A context is dynamically routed to the Stage-2 Semantic Judge if it is overtly flagged or if it triggers a defensive uncertainty fallback (defined as a `trust_score < 0.85` or an `uncertain` verdict).

To prevent payload evasion at the boundaries of lengthy documents, Stage 2 abandons fixed-prefix inspection in favor of overlapping sliding token windows (a 450-character window with a 120-character overlap). The LLM judge is constrained to output strict tri-state structured verdicts (`benign`, `malicious`, `uncertain`) with bounded retries and SHA-256 caching. When malicious spans are identified, a downstream `Sanitizer` excises the localized threat while meticulously preserving structural Markdown integrity (e.g., semantic headings and code blocks). Surviving legitimate contexts are deep-copied and re-verified for evidence preservation before generation.

**D. Policy selection and evaluation**
Thresholds are dynamically guided by the cascaded Expected Calibration Error (ECE) to minimize overconfidence on out-of-distribution prompts. Evaluation metrics decouple the four comparison arms (Baseline, Stage 1 Alone, Stage 2 Alone, and Two-Stage Cascaded) using independent node deep copies to prevent state contamination. 

We measure raw infiltration exposure, objective-specific ASR, system abstention, generative F1 score, and False Positive Rate (FPR). Crucially, we introduce a *Factual Evidence Rescue Rate*, which quantifies the proportion of poisoned documents that were successfully sanitized and functionally utilized by the LLM to answer the user's query, rather than being indiscriminately discarded.

#### IV Implementation and Results

**A. Local implementation and cascaded defense outcomes**
The core pipeline is implemented in Python, governed by strict state isolation and validated against a 100% passing test suite. In the Setting A controlled trial (60 cases), baseline exposure was 100% with a clean F1 score of 46.35%. The fine-grained sanitized defence reduced malicious exposure to 11.67% and entirely neutralized the ASR (from 3.33% to 0.00%), successfully rescuing factual evidence in 21 out of 60 heavily poisoned cases while increasing the sanitized F1 score to 32.53% (a +6.38% improvement over coarse filtering).

Table I illustrates the large-scale efficacy on the Setting B cohort (1,300 queries). Raw retrieval infiltration reached 85.23%. While Stage 2 alone mitigated exposure to 77.54% and Stage 1 alone to 59.92%, the combined Two-Stage Cascaded pipeline achieved the lowest exposure at 42.08%, representing a 50.6% relative mitigation. 

Notably, the cascaded system retained 98.62% of benign knowledge utility on hard-negative scientific texts, accompanied by a minimal False Positive Rate (FPR) of 1.38%. The precise span-level sanitization successfully rescued academic facts across 508 poisoned retrieved nodes (a 22.1% rescue rate).

**B. Generalization and calibration under domain shift**
We assessed the out-of-distribution (OOD) robustness of the semantic judge across 50 standalone Garak probes. On the 10 seen archetypes, the cascaded defence achieved 100.0% complete blocking. On the 40 unseen zero-shot probes, the Two-Stage defence generalized to an 82.7% mitigation rate, establishing a substantial +9.4% empirical leap over the Stage 1 baseline.

Furthermore, joint Platt calibration successfully stabilized the system's confidence distributions. The cascaded test-set ECE on unseen probes dropped to 26.06% (reducing calibration error by 3.10% under severe domain shift), demonstrating that the tiered pipeline remains reliably calibrated even when encountering unprecedented injection phrasing.

**C. Analytical Figures and Trade-offs**

1. **Calibration Reliability**
![Reliability Diagram](eval_figure_1_reliability_diagram.png)
*Figure 1: Reliability Diagram demonstrating the effect of joint logistic calibration. It compares the Stage 1 standalone gate (ECE: 29.16%) against the Two-Stage Cascaded Defense (ECE: 26.06%). The cascaded Platt scaling mitigates overconfident predictions under OOD domain drift, aligning empirical accuracy closer to the optimal diagonal.*

2. **Zero-Shot Robustness on Garak Probes**
![Generalization Distribution](eval_figure_2_probes_generalization.png)
*Figure 2: Performance distribution across 50 NVIDIA Garak Probes. The figure highlights passage-level detection robustness across all threat archetypes without synthetic test leakage, demonstrating complete neutralization (100.0%) on seen archetypes and a substantial generalization capability (82.7%) on unseen probes—a significant +9.4% improvement over the standalone Stage 1 baseline.*

3. **Security-Utility Trade-off and Evidence Rescue**
![Security-Utility Trade-off](eval_figure_3_security_utility_tradeoff.png)
*Figure 3: Security-Utility Trade-off showcasing the multi-tier mitigation outcomes. The visualization clearly illustrates how fine-grained sanitization successfully recovers and preserves 508 factual passages (evidence rescue) while simultaneously maintaining a 98.62% benign context retention rate against strict hard negatives.*

#### V Conclusion, Limitations, and Future Work

**A. Conclusion**
This study demonstrates that decoupling threat detection into a calibrated heuristic gate and a sliding-window semantic judge significantly advances the security-utility frontier in RAG systems. Unlike whole-passage rejection models that indiscriminately discard valuable knowledge, the Two-Stage cascaded approach with precise span sanitization rescues over 22% of factual evidence while holding the false positive rate below 1.5%. Furthermore, the implementation of 120-character overlapping sliding windows successfully neutralizes the evasion vulnerabilities inherent in fixed-prefix scanning. These findings validate that defensive routing and tri-state uncertainty fallbacks are critical prerequisites for robust indirect prompt injection mitigation.

**B. Current Limitations**
Despite its efficacy, the current implementation exhibits several architectural and operational flaws that must be addressed before production deployment:
1. **Judge Vulnerability (Meta-Injection):** The Stage-2 Semantic LLM Judge evaluates raw, potentially toxic context. Currently, it remains theoretically susceptible to highly sophisticated recursive meta-injections (where the payload instructs the *judge itself* to classify the text as benign). Although bounded retries and structured formatting mitigate parsing failures, they do not guarantee cryptographic immunity against judge subversion.
2. **Computational Latency Overhead:** The 450-character sliding window with 120-character overlap scales linearly with document length. For exceptionally long retrieved contexts, the sequential nature of these LLM calls introduces significant latency overhead, rendering the current pipeline unsuitable for real-time streaming applications.
3. **Cross-Document Blind Spots:** The Semantic Judge currently processes nodes in isolation. It lacks global cross-document verification, meaning it cannot detect sophisticated semantic poisoning attacks where a malicious instruction is fractured and distributed across multiple retrieved documents.
4. **Adversarial Calibration Decay:** The Platt scaling calibrator is trained on static embeddings. As novel out-of-distribution attacks evolve over time, the heuristic Stage-1 gate may experience confidence drift, forcing more context into the computationally expensive Stage-2 judge and potentially degrading the system to an unacceptably high abstention rate.

**C. Future Research Directions**
To overcome these limitations, future extensions should prioritize the following directions:
- **Judge-Immune Sandboxing:** Deploying the Semantic Judge within a constrained grammatical decoding environment (e.g., restricted CFG generation) to strictly enforce tri-state outputs and block the execution of meta-injection commands.
- **Dynamic Entropy-Based Windowing:** Instead of static 450-token windows, implement a dynamic chunking mechanism driven by the localized predictive entropy of the text. High-entropy (suspicious) regions receive fine-grained inspection, while low-entropy (predictable factual) regions are skipped, drastically reducing latency.
- **Speculative Cascaded Routing:** Exploring dynamic routing of rare ambiguous border cases to larger, high-capacity implicit reasoning models (e.g., 14B+ parameters) via parallel speculative decoding to balance sub-second latency with deep injection reasoning.
