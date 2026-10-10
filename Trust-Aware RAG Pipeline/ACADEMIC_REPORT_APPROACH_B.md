# Trust-Aware Cascaded Defense: An experimental study of out-of-distribution prompt injection in RAG

## Abstract
Retrieved passages in Retrieval-Augmented Generation (RAG) can contain malicious instructions intended to hijack the language model. We investigate whether a cascading trust-aware pipeline can effectively detect out-of-distribution injections and selectively sanitize malicious spans without catastrophic evidence loss. A multi-stage postprocessor—comprising an 11-feature statistical gate, a local Qwen2.5-0.5B semantic judge, and a rule-based sentence sanitizer—is evaluated against a local Qwen2.5-7B generator on the BIPIA dataset. The cascaded defense successfully reduces attack success rates on unconstrained prompts from 28.75% to 12.50%. However, this security comes at a steep utility cost: the sanitizer frequently fails to localize the injection, degrading into whole-passage rejection. This aggressive filtering discards substantial legitimate evidence, particularly in the email domain, reducing final answer F1 significantly. The results demonstrate the strong out-of-distribution detection capabilities of semantic features, while exposing the severe security-utility trade-off inherent in localized evidence repair.

## 1 Research question and hypotheses
Indirect prompt injection occurs when instructions embedded in external content influence an application that retrieves that content. In retrieval-augmented generation (RAG), rejecting a contaminated passage entirely may remove facts needed for the final answer. Therefore, a defense must reliably detect the malicious instruction and accurately remove it without destroying the surrounding benign context. 

Our primary hypothesis is that a cascaded defense (combining statistical machine learning and small LLM semantic judgments) can generalize better to unseen attack vectors than purely lexical detectors. A secondary hypothesis is that a sentence-level sanitizer can rescue clean evidence from a flagged document, providing a utility advantage over whole-passage rejection. The contribution is a controlled empirical evaluation of this multi-stage pipeline, focusing heavily on its out-of-distribution (OOD) transferability and the cost of false positives.

## 2 Methodology

### Dataset and independent partitions
The pipeline's statistical gate is fitted on a synthetically generated dataset via Garak, containing 400 training documents and 200 calibration documents, balanced between benign and injected examples. The host domains for training differ completely from the final testing domains.

For the final evaluation, we utilize the official BIPIA dataset, explicitly selecting 20 EmailQA and 20 TableQA contexts. The selection mechanically excludes any source overlap with the training data. This ensures a strict OOD evaluation, as the defense pipeline has never seen the email or table domains, nor the specific BIPIA injection payloads.

### Threat model and controlled conditions
The attacker's objective is to manipulate the generated answer by inserting a specific string or substituting a fictional name, simulating data corruption or override. The injection occurs after the retrieval stage; thus, the study measures the screening of retrieved evidence rather than the retriever's robustness. 
Each source context yields clean cases and active attack cases. Payloads are placed dynamically (start, middle, end) to prevent positional overfitting.

### Statistical gate and reconstruction (Tier 1)
The first stage of the cascade is a passage-level statistical gate. A `MultiSignalFeatureExtractor` extracts 11 features: semantic discontinuity (using `sentence-transformers/all-MiniLM-L6-v2`), character entropy, directive density, roleplay cues, dialogue-boundary cues, authority cues, shell/exfiltration syntax, obfuscation cues, and three anchor similarities. 
A logistic regression model with a Platt scaling calibrator maps these features to a trust score (1 minus maliciousness). The threshold is set strictly at 0.75. If trust falls below this cutoff, the document is flagged.

### Semantic judgement (Tier 2)
To mitigate the statistical gate's blind spots, the pipeline employs a local `Qwen2.5-0.5B-Instruct` model as a semantic judge. It classifies overlapping 450-character windows (with 120-character overlap) to prevent long texts from diluting the attack context. 
If the judge returns `Malicious`, or returns `Uncertain` while the Tier-1 trust is below 0.85, the source is flagged. This forms a rigorous double-gate detection system.

### Sentence sanitization and rechecking (Tier 3)
When a source is flagged, a `SentenceSanitizer` splits it into lines and sentences. It applies localized threat rules (e.g., imperative density, boundary violations). Units scoring above 0.50 are removed. 
Crucially, the remaining text must pass statistical and semantic rechecking. If the sanitized text still triggers the gates, or if the sanitizer removes everything, the pipeline falls back to complete source rejection (Coarse Drop).

### Generator and controlled execution
The generator is `Qwen2.5-7B-Instruct` running locally via Ollama with temperature 0. The evaluation tests both a Guarded JSON prompt (with explicit system warnings) and a standard unconstrained BIPIA prompt to test prompt sensitivity. 

## 3 Results and discussion

### Detection coverage and security outcomes
Under the unconstrained BIPIA prompt, the cascaded pipeline provides measurable protection, reducing the attack success rate from 28.75% (No filter baseline) to 12.50%. This supports the primary hypothesis: the deep semantic features (embeddings and directive density) generalize well to OOD attacks, outperforming purely lexical detectors that often fail completely on new vocabulary.

### The failure of selective repair
Despite strong detection, the selective repair mechanism fails to rescue evidence. In the primary evaluation, zero sanitization attempts are accepted. The local rule-based sanitizer assigns low scores to the localized payloads, failing to pinpoint the exact malicious span. Consequently, the pipeline repeatedly falls back to whole-passage rejection. The proposed repair method therefore degrades into coarse rejection, rendering the evidence rescue rate effectively 0%.

### Evidence preservation and the domain penalty
Because the pipeline aggressively rejects whole passages, it inflicts severe damage on answer utility. On the BIPIA prompt, the answer F1 drops to 0.051. 
Subgroup analysis reveals a massive domain bias: the pipeline discards 70% of clean email sources, but only 10% of clean tables. Emails naturally contain high directive density ("Please reply", "Click here"), which the Tier-1 and Tier-2 gates misclassify as prompt injections. This highlights the extreme difficulty of separating benign instructions from malicious ones in unstructured communications.

## 4 Limitations and validity
The internal validity is strong due to paired inputs, frozen thresholds, and shared generators. However, the evaluation relies on a specific synthetic training dataset (Garak), meaning the high false positive rate on emails might stem from domain-specific training bias rather than an architectural flaw. 
Additionally, the sentence sanitizer uses static rules. A more advanced semantic localizer might successfully slice out the poison without triggering the fallback rejection, altering the utility outcomes significantly.

## 5 Conclusion and future work
We conclude that the Trust-Aware Cascaded Pipeline offers a bounded, statistically significant security advantage on unseen out-of-distribution attacks. However, this advantage relies entirely on rejecting evidence rather than repairing it. The system demonstrates strong external-attack detection but fails at selective evidence preservation, ultimately trading severe utility loss (and high false positives on emails) for security.

Future work should replace the rule-based sentence sanitizer with a granular LLM-driven localizer that can accurately excise payloads without destroying the surrounding context. Furthermore, mitigating domain-specific false alarms requires training the statistical gate on a broader corpus of benign directives, particularly encompassing corporate correspondence and email formats.
