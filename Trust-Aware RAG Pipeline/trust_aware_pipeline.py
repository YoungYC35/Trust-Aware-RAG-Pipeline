#!/usr/bin/env python3
"""
Trust-Aware RAG Pipeline: Indirect Prompt Injection Defense.

Academic Stress-Test Suite: 50 Distinct Garak Probes Comprehensive Benchmark
with Two-Stage Tiered Defense-in-Depth:
- Stage 1: MultiSignalFeatureExtractor + PlattCalibrator (Statistical Trust Gate)
- Stage 2: TaskIsolatedSemanticJudge (Cognitive Meta-Instruction Sandbox Judge)
- Integration: LlamaIndex BaseNodePostprocessor & LangChain LCEL Calibrated Chains

This module serves as the primary executable entry point and top-level orchestrator.
Core functionality is modularly organized in the `trust_aware_rag` package:
- `trust_aware_rag.core`: RAG retrieval index, TwoStageTrustPostprocessor, and LCEL chain
- `trust_aware_rag.extractors`: 11-Signal feature extraction with sliding-window pooling and threat anchors
- `trust_aware_rag.calibrators`: Platt Scaling calibration and ECE calculation
- `trust_aware_rag.judges`: Sandboxed Task-Isolated Semantic Judge
- `trust_aware_rag.benchmark`: 50 Garak probes dataset engine and 3-way partitioner
- `trust_aware_rag.eval`: Comprehensive evaluator and publication-grade plotters
"""

import os
import logging
import numpy as np

# Suppress external verbose logging and HF Hub notices before loading libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.basicConfig(level=logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Modular imports from trust_aware_rag package
from trust_aware_rag.core.rag_engine import LlamaIndexRAG  # noqa: E402
from trust_aware_rag.core.generator import QwenLocalGenerator  # noqa: E402
from trust_aware_rag.core.postprocessor import (  # noqa: E402
    TwoStageTrustPostprocessor,
    build_lcel_calibrated_chain,
)
from trust_aware_rag.extractors.multi_signal import MultiSignalFeatureExtractor  # noqa: E402
from trust_aware_rag.calibrators.platt import PlattCalibrator  # noqa: E402
from trust_aware_rag.judges.semantic_judge import (  # noqa: E402
    TaskIsolatedSemanticJudge,
    LLMSemanticJudge,
)
from trust_aware_rag.benchmark.garak_engine import GarakDataEngine  # noqa: E402
from trust_aware_rag.eval.evaluator import RAGEvaluator  # noqa: E402

__all__ = [
    "LlamaIndexRAG",
    "QwenLocalGenerator",
    "TwoStageTrustPostprocessor",
    "build_lcel_calibrated_chain",
    "MultiSignalFeatureExtractor",
    "PlattCalibrator",
    "TaskIsolatedSemanticJudge",
    "LLMSemanticJudge",
    "GarakDataEngine",
    "RAGEvaluator",
    "main",
]


def optimize_threshold_on_validation(
    calibrator: PlattCalibrator,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """
    Selects optimal decision threshold tau* on Validation Set using an explicit Security-Utility Objective:
    Objective = TPR - 1.5 * FPR (explicitly balancing detection with false positive minimization to preserve evidence and utility).
    Prevents test-set snooping by ensuring tau* is calibrated before touching the test split.
    """
    print("[Module 3] Optimizing decision threshold tau* on Validation Set (Security-Utility Objective)...")
    val_probs = calibrator.predict_calibrated_proba(X_val)
    optimal_thresholds = []
    best_obj = -999.0

    for th in np.linspace(0.20, 0.80, 61):
        preds_malicious = (val_probs >= (1.0 - th)).astype(int)
        tp = np.sum((preds_malicious == 1) & (y_val == 1))
        fn = np.sum((preds_malicious == 0) & (y_val == 1))
        tn = np.sum((preds_malicious == 0) & (y_val == 0))
        fp = np.sum((preds_malicious == 1) & (y_val == 0))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        obj = tpr - 1.5 * fpr
        if obj > best_obj:
            best_obj = obj
            optimal_thresholds = [th]
        elif np.isclose(obj, best_obj):
            optimal_thresholds.append(th)

    calibrated_threshold = round(float(np.mean(optimal_thresholds)), 2)
    print(f"  * Calibrated Security Threshold tau*: {calibrated_threshold:.2f} (Security-Utility Obj: {best_obj:.3f})")
    return calibrated_threshold


def main():
    import torch
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 84)
    print("  TRUST-AWARE RAG PIPELINE: TWO-STAGE TIERED DEFENSE BENCHMARK (50 PROBES)")
    print("  Formal Academic Experiment Suite with Platt Scaling & Semantic Judge")
    print("=" * 84)

    # 1. Module 2: Strict 3-way partition across 50 Garak probes
    train_data, val_data, test_data = GarakDataEngine.generate_partitioned_dataset(seed=42)

    # 2. Module 1: Build in-memory LlamaIndex VectorStoreIndex and initialize Local LLM Generator
    all_corpus = train_data + val_data + test_data
    generator = QwenLocalGenerator()
    rag = LlamaIndexRAG(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        generator=generator,
    )
    rag.build_index(all_corpus)

    # 3. Module 3: 11-Signal Feature Extractor & Platt Scaling Calibration (Stage 1)
    print("\n[Module 3] Initializing Stage 1 Feature Extractor (11 Signals: 8 Statistical + 3 Threat Concept Anchors)...")
    feature_extractor = MultiSignalFeatureExtractor(embedder=rag.sentence_transformer)

    print("[Module 3] Extracting features for Train and Validation Sets...")
    X_train = np.array([feature_extractor.extract_feature_vector(d["text"]) for d in train_data])
    y_train = np.array([1 if d["is_injected"] else 0 for d in train_data])

    X_val = np.array([feature_extractor.extract_feature_vector(d["text"]) for d in val_data])
    y_val = np.array([1 if d["is_injected"] else 0 for d in val_data])

    # Fit base classifier on Train Set
    calibrator = PlattCalibrator()
    calibrator.fit_base(X_train, y_train)

    # Fit Platt Scaling strictly on Validation Set (Seen Archetypes only)
    print("[Module 3] Fitting Platt Scaling Calibration on Validation Set...")
    calibrator.fit_platt_scaling(X_val, y_val)

    # Compute Validation ECE
    val_probs = calibrator.predict_calibrated_proba(X_val)
    val_ece = PlattCalibrator.compute_ece(y_val, val_probs)
    print(f"  * Validation ECE after Platt Scaling: {val_ece * 100:.2f}%")

    # Select optimal decision threshold solely on Validation Set via Maximum-Margin Youden's J
    calibrated_threshold = optimize_threshold_on_validation(calibrator, X_val, y_val)

    # Build LangChain LCEL Calibrated Chain
    trust_lcel_chain = build_lcel_calibrated_chain(feature_extractor, calibrator)

    # Initialize Stage 2 LLM Semantic Judge (Qwen/Qwen2.5-0.5B-Instruct + Anti-Hijack Guard)
    print("\n[Module 3.5] Initializing Stage 2 LLM Semantic Judge (Qwen/Qwen2.5-0.5B-Instruct)...")
    semantic_judge = LLMSemanticJudge(
        model=generator.model,
        tokenizer=generator.tokenizer,
        device=generator.device,
    )

    # Fit Cascaded Joint Calibrator strictly on Validation Set (Issue 9)
    print("[Module 3.5] Learning joint calibration model combining Stage 1 and Stage 2 on Validation Set...")
    from trust_aware_rag.calibrators.platt import CascadedCalibrator
    val_s1_logits = calibrator.base_classifier.decision_function(X_val)
    val_s2_flags = np.array([
        1.0 if semantic_judge.evaluate_node(d["text"]).get("is_flagged", False) else 0.0
        for d in val_data
    ])
    cascaded_calibrator = CascadedCalibrator()
    cascaded_calibrator.fit(val_s1_logits, val_s2_flags, y_val)

    # Insert into TwoStageTrustPostprocessor with In-Batch Peer Consensus & Geometric Outlier Check
    two_stage_postprocessor = TwoStageTrustPostprocessor(
        trust_chain=trust_lcel_chain,
        semantic_judge=semantic_judge,
        threshold=calibrated_threshold,
        embedder=rag.sentence_transformer,
        enable_peer_outlier_check=True,
    )
    print("[Module 3.5] TwoStageTrustPostprocessor armed with Stage 1 Platt Gate + Stage 2 LLM Judge + In-Batch Peer Consensus.")

    # 4. Module 4: Run Evaluation on Strictly Held-Out Test Set (50 Probes, 40 Zero-Shot Unseen)
    evaluator = RAGEvaluator(
        rag_engine=rag,
        postprocessor=two_stage_postprocessor,
        test_dataset=test_data,
        cascaded_calibrator=cascaded_calibrator,
    )
    results = evaluator.run_evaluation()

    # Compute Test ECE for Stage 1 alone and Two-Stage Cascaded
    y_test_true = np.array(results["y_true"])
    y_test_prob_s1 = np.array(results["y_prob_stage1"])
    y_test_prob_casc = np.array(results["y_prob_cascaded"])
    test_ece_s1 = PlattCalibrator.compute_ece(y_test_true, y_test_prob_s1)
    test_ece_casc = PlattCalibrator.compute_ece(y_test_true, y_test_prob_casc)

    metrics = evaluator.print_academic_report(
        results=results,
        val_ece=val_ece,
        test_ece_s1=test_ece_s1,
        test_ece_cascaded=test_ece_casc,
        calibrated_threshold=calibrated_threshold,
    )

    # Render 3 SEPARATE publication figures
    print("\n[Module 4] Rendering 3 Separate Publication Figures...")
    evaluator.plot_figure_1_reliability_diagram(results, metrics, output_path="eval_figure_1_reliability_diagram.png")
    evaluator.plot_figure_2_probes_generalization(metrics, output_path="eval_figure_2_probes_generalization.png")
    evaluator.plot_figure_3_security_utility_tradeoff(metrics, output_path="eval_figure_3_security_utility_tradeoff.png")

    # 5. Module 5: Downstream Answer-Level Generation Benchmark (Qwen2.5-0.5B-Instruct)
    print("\n[Module 5] Evaluating Downstream LLM Generation with Qwen2.5-0.5B-Instruct...")
    evaluator.evaluate_downstream_generation(sample_limit=60)

    print("\nTwo-stage academic verification & end-to-end LLM generation experiment completed successfully!")


if __name__ == "__main__":
    main()
