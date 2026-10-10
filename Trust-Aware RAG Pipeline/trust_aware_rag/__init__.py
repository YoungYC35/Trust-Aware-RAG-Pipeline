"""
Trust-Aware RAG Pipeline Package.

A Two-Stage Tiered Defense-in-Depth Framework for Indirect Prompt Injection in RAG Systems:
- Stage 1: MultiSignalFeatureExtractor + PlattCalibrator (Statistical Trust Gate)
- Stage 2: TaskIsolatedSemanticJudge (Cognitive Meta-Instruction Sandbox Judge)
- Core: LlamaIndexRAG, TwoStageTrustPostprocessor, LCEL Integration
- Benchmark: GarakDataEngine (50 Garak Probes Benchmark Suite)
- Eval: RAGEvaluator with formal ECE, Brier Score, and Publication-Grade Plotters
"""

from trust_aware_rag.core.rag_engine import LlamaIndexRAG
from trust_aware_rag.core.generator import QwenLocalGenerator
from trust_aware_rag.core.postprocessor import (
    TwoStageTrustPostprocessor,
    build_lcel_calibrated_chain,
)
from trust_aware_rag.core.sanitizer import SentenceSanitizer
from trust_aware_rag.extractors.multi_signal import MultiSignalFeatureExtractor
from trust_aware_rag.calibrators.platt import PlattCalibrator, CascadedCalibrator
from trust_aware_rag.judges.semantic_judge import (
    TaskIsolatedSemanticJudge,
    LLMSemanticJudge,
)
from trust_aware_rag.benchmark.garak_engine import GarakDataEngine
from trust_aware_rag.eval.evaluator import RAGEvaluator

__all__ = [
    "LlamaIndexRAG",
    "QwenLocalGenerator",
    "TwoStageTrustPostprocessor",
    "build_lcel_calibrated_chain",
    "SentenceSanitizer",
    "MultiSignalFeatureExtractor",
    "PlattCalibrator",
    "CascadedCalibrator",
    "TaskIsolatedSemanticJudge",
    "LLMSemanticJudge",
    "GarakDataEngine",
    "RAGEvaluator",
]
