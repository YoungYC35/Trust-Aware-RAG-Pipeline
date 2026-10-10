"""
Two-Stage Trust Postprocessor & LCEL Integration.

Provides LangChain LCEL calibration pipeline and LlamaIndex TwoStageTrustPostprocessor.
"""

from typing import List, Dict, Any, Optional
import copy
import numpy as np

from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableSerializable,
)

from trust_aware_rag.extractors.multi_signal import MultiSignalFeatureExtractor
from trust_aware_rag.calibrators.platt import PlattCalibrator
from trust_aware_rag.judges.semantic_judge import TaskIsolatedSemanticJudge
from trust_aware_rag.core.sanitizer import SentenceSanitizer


def build_lcel_calibrated_chain(
    feature_extractor: MultiSignalFeatureExtractor,
    calibrator: PlattCalibrator,
) -> RunnableSerializable:
    """Constructs LangChain LCEL chain predicting calibrated trust scores."""

    def extract_features_step(inputs: Dict[str, Any]) -> Dict[str, Any]:
        text = inputs.get("text", "")
        features = feature_extractor.extract_feature_vector(text)
        return {"features": features.reshape(1, -1)}

    def calibrate_step(inputs: Dict[str, Any]) -> Dict[str, Any]:
        feats = inputs["features"]
        p_malicious = float(calibrator.predict_calibrated_proba(feats)[0])
        p_trust = 1.0 - p_malicious
        return {
            "p_malicious": p_malicious,
            "trust_score": p_trust,
            "is_threat": p_trust < 0.50,
        }

    chain = (
        RunnablePassthrough()
        | RunnableLambda(extract_features_step)
        | RunnableLambda(calibrate_step)
    )
    return chain


class TwoStageTrustPostprocessor(BaseNodePostprocessor):
    """
    Two-Stage Cascaded RAG Defense Postprocessor:
    Stage 1: Statistical Trust Scorer with Platt Scaling (Tier-1 Gating)
             Fast, drops coarse attacks, preserves high benign utility under hard negative stress tests.
    Stage 2: Task-Isolated Semantic Judge (Tier-2 Intent Verification)
             Eliminates stealthy second-order cognitive/metaprompt attacks (LeakReplay, Poetry, etc.).
    Fine-Grained Sanitizer: Excises localized rogue instructions while rescuing factual knowledge,
             resolving the empirical "Evidence Loss Dilemma" in RAG systems.
    """

    trust_chain: Any
    semantic_judge: TaskIsolatedSemanticJudge
    threshold: float
    sanitizer: Optional[SentenceSanitizer] = None
    enable_sanitization: bool = True
    embedder: Optional[Any] = None
    enable_peer_outlier_check: bool = True

    def __init__(
        self,
        trust_chain: Any,
        semantic_judge: TaskIsolatedSemanticJudge,
        threshold: float = 0.54,
        sanitizer: Optional[SentenceSanitizer] = None,
        enable_sanitization: bool = True,
        embedder: Optional[Any] = None,
        enable_peer_outlier_check: bool = True,
    ):
        # SentenceSanitizer utilizes fast task-isolated boundary rules for sentence-level threat localization
        sanitizer_inst = sanitizer or SentenceSanitizer(semantic_judge=TaskIsolatedSemanticJudge())
        super().__init__(
            trust_chain=trust_chain,
            semantic_judge=semantic_judge,
            threshold=threshold,
            sanitizer=sanitizer_inst,
            enable_sanitization=enable_sanitization,
            embedder=embedder,
            enable_peer_outlier_check=enable_peer_outlier_check,
        )

    @classmethod
    def class_name(cls) -> str:
        return "TwoStageTrustPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        retained = []

        # In-Batch Peer Consensus & Topological Outlier Scoring
        peer_outlier_scores = {}
        if self.enable_peer_outlier_check and self.embedder is not None and len(nodes) >= 2:
            try:
                snippets = [n.node.get_content()[:300] for n in nodes]
                embs = self.embedder.encode(snippets, normalize_embeddings=True)
                sim_mat = np.dot(embs, embs.T)
                for idx_i in range(len(nodes)):
                    peer_sims = [sim_mat[idx_i, idx_j] for idx_j in range(len(nodes)) if idx_j != idx_i]
                    mean_affinity = float(np.mean(peer_sims))
                    outlier_deg = max(0.0, 1.0 - mean_affinity)
                    peer_outlier_scores[idx_i] = (mean_affinity, outlier_deg)
            except Exception:
                pass

        for idx_i, node_score in enumerate(nodes):
            content = node_score.node.get_content()

            # Record In-Batch Peer Affinity and Outlier Degree
            if idx_i in peer_outlier_scores:
                mean_affinity, outlier_deg = peer_outlier_scores[idx_i]
                node_score.node.metadata["peer_affinity"] = mean_affinity
                node_score.node.metadata["outlier_degree"] = outlier_deg
            else:
                outlier_deg = 0.5
                node_score.node.metadata["peer_affinity"] = 0.5
                node_score.node.metadata["outlier_degree"] = 0.5

            # Dynamic outlier-compensated threshold: severe geometric outliers require stricter trust
            effective_threshold = self.threshold
            if outlier_deg > 0.85:
                effective_threshold = min(0.92, self.threshold + 0.12)

            # --- Stage 1: Tier-1 Statistical Platt-Calibrated Trust Gate ---
            stage1_res = self.trust_chain.invoke({"text": content})
            trust_score = stage1_res["trust_score"]
            node_score.node.metadata["stage1_trust"] = trust_score
            node_score.node.metadata["p_malicious"] = stage1_res["p_malicious"]

            # --- Stage 2: Tier-2 Task-Isolated Semantic Judge ---
            stage2_res = self.semantic_judge.evaluate_node(content)
            is_flagged = stage2_res.get("is_flagged", False)
            verdict = stage2_res.get("verdict", "malicious" if is_flagged else "benign")
            node_score.node.metadata["stage2_flagged"] = is_flagged
            node_score.node.metadata["stage2_verdict"] = verdict
            node_score.node.metadata["stage2_intent"] = stage2_res.get("intent_family", "Unknown")

            flagged_reason = None
            if trust_score < effective_threshold:
                flagged_reason = "Stage 1 (Statistical Gate)"
            elif verdict == "malicious":
                flagged_reason = "Stage 2 (Semantic Judge)"
            elif verdict == "uncertain" and trust_score < 0.85:
                # Defensive security fallback: uncertain outputs trigger quarantine if trust is not very high
                flagged_reason = "Stage 2 (Uncertain Arbiter Fallback)"

            # If node is flagged as untrusted:
            if flagged_reason:
                if self.enable_sanitization and self.sanitizer is not None:
                    # Attempt surgical sentence-level threat neutralization
                    sanitized_text, removed_threats, retained_facts = self.sanitizer.sanitize_text(content)
                    supporting_facts = node_score.node.metadata.get("supporting_facts", [])
                    all_facts_retained, fact_ratio = self.sanitizer.verify_evidence_preservation(
                        sanitized_text, supporting_facts
                    )

                    if removed_threats and retained_facts:
                        # Re-verify sanitized facts against the Two-Stage gate
                        new_stage1 = self.trust_chain.invoke({"text": sanitized_text})
                        new_stage2 = self.semantic_judge.evaluate_node(sanitized_text)
                        re_verified_safe = (
                            new_stage1["trust_score"] >= effective_threshold
                            and not new_stage2.get("is_flagged", False)
                            and not (new_stage2.get("verdict") == "uncertain" and new_stage1["trust_score"] < 0.85)
                        )
                        if re_verified_safe:
                            # Threat neutralized; academic factual statements preserved!
                            clean_node = copy.deepcopy(node_score)
                            clean_node.node.set_content(sanitized_text)
                            clean_node.node.metadata["was_sanitized"] = True
                            clean_node.node.metadata["evidence_preserved"] = all_facts_retained
                            clean_node.node.metadata["evidence_retention_ratio"] = fact_ratio
                            clean_node.node.metadata["retained_facts_count"] = len(retained_facts)
                            clean_node.node.metadata["removed_threats_count"] = len(removed_threats)
                            clean_node.node.metadata["drop_stage"] = "None (Sanitized & Facts Rescued)"
                            retained.append(clean_node)
                            continue

                # If sanitization is disabled or failed to rescue clean text, safely drop node
                node_score.node.metadata["drop_stage"] = flagged_reason
                node_score.node.metadata["was_sanitized"] = False
                continue

            # Passed both stages cleanly without needing sanitization
            node_score.node.metadata["drop_stage"] = "None (Retained)"
            node_score.node.metadata["was_sanitized"] = False
            retained.append(node_score)

        return retained
