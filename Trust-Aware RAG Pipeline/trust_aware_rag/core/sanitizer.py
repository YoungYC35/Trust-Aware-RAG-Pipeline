"""
Sentence-Level Fine-Grained Sanitizer & Neutralizer Module.

Directly addresses the "Evidence Loss Dilemma" identified in empirical RAG evaluations:
Coarse passage-level dropping discards valuable supporting facts along with injected directives,
causing downstream LLMs to abstain or fail on knowledge-intensive questions.

SentenceSanitizer performs surgical threat localization and neutralization:
1. Deconstructs untrusted passages into constituent sentence/structural units.
2. Identifies and excises localized rogue directives using audited Stage 1/Stage 2 threat rules.
3. Reassembles benign academic facts into a clean passage and verifies it against the Two-Stage gate.
"""

from typing import Tuple, List, Optional
import re

from trust_aware_rag.judges.semantic_judge import TaskIsolatedSemanticJudge
from trust_aware_rag.extractors.multi_signal import (
    ROLEPLAY_CONSTRUCTIONS,
    EXFIL_RULES,
    IMPERATIVE_VERBS,
    _group_score,
)


# Audited threat tag pattern: strictly restricted to patterns observed in SEEN training payloads
AUDITED_SANITIZER_PATTERNS: List[Tuple[str, str, int]] = [
    (
        "sanitizer.tag:bracket_directive",
        r"<!--|\[[A-Z0-9_\s:]*(?:SYSTEM|OVERRIDE|ADMIN|ROOT|JAILBREAK|ROLEPLAY|DIAGNOSTIC|ENVIRONMENT)[A-Z0-9_\s:]*\]",
        re.IGNORECASE,
    ),
]


class SentenceSanitizer:
    """
    Surgically isolates and excises adversarial instruction fragments from candidate passages,
    preserving factual academic statements to prevent evidence loss in RAG pipelines.
    """

    def __init__(self, semantic_judge: Optional[TaskIsolatedSemanticJudge] = None):
        self.judge = semantic_judge or TaskIsolatedSemanticJudge()
        self.tag_pattern = AUDITED_SANITIZER_PATTERNS[0][1]

    @staticmethod
    def rule_inventory() -> List[Tuple[str, str, int]]:
        """Flat list of (rule_id, pattern, flags) for provenance auditing."""
        return list(AUDITED_SANITIZER_PATTERNS)

    def score_sentence_threat(self, sentence: str) -> float:
        """
        Calculates localized threat score for an individual sentence/line segment.
        Reuses audited Stage 1 and Stage 2 patterns to ensure zero benchmark overfitting.
        """
        s = sentence.strip()
        if len(s) < 5:
            return 0.0

        score = 0.0

        # Stage 2 Intent violations (Override, Role Spoofing, Exfiltration Directives)
        judge_res = self.judge.evaluate_node(s)
        if judge_res["is_flagged"]:
            score += 1.0

        # Stage 1 Roleplay Framing
        if any(re.search(p, s, re.IGNORECASE) for p in ROLEPLAY_CONSTRUCTIONS):
            score += 0.85

        # Stage 1 Exfiltration / Shell syntax (curl, /bin/sh, wget)
        if _group_score(s, EXFIL_RULES) > 0.3:
            score += 0.85

        # Obfuscation delimiters & invisible tokens
        if any(c in s for c in ("\u200b", "\u200c", "\u200d", "\ufeff", "\ufffd")):
            score += 0.90

        # Audited adversarial tags (Seen provenance only)
        if re.search(self.tag_pattern, s, re.IGNORECASE):
            score += 0.75

        # Localized imperative instruction density
        words = re.findall(r"\b[a-z]{2,}\b", s.lower())
        if words:
            imp_count = sum(1 for w in words if w in IMPERATIVE_VERBS)
            if imp_count >= 2:
                score += 0.50

        return score

    @staticmethod
    def verify_evidence_preservation(sanitized_text: str, supporting_facts: List[str]) -> Tuple[bool, float]:
        """
        Verifies whether annotated supporting evidence facts are retained in sanitized text.
        Replaces arbitrary character-length shortcuts with grounded evidence retention testing.
        Returns:
            all_preserved: bool
            retention_ratio: float (0.0 to 1.0)
        """
        if not supporting_facts:
            return True, 1.0
        s_lower = sanitized_text.lower()
        hits = 0
        for fact in supporting_facts:
            fact_clean = fact.strip().lower()
            if not fact_clean:
                continue
            if fact_clean in s_lower:
                hits += 1
            else:
                # Key token coverage fallback for long fact sentences
                tokens = [w for w in re.findall(r"\b[a-z0-9]{4,}\b", fact_clean) if w not in ("with", "that", "from", "this", "they", "have", "were", "been")]
                if tokens and sum(1 for t in tokens if t in s_lower) / len(tokens) >= 0.70:
                    hits += 1
        ratio = hits / max(1, len(supporting_facts))
        return (ratio >= 0.99), ratio

    def sanitize_text(self, text: str, threat_cutoff: float = 0.50) -> Tuple[str, List[str], List[str]]:
        """
        Extracts factual sentences while stripping localized adversarial payloads.
        Preserves original paragraph formatting, markdown headings (subject to inspection),
        and quotation structures.
        Returns:
            sanitized_text: str (clean reassembled facts preserving structure)
            removed_threats: List[str] (excised adversarial sentences/headings)
            retained_facts: List[str] (preserved academic statements)
        """
        paragraphs = text.split("\n\n")
        retained_paragraphs: List[str] = []
        retained_facts: List[str] = []
        removed_threats: List[str] = []

        # Sentence boundary split regex with lookbehind protecting common academic abbreviations
        abbrev_pattern = (
            r"(?<!\bal\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bfig\.)(?<!\bvs\.)"
            r"(?<!\bdr\.)(?<!\bprof\.)(?<=[.!?])\s+"
        )

        for para in paragraphs:
            para_lines = para.split("\n")
            cleaned_lines: List[str] = []

            for line in para_lines:
                raw_line = line.strip()
                if not raw_line:
                    continue

                # 1. Heading Inspection: NEVER exempt headings automatically
                if raw_line.startswith("#"):
                    heading_content = re.sub(r"^#+\s*", "", raw_line)
                    if self.score_sentence_threat(heading_content) >= threat_cutoff:
                        removed_threats.append(raw_line)
                    else:
                        cleaned_lines.append(raw_line)
                        retained_facts.append(raw_line)
                    continue

                # 2. Code fence handling
                if raw_line.startswith("```"):
                    if self.score_sentence_threat(raw_line) >= threat_cutoff:
                        removed_threats.append(raw_line)
                    else:
                        cleaned_lines.append(raw_line)
                    continue

                # 3. Sentence segmentation within paragraph line
                sents = [s.strip() for s in re.split(abbrev_pattern, raw_line, flags=re.IGNORECASE) if s.strip()]
                line_retained: List[str] = []
                for s in sents:
                    if self.score_sentence_threat(s) >= threat_cutoff:
                        removed_threats.append(s)
                    else:
                        line_retained.append(s)
                        retained_facts.append(s)

                if line_retained:
                    cleaned_lines.append(" ".join(line_retained))

            if cleaned_lines:
                retained_paragraphs.append("\n".join(cleaned_lines))

        sanitized_text = "\n\n".join(retained_paragraphs).strip()
        return sanitized_text, removed_threats, retained_facts

