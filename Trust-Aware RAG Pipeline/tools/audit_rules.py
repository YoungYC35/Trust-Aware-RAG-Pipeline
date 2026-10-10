#!/usr/bin/env python3
"""
Rule Provenance Audit.

Checks every Stage 1 lexicon word / regex and every Stage 2 regex against the attack
payloads. A rule is LEGITIMATE only if it fires on at least one SEEN (training) probe
payload. Rules that fire only on Unseen payloads, or on no payload at all, could only
have been written by looking at the held-out test probes (or are dead code).

Usage:  python tools/audit_rules.py
Exit code is 1 if any non-compliant rule exists.
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trust_aware_rag.benchmark.garak_engine import GarakDataEngine
from trust_aware_rag.extractors.multi_signal import MultiSignalFeatureExtractor
from trust_aware_rag.judges.semantic_judge import (
    TaskIsolatedSemanticJudge,
    LLMSemanticJudge,
)
from trust_aware_rag.core.sanitizer import SentenceSanitizer


def collect_payloads():
    seen, unseen = [], []
    for probe, payloads in GarakDataEngine.GARAK_50_PROBES.items():
        (seen if probe in GarakDataEngine.SEEN_ARCHETYPES else unseen).extend((probe, p) for p in payloads)
    return seen, unseen


def audit():
    seen, unseen = collect_payloads()
    # Audit all rules across Feature Extractor, Base Judge, LLM Hijack Guard, and Sanitizer
    rules = (
        MultiSignalFeatureExtractor.rule_inventory()
        + TaskIsolatedSemanticJudge().rule_inventory()
        + [r for r in LLMSemanticJudge(use_llm=False).rule_inventory() if r[0].startswith("s2.hijack")]
        + SentenceSanitizer.rule_inventory()
    )
    report = {"ok": [], "unseen_only": [], "dead": []}
    for rule_id, pattern, flags in rules:
        s_hits = sorted({pr for pr, p in seen if re.search(pattern, p, flags)})
        u_hits = sorted({pr for pr, p in unseen if re.search(pattern, p, flags)})
        if s_hits:
            report["ok"].append((rule_id, s_hits))
        elif u_hits:
            report["unseen_only"].append((rule_id, u_hits))
        else:
            report["dead"].append((rule_id, []))
    return report


def main():
    report = audit()
    print(f"Compliant rules (fire on >=1 Seen payload): {len(report['ok'])}")
    print(f"\nNON-COMPLIANT - fire ONLY on Unseen payloads ({len(report['unseen_only'])}):")
    for rule_id, hits in report["unseen_only"]:
        print(f"  - {rule_id}   <- {', '.join(hits)}")
    print(f"\nNON-COMPLIANT - fire on NO payload ({len(report['dead'])}):")
    for rule_id, _ in report["dead"]:
        print(f"  - {rule_id}")
    return 1 if (report["unseen_only"] or report["dead"]) else 0


if __name__ == "__main__":
    sys.exit(main())
