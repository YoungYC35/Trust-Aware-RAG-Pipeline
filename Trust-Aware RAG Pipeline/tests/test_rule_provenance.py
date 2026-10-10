"""
Unit test for rule provenance and zero-leakage compliance.

Asserts that every rule in Stage 1 and Stage 2 matches at least one SEEN (training)
probe payload and zero rules match exclusively on UNSEEN (held-out) payloads.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.audit_rules import audit


def test_rule_provenance_zero_data_leakage():
    """Verify that no rules were designed by snooping on held-out test probes."""
    report = audit()
    assert len(report["unseen_only"]) == 0, (
        f"Non-compliant rules found firing ONLY on Unseen probes: {report['unseen_only']}"
    )
    assert len(report["dead"]) == 0, (
        f"Dead rules found matching NO payload: {report['dead']}"
    )
    assert len(report["ok"]) > 0, "Expected compliant rules firing on seen payloads."
