"""Unit tests for the non-retrieval tools (deadline, web search)."""

from __future__ import annotations

from datetime import date

from agentic_rag.tools.deadline import compute_deadline


def test_deadline_prohibitions_in_force():
    r = compute_deadline(
        "When do the AI Act prohibited practices start applying?",
        today=date(2026, 4, 23),
    )
    assert r.phase == "prohibitions"
    assert r.is_in_force is True


def test_deadline_high_risk_future():
    r = compute_deadline(
        "When does Article 6(1) apply to Annex I safety components?",
        today=date(2026, 4, 23),
    )
    assert r.phase == "high_risk_annex1"
    assert r.is_in_force is False
    assert r.days_until > 0


def test_deadline_gpai_phase():
    r = compute_deadline(
        "What's the deadline for general-purpose AI model obligations?",
        today=date(2026, 4, 23),
    )
    assert r.phase == "gpai"


def test_deadline_article_number_fallback():
    r = compute_deadline("Article 9 applies when?", today=date(2026, 4, 23))
    assert r.phase == "high_risk_default"
