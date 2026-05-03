"""AI Act deadline calculator — non-retrieval tool.

The AI Act (Regulation 2024/1689) applies in staged phases. This tool maps a
question mentioning an article / chapter to the applicable date and returns
days/months until (or since) that date.

Reference: Article 113 of Regulation (EU) 2024/1689 — entry into force.
"""

from __future__ import annotations

import re
from datetime import date
from dataclasses import dataclass

# Staged application per Article 113.
# Keys are buckets; values are the application date.
_PHASE_DATES: dict[str, date] = {
    "prohibitions": date(2025, 2, 2),     # Chapter I + II (prohibited practices, incl. Article 5)
    "gpai": date(2025, 8, 2),             # Chapter V (general-purpose AI models)
    "high_risk_default": date(2026, 8, 2),  # General application date
    "high_risk_annex1": date(2027, 8, 2),   # Article 6(1) — Annex I safety components
}

# Keyword routing to phase.
_PHASE_KEYWORDS: list[tuple[str, list[str]]] = [
    ("prohibitions", ["prohibited", "article 5", "chapter i", "chapter ii", "social scoring",
                      "biometric identification", "subliminal"]),
    ("gpai", ["general-purpose", "gpai", "foundation model", "article 51", "article 55",
              "systemic risk"]),
    ("high_risk_annex1", ["annex i", "safety component", "article 6(1)"]),
    ("high_risk_default", ["high-risk", "high risk", "conformity assessment",
                           "risk management", "article 9", "article 10"]),
]


@dataclass
class DeadlineResult:
    """Output payload of :py:func:`compute_deadline`.

    Attributes:
        phase: One of the :data:`_PHASE_DATES` bucket keys.
        applicable_from: Date on which the phase starts applying under
            Article 113 of the AI Act.
        today: Reference date used for the comparison (defaults to
            :py:meth:`date.today`).
        days_until: Signed days from ``today`` to ``applicable_from``;
            negative when the phase is already in force.
        months_until: ``days_until / 30.44`` for human-friendly summaries.
        is_in_force: ``True`` iff ``today >= applicable_from``.
        explanation: Pre-rendered English summary suitable for direct
            display in a chat reply.
    """

    phase: str
    applicable_from: date
    today: date
    days_until: int
    months_until: float
    is_in_force: bool
    explanation: str

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict (dates as ISO strings)."""
        return {
            "phase": self.phase,
            "applicable_from": self.applicable_from.isoformat(),
            "today": self.today.isoformat(),
            "days_until": self.days_until,
            "months_until": round(self.months_until, 1),
            "is_in_force": self.is_in_force,
            "explanation": self.explanation,
        }


def _classify(question: str) -> str:
    """Map a free-text question to one of the AI Act phase buckets.

    Tries phrase-level keyword matches first; falls back to a direct
    ``Article N`` lookup when no keyword fires. The default bucket is
    ``high_risk_default`` because the bulk of the AI Act's obligations land
    on that date and an unknown question is most likely a high-risk one.
    """
    q = question.lower()
    for phase, kws in _PHASE_KEYWORDS:
        if any(k in q for k in kws):
            return phase
    # Direct "Article N" mention without other keywords.
    m = re.search(r"article\s+(\d+)", q)
    if m:
        n = int(m.group(1))
        if n == 5:
            return "prohibitions"
        if n in {51, 55, 53}:
            return "gpai"
        if n == 6:
            return "high_risk_annex1"
        if n in {9, 10, 11, 12, 13, 14, 15}:
            return "high_risk_default"
    return "high_risk_default"


def compute_deadline(question: str, today: date | None = None) -> DeadlineResult:
    """Compute the AI Act application date that best fits ``question``.

    Args:
        question: Natural-language question, e.g. *"When do the AI Act
            prohibited practices apply?"*.
        today: Optional override for the reference date. Useful for unit
            tests that need stable output regardless of clock drift.

    Returns:
        A :class:`DeadlineResult` with both raw fields (days, months,
        in-force flag) and a pre-rendered ``explanation`` string suitable
        for direct surfacing in chat replies.
    """
    today = today or date.today()
    phase = _classify(question)
    applicable = _PHASE_DATES[phase]
    days = (applicable - today).days
    months = days / 30.44
    in_force = days <= 0
    friendly = {
        "prohibitions": "prohibited AI practices (Chapter I & II)",
        "gpai": "general-purpose AI model obligations (Chapter V)",
        "high_risk_default": "high-risk AI systems — general application",
        "high_risk_annex1": "Article 6(1) high-risk systems with Annex I safety components",
    }[phase]
    if in_force:
        explanation = (
            f"The {friendly} have applied since {applicable.isoformat()} "
            f"({abs(days)} days ago)."
        )
    else:
        explanation = (
            f"The {friendly} apply from {applicable.isoformat()} — "
            f"{days} days ({months:.1f} months) from today ({today.isoformat()})."
        )
    return DeadlineResult(
        phase=phase,
        applicable_from=applicable,
        today=today,
        days_until=days,
        months_until=months,
        is_in_force=in_force,
        explanation=explanation,
    )
