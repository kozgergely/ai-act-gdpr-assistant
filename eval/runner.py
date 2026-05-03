"""Evaluation runner for the agentic RAG chatbot.

Three scoring layers (per D12 in DEVLOG.md):
  1. retrieval_recall  — determine if the expected chunk ids appear in the
                         retrieved documents (router/retrieval quality)
  2. citation_precision — all citation ids in the final answer must trace
                          back to retrieved chunks (no hallucinated cites)
  3. answer_quality    — LLM-as-judge on a 1–5 rubric (factuality,
                         groundedness, completeness); skippable if the LLM
                         is Dummy

Outputs a markdown report to eval/reports/report_<timestamp>.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import yaml

# Allow running without installing the package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag.graph.main import build_agent_graph  # noqa: E402
from agentic_rag.llm.base import LLM, DummyLLM, Message, make_llm  # noqa: E402


@dataclass
class QuestionResult:
    id: str
    category: str
    query: str
    expected_citations: list[str]
    expected_intent: str
    actual_intent: str
    retrieved_ids: list[str]
    citation_ids: list[str]
    final_answer: str
    latency_ms: float
    retrieval_recall: float  # 0..1
    citation_precision: float  # 0..1
    answer_quality: float  # 0..5 (NaN if skipped)
    intent_correct: bool
    judge_explanation: str = ""


@dataclass
class ReportSummary:
    n: int
    mean_recall: float
    mean_citation_precision: float
    mean_answer_quality: float
    intent_accuracy: float
    per_category: dict[str, dict[str, float]] = field(default_factory=dict)
    mean_latency_ms: float = 0.0


# --- layer 1: retrieval recall ---------------------------------------------

def _recall(expected: list[str], retrieved: list[str]) -> float:
    if not expected:
        return 1.0
    hits = sum(1 for e in expected if e in retrieved)
    return hits / len(expected)


# --- layer 2: citation precision (no-hallucination) -------------------------

def _citation_precision(citation_ids: list[str], retrieved_ids: list[str]) -> float:
    """Fraction of cited chunks that appear in the retrieved set."""
    if not citation_ids:
        return 1.0
    retrieved_set = set(retrieved_ids)
    hits = sum(1 for c in citation_ids if c in retrieved_set)
    return hits / len(citation_ids)


# --- layer 3: LLM-as-judge --------------------------------------------------

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "factuality": {"type": "integer", "minimum": 1, "maximum": 5},
        "groundedness": {"type": "integer", "minimum": 1, "maximum": 5},
        "completeness": {"type": "integer", "minimum": 1, "maximum": 5},
        "explanation": {"type": "string"},
    },
}


def _judge(
    llm: LLM, q: dict, draft: str, context_hint: str, *, query: str
) -> tuple[float, str]:
    if isinstance(llm, DummyLLM):
        return (float("nan"), "skipped (dummy judge)")
    prompt = (
        "You grade an AI assistant's answer against a gold-standard reference, "
        "on three 1–5 integer scales:\n"
        "  * factuality   (does it contradict the reference?)\n"
        "  * groundedness (are claims supported by the retrieved context?)\n"
        "  * completeness (does it cover the reference's key points?)\n\n"
        f"Question:\n{query}\n\n"
        f"Reference answer:\n{q['reference'].strip()}\n\n"
        f"Retrieved context (brief):\n{context_hint[:1500]}\n\n"
        f"Assistant's draft answer:\n{draft}\n"
    )
    try:
        out = llm.generate_json(prompt, JUDGE_SCHEMA)
        scores = [out.get("factuality", 3), out.get("groundedness", 3), out.get("completeness", 3)]
        return (mean(scores), out.get("explanation", ""))
    except Exception as e:
        return (float("nan"), f"judge failed: {e}")


# --- runner -----------------------------------------------------------------

def _load_questions(path: Path) -> list[dict]:
    return yaml.safe_load(path.read_text())


def _citation_to_id(c: dict) -> str:
    return f"{c['regulation']}:{c['kind']}:{str(c['number']).split('.')[0]}"


def run_eval(
    questions_path: Path,
    *,
    use_judge: bool = True,
    limit: int | None = None,
) -> tuple[list[QuestionResult], ReportSummary]:
    llm = make_llm()
    app = build_agent_graph(llm)
    judge_llm = llm if use_judge else DummyLLM()

    questions = _load_questions(questions_path)
    if limit:
        questions = questions[:limit]

    results: list[QuestionResult] = []
    for q in questions:
        query = _query_text(q)
        t0 = time.perf_counter()
        state = app.invoke({"query": query})
        elapsed = (time.perf_counter() - t0) * 1000

        citations = state.get("citations") or []
        citation_ids = [_citation_to_id(c) for c in citations]
        # The RAG docs list holds everything retrieved across sub-questions.
        retrieved_ids = list({_citation_to_id(d) for d in (state.get("rag_docs") or [])})

        expected = q.get("expected_citations") or []
        recall = _recall(expected, retrieved_ids)
        cprec = _citation_precision(citation_ids, retrieved_ids)

        actual_intent = state.get("intent", "")
        intent_correct = (actual_intent == q.get("expected_intent", ""))

        draft = state.get("final_answer", "")
        ctx_hint = "\n\n".join(state.get("rag_contexts") or [])
        quality, judge_exp = _judge(judge_llm, q, draft, ctx_hint, query=query)

        results.append(
            QuestionResult(
                id=q["id"],
                category=q.get("category", "?"),
                query=query,
                expected_citations=expected,
                expected_intent=q.get("expected_intent", ""),
                actual_intent=actual_intent,
                retrieved_ids=retrieved_ids,
                citation_ids=citation_ids,
                final_answer=draft,
                latency_ms=elapsed,
                retrieval_recall=recall,
                citation_precision=cprec,
                answer_quality=quality,
                intent_correct=intent_correct,
                judge_explanation=judge_exp,
            )
        )
    return results, _summarize(results)


def _query_text(q: dict) -> str:
    # Derive a natural question from the id if the YAML omits a prose 'query'
    # field (some entries rely on category + reference for spec only).
    if q.get("query"):
        return q["query"]
    canned = {
        "q01_prohibited_practices": "What AI practices are prohibited under the AI Act?",
        "q02_high_risk_classification": "How does the AI Act classify an AI system as high-risk?",
        "q03_risk_management_requirements": "What are the requirements for risk management of high-risk AI systems?",
        "q04_gpai_systemic_risk_threshold": "When is a general-purpose AI model considered to have systemic risk, and what obligations apply?",
        "q05_deadline_prohibitions": "When do the AI Act prohibited practices start to apply?",
        "q06_deadline_high_risk": "When does the AI Act apply to high-risk systems under Annex I safety components?",
        "q07_gdpr_lawful_bases": "What are the lawful bases for processing personal data under GDPR?",
        "q08_right_to_erasure_scope": "When can a data subject exercise the GDPR right to erasure?",
        "q09_dpia_when_required": "When is a Data Protection Impact Assessment required under GDPR?",
        "q10_gdpr_ai_act_crossover": "How does GDPR Article 22 relate to AI Act high-risk systems?",
        "q11_ai_training_special_data": "Can AI systems be trained on GDPR special categories of personal data?",
        "q12_transparency_obligations": "What transparency obligations does the AI Act impose on high-risk systems?",
        "q13_gdpr_fines_principles": "What GDPR fines apply for breaching the basic processing principles?",
        "q14_greeting": "Hello!",
        "q15_ai_system_definition": "How does the AI Act define an 'AI system'?",
        "q16_gdpr_article9_referenced_by": "Which GDPR articles cross-reference the rules on special categories of personal data in Article 9?",
        "q17_aiact_article9_referenced_by": "Which AI Act articles invoke or rely on the risk management system requirements of Article 9?",
        "q18_article5_trigger_chain": "What other AI Act provisions does Article 5 directly reference for its prohibited-practice rules?",
    }
    return canned.get(q["id"], q["id"])


def _summarize(results: list[QuestionResult]) -> ReportSummary:
    if not results:
        return ReportSummary(0, 0.0, 0.0, 0.0, 0.0)
    valid_quality = [r.answer_quality for r in results if r.answer_quality == r.answer_quality]
    per_cat: dict[str, list[QuestionResult]] = {}
    for r in results:
        per_cat.setdefault(r.category, []).append(r)
    per_cat_summary = {
        cat: {
            "n": len(rs),
            "recall": mean(r.retrieval_recall for r in rs),
            "citation_precision": mean(r.citation_precision for r in rs),
            "intent_accuracy": sum(r.intent_correct for r in rs) / len(rs),
        }
        for cat, rs in per_cat.items()
    }
    return ReportSummary(
        n=len(results),
        mean_recall=mean(r.retrieval_recall for r in results),
        mean_citation_precision=mean(r.citation_precision for r in results),
        mean_answer_quality=mean(valid_quality) if valid_quality else float("nan"),
        intent_accuracy=sum(r.intent_correct for r in results) / len(results),
        per_category=per_cat_summary,
        mean_latency_ms=mean(r.latency_ms for r in results),
    )


# --- report -----------------------------------------------------------------

def _write_report(
    path: Path,
    results: list[QuestionResult],
    summary: ReportSummary,
    llm_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines.append(f"# Eval report — {ts}\n")
    lines.append(f"- LLM backend: `{llm_name}`")
    lines.append(f"- Questions: {summary.n}")
    lines.append(f"- **Retrieval recall (macro):** {summary.mean_recall:.2%}")
    lines.append(f"- **Citation precision:** {summary.mean_citation_precision:.2%}")
    lines.append(f"- **Intent routing accuracy:** {summary.intent_accuracy:.2%}")
    q = summary.mean_answer_quality
    lines.append(
        f"- **Answer quality (LLM-as-judge, 1–5):** "
        + (f"{q:.2f}" if q == q else "skipped")
    )
    lines.append(f"- **Mean latency:** {summary.mean_latency_ms:.0f} ms\n")

    lines.append("## Per-category breakdown\n")
    lines.append("| Category | N | Recall | Cite-prec | Intent acc |")
    lines.append("|---|---|---|---|---|")
    for cat, s in sorted(summary.per_category.items()):
        lines.append(
            f"| {cat} | {s['n']} | {s['recall']:.0%} | "
            f"{s['citation_precision']:.0%} | {s['intent_accuracy']:.0%} |"
        )

    lines.append("\n## Per-question results\n")
    for r in results:
        lines.append(f"### {r.id} — {r.category}")
        lines.append(f"- **Q:** {r.query}")
        lines.append(
            f"- intent: expected=`{r.expected_intent}` got=`{r.actual_intent}` "
            f"→ {'✅' if r.intent_correct else '❌'}"
        )
        lines.append(
            f"- recall: {r.retrieval_recall:.0%} "
            f"(expected={r.expected_citations}, "
            f"retrieved top={r.retrieved_ids[:8]})"
        )
        lines.append(f"- citation precision: {r.citation_precision:.0%} (cites={r.citation_ids})")
        lines.append(f"- latency: {r.latency_ms:.0f} ms")
        if r.answer_quality == r.answer_quality:
            lines.append(f"- LLM-judge: {r.answer_quality:.1f} / 5 — {r.judge_explanation}")
        lines.append(f"- answer:\n\n> {r.final_answer.strip().replace(chr(10), ' ')}\n")

    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the eval suite.")
    parser.add_argument("--questions", default="eval/questions.yaml")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--report-dir", default="eval/reports")
    args = parser.parse_args(argv)

    llm = make_llm()
    results, summary = run_eval(
        Path(args.questions), use_judge=not args.no_judge, limit=args.limit
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = Path(args.report_dir) / f"report_{ts}.md"
    _write_report(report_path, results, summary, llm.name)

    # Console summary for CI-friendly output.
    print(json.dumps(
        {
            "llm": llm.name,
            "n": summary.n,
            "mean_recall": round(summary.mean_recall, 3),
            "citation_precision": round(summary.mean_citation_precision, 3),
            "intent_accuracy": round(summary.intent_accuracy, 3),
            "mean_answer_quality": (
                round(summary.mean_answer_quality, 2)
                if summary.mean_answer_quality == summary.mean_answer_quality
                else None
            ),
            "mean_latency_ms": round(summary.mean_latency_ms, 1),
            "report": str(report_path),
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
