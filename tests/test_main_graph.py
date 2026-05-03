"""End-to-end smoke test for the main agent graph."""

from __future__ import annotations

import os

import pytest

from agentic_rag.graph.main import build_agent_graph
from agentic_rag.llm.base import DummyLLM


def _indices_ready() -> bool:
    return os.path.exists("data/chroma") and os.path.exists(
        "data/graph/citation_graph.pkl"
    )


@pytest.mark.skipif(not _indices_ready(), reason="indices not built")
def test_rag_flow_prohibited():
    app = build_agent_graph(DummyLLM())
    out = app.invoke({"query": "What are prohibited AI practices?"})
    assert out.get("final_answer"), "expected a final answer"
    trace = out.get("trace") or []
    kinds = {line.split(":")[0] for line in trace}
    # Must go through router, planner, rag_invoker, composer, verifier.
    assert {"router", "planner", "rag_invoker", "composer", "verifier"} <= kinds, (
        f"expected all core nodes to run, got {sorted(kinds)}"
    )


@pytest.mark.skipif(not _indices_ready(), reason="indices not built")
def test_tool_flow_deadline():
    """Query with deadline keyword should route to tool path."""
    app = build_agent_graph(DummyLLM())
    out = app.invoke({"query": "When does the AI Act prohibited practices deadline apply?"})
    trace = out.get("trace") or []
    assert any("tool_executor" in line for line in trace), (
        f"expected tool_executor in trace, got {trace}"
    )


@pytest.mark.skipif(not _indices_ready(), reason="indices not built")
def test_citations_extracted():
    app = build_agent_graph(DummyLLM())
    out = app.invoke({"query": "What are prohibited AI practices?"})
    citations = out.get("citations") or []
    assert citations, "expected at least one citation"
    assert all("regulation" in c for c in citations)
