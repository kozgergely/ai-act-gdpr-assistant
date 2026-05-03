"""End-to-end smoke test for the RAG subgraph using the Dummy LLM.

Assumes the indices have already been built (``scripts/build_indices.py``).
"""

from __future__ import annotations

import os

import pytest

from agentic_rag.llm.base import DummyLLM
from agentic_rag.rag.subgraph import build_rag_subgraph


@pytest.fixture(autouse=True)
def _dummy_llm(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "dummy")


def _indices_ready() -> bool:
    return os.path.exists("data/chroma") and os.path.exists(
        "data/graph/citation_graph.pkl"
    )


@pytest.mark.skipif(not _indices_ready(), reason="indices not built")
def test_prohibited_practices_retrieval():
    sg = build_rag_subgraph(DummyLLM())
    out = sg.invoke({"query": "What are prohibited AI practices under the AI Act?"})
    assert out["context"], "context should not be empty"
    # Long articles are split into sub-chunks (e.g. 5.1, 5.2, 5.3, 5.4) by the
    # parser's soft-split; we accept any sub-chunk of Article 5 as a hit.
    ids = {h["id"] for h in out["fused"]}
    article_5_chunks = {i for i in ids if i == "AI Act:article:5" or i.startswith("AI Act:article:5.")}
    assert article_5_chunks, (
        f"expected at least one chunk of Article 5 among fused hits, got {sorted(ids)}"
    )


@pytest.mark.skipif(not _indices_ready(), reason="indices not built")
def test_graph_expansion_pulls_neighbors():
    """Graph expansion must surface at least one chunk that vector retrieval missed.

    The exact neighbouring article is corpus-dependent (e.g. on the fixture
    Article 5 → Article 9; on the real EUR-Lex AI Act the cross-reference
    edges look quite different), so we assert the *mechanism* — that some
    chunk in the fused output came in via the citation graph rather than
    the semantic vector search.
    """
    sg = build_rag_subgraph(DummyLLM())
    out = sg.invoke({"query": "prohibited AI practices"})
    sources = {h["id"]: h["sources"] for h in out["fused"]}
    graph_only = [cid for cid, srcs in sources.items() if srcs == ["graph"]]
    assert graph_only, (
        f"expected at least one graph-only chunk in the fused result, got {sources}"
    )
