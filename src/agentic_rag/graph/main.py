"""Main agent LangGraph — orchestrates router → planner → rag/tool → composer → verifier.

Satisfies the brief's requirements:
 * >=5 nodes (router, planner, rag_invoker, tool_executor, composer, verifier)
 * autonomous conditional routing at router and verifier
 * sub-task decomposition (planner -> loop over sub-questions in rag_invoker)
 * explicit state management (AgentState TypedDict)
 * >=2 tools (RAG subgraph + web_search + deadline_calc)
 * dedicated RAG subgraph, composed in as a node but built separately
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agentic_rag.graph.nodes import (
    composer_node,
    planner_node,
    rag_invoker_node,
    router_node,
    tool_executor_node,
    verifier_node,
)
from agentic_rag.graph.state import AgentState
from agentic_rag.llm.base import LLM, make_llm
from agentic_rag.rag.subgraph import build_rag_subgraph


def _route_after_router(state: AgentState) -> str:
    """Branch out of the router based on the classified intent.

    Returns the name of the next node: ``planner`` for retrieval-bound
    questions, ``tool_executor`` for tool-bound ones, ``composer`` for
    direct/trivial replies that need no upstream work.
    """
    intent = state.get("intent", "rag")
    if intent == "tool":
        return "tool_executor"
    if intent == "direct":
        return "composer"
    return "planner"


def _route_after_rag(state: AgentState) -> str:
    """Either loop ``rag_invoker`` for the next sub-question or move on.

    The planner queues sub-questions in ``state["remaining"]``. Each
    ``rag_invoker`` invocation pops one; the conditional edge sends control
    back here until the queue is empty, at which point we hand off to the
    composer.
    """
    return "rag_invoker" if (state.get("remaining") or []) else "composer"


def _route_after_verifier(state: AgentState) -> str:
    """Decide between ending the run and looping back for another retrieval pass.

    Ends the run when the verifier marks the draft as grounded (or when
    retries have been exhausted — the verifier signals this by leaving
    ``remaining`` empty). Otherwise re-enters the rag_invoker loop with a
    re-queued plan.
    """
    v = state.get("verification") or {}
    if v.get("grounded", True):
        return END
    if (state.get("remaining") or []):
        # Retry signal from the verifier: re-queue sub-questions.
        return "rag_invoker"
    return END


def build_agent_graph(llm: LLM | None = None):
    """Compile the main agent graph and return a runnable LangGraph app.

    The same ``llm`` is injected into every node that needs it. The RAG
    subgraph is compiled once and embedded as a regular node, so the main
    graph keeps a single hop count of 6 nodes (router, planner, rag_invoker,
    tool_executor, composer, verifier) plus the dedicated RAG subgraph that
    sits behind ``rag_invoker``.

    Args:
        llm: an :class:`LLM` instance to use for every LLM-backed node. When
            ``None``, :py:func:`make_llm` resolves the configured backend
            (Ollama with Dummy fallback).

    Returns:
        A compiled LangGraph app exposing the standard ``invoke`` /
        ``stream`` API.
    """
    llm = llm or make_llm()
    rag_subgraph = build_rag_subgraph(llm)

    g = StateGraph(AgentState)
    g.add_node("router", router_node(llm))
    g.add_node("planner", planner_node(llm))
    g.add_node("rag_invoker", rag_invoker_node(rag_subgraph))
    g.add_node("tool_executor", tool_executor_node(llm))
    g.add_node("composer", composer_node(llm))
    g.add_node("verifier", verifier_node(llm))

    g.add_edge(START, "router")
    g.add_conditional_edges(
        "router",
        _route_after_router,
        {"planner": "planner", "tool_executor": "tool_executor", "composer": "composer"},
    )
    g.add_edge("planner", "rag_invoker")
    g.add_conditional_edges(
        "rag_invoker",
        _route_after_rag,
        {"rag_invoker": "rag_invoker", "composer": "composer"},
    )
    g.add_edge("tool_executor", "composer")
    g.add_edge("composer", "verifier")
    g.add_conditional_edges(
        "verifier",
        _route_after_verifier,
        {"rag_invoker": "rag_invoker", END: END},
    )
    return g.compile()


def run_query(query: str, llm: LLM | None = None) -> AgentState:
    """Convenience wrapper for one-shot CLI use.

    Builds a fresh graph (no caching) and invokes it with ``query`` as the
    only seed field. For repeated queries prefer :py:func:`build_agent_graph`
    once and call ``app.invoke({"query": ...})`` per question to avoid
    re-loading the embedding model and citation graph each time.
    """
    app = build_agent_graph(llm)
    result = app.invoke({"query": query})
    return result
