"""Main-graph node implementations.

Each node is a *factory*: a top-level function that takes its dependencies
(typically an :class:`LLM`) and returns a callable ``run(state) -> updates``.
This shape lets us inject the same LLM (or a Dummy substitute) into every
node from a single :py:func:`build_agent_graph` call without globals, and it
keeps each node a pure-ish function: deterministic given its inputs and a
fixed LLM, easy to unit-test in isolation.

The conditional routing logic lives in :mod:`agentic_rag.graph.main`; this
module is concerned only with what each node does locally to the state.
"""

from __future__ import annotations

import json
from typing import Any

from agentic_rag.graph.state import AgentState
from agentic_rag.llm.base import LLM, Message
from agentic_rag.tools.deadline import compute_deadline
from agentic_rag.tools.web_search import web_search

MAX_RETRIES = 1
"""Maximum number of verifier-driven retries before we accept the latest draft."""


# --- router ----------------------------------------------------------------

ROUTER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["rag", "tool", "direct"],
            "description": "rag: needs document retrieval. tool: needs web search or deadline calc. direct: trivial answer.",
        },
        "tool_name": {
            "type": "string",
            "enum": ["web_search", "deadline_calc", ""],
        },
        "reasoning": {"type": "string"},
    },
    "required": ["intent"],
}


def router_node(llm: LLM):
    """Build the router node — decides between RAG, tool, or direct response.

    The returned ``run`` callable classifies the user query into one of three
    intents and, for tool intents, names the specific tool. On any LLM error
    we fall back to ``"rag"`` because retrieval is the safe default for a
    legal-corpus assistant.
    """

    def run(state: AgentState) -> AgentState:
        prompt = (
            "You are the router for an EU AI Act / GDPR compliance assistant. "
            "Classify the user question into exactly one intent:\n"
            "  - 'rag': answer by retrieving from the AI Act / GDPR corpus (factual legal questions).\n"
            "  - 'tool': the answer needs either recent web information ('web_search') or a deadline/date computation ('deadline_calc').\n"
            "  - 'direct': trivial greeting or meta question (no retrieval needed).\n\n"
            f"Question: {state['query']}\n\n"
            "If intent is 'tool', also set tool_name. Otherwise leave tool_name empty."
        )
        try:
            out = llm.generate_json(prompt, ROUTER_SCHEMA)
        except Exception:
            out = {"intent": "rag", "tool_name": "", "reasoning": "fallback"}
        intent = out.get("intent", "rag")
        tool_name = out.get("tool_name", "") or ""
        return {
            "intent": intent,
            "tool_name": tool_name,
            "trace": [f"router: intent={intent} tool={tool_name or '-'}"],
        }

    return run


# --- planner ---------------------------------------------------------------

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "subquestions": {"type": "array", "items": {"type": "string"}},
    },
}


def planner_node(llm: LLM):
    """Build the planner node — decomposes the query into 1–3 sub-questions.

    For atomic questions the LLM is encouraged to return a single sub-question
    (effectively a no-op decomposition); for compound or comparative
    questions we fan out so each sub-question can be retrieved independently
    and the composer sees focused per-sub-question contexts.
    """

    def run(state: AgentState) -> AgentState:
        prompt = (
            "Decompose the following compliance question into 1-3 concrete "
            "retrieval-ready sub-questions. A single sub-question is fine if the "
            "query is already atomic.\n\n"
            f"Question: {state['query']}"
        )
        try:
            out = llm.generate_json(prompt, PLAN_SCHEMA)
            subqs = [s for s in (out.get("subquestions") or []) if isinstance(s, str)]
        except Exception:
            subqs = []
        if not subqs:
            subqs = [state["query"]]
        subqs = subqs[:3]
        return {
            "plan": subqs,
            "remaining": list(subqs),
            "trace": [f"planner: {len(subqs)} sub-question(s)"],
        }

    return run


# --- rag invoker -----------------------------------------------------------

def rag_invoker_node(rag_subgraph):
    """Build the rag-invoker node — runs the RAG subgraph for one sub-question.

    The node pops one sub-question from ``state["remaining"]`` per invocation;
    the conditional edge in :mod:`agentic_rag.graph.main` loops back here
    until the queue is empty, so retrieval contexts accumulate one
    sub-question at a time. The shared ``rag_contexts`` and ``rag_docs``
    fields use the ``add`` reducer (see :mod:`agentic_rag.graph.state`) so
    state survives the loop without manual book-keeping.

    Args:
        rag_subgraph: a compiled LangGraph state graph that takes
            ``{"query": ...}`` and returns the standard RAG subgraph output
            (``fused`` doc list + ``context`` string).
    """

    def run(state: AgentState) -> AgentState:
        remaining = list(state.get("remaining") or [])
        if not remaining:
            return {"trace": ["rag_invoker: no sub-questions left"]}
        subq = remaining.pop(0)
        result = rag_subgraph.invoke({"query": subq})
        # Shallow copy the docs we care about for citations.
        docs = [
            {
                "id": h["id"],
                "regulation": h["meta"]["regulation"],
                "kind": h["meta"]["kind"],
                "number": h["meta"]["number"],
                "title": h["meta"].get("title", ""),
                "page": h["meta"]["page"],
                "sources": h["sources"],
                "score": h["score"],
            }
            for h in (result.get("fused") or [])
        ]
        return {
            "remaining": remaining,
            "rag_contexts": [f"[Sub-question: {subq}]\n{result.get('context', '')}"],
            "rag_docs": docs,
            "trace": [f"rag_invoker: subq='{subq[:50]}...' got {len(docs)} docs"],
        }

    return run


# --- tool executor ---------------------------------------------------------

def tool_executor_node(llm: LLM | None = None):  # llm kept for future tool planning
    """Build the tool-executor node — dispatches to the selected non-retrieval tool.

    Currently routes to :py:func:`compute_deadline` (for application-date
    questions about the AI Act) or :py:func:`web_search` (DuckDuckGo).
    The ``llm`` argument is unused today; it is kept on the signature so a
    future iteration can let the LLM pick tool arguments without changing
    the graph wiring.
    """

    def run(state: AgentState) -> AgentState:
        tool = state.get("tool_name") or ""
        if tool == "deadline_calc":
            result = compute_deadline(state["query"]).to_dict()
        elif tool == "web_search":
            hits = web_search(state["query"], max_results=5)
            result = {"hits": [h.to_dict() for h in hits]}
        else:
            result = {"error": f"unknown or empty tool: {tool!r}"}
        return {
            "tool_result": result,
            "trace": [f"tool_executor: {tool} -> {json.dumps(result)[:120]}"],
        }

    return run


# --- composer --------------------------------------------------------------

def composer_node(llm: LLM):
    """Build the composer node — writes the user-facing draft answer.

    The draft is assembled from the accumulated ``rag_contexts`` and any
    ``tool_result`` placed in state by upstream nodes. The system prompt
    forces inline citations and a "not legal advice" disclaimer; we do not
    enforce these mechanically, but the verifier downstream will reject a
    draft whose claims aren't backed by the retrieved context.
    """

    def run(state: AgentState) -> AgentState:
        intent = state.get("intent", "rag")
        contexts = state.get("rag_contexts") or []
        tool_result = state.get("tool_result")

        context_block = "\n\n".join(contexts) if contexts else "(no retrieved context)"
        tool_block = (
            f"\n\nTool result:\n{json.dumps(tool_result, indent=2)}"
            if tool_result
            else ""
        )

        system = (
            "You are an EU AI Act + GDPR compliance assistant. Answer the user "
            "using ONLY the retrieved context and tool results. Cite sources "
            "inline as (Regulation Article N) or (Regulation Recital N). End "
            "with a one-line disclaimer: 'This is not legal advice.'"
        )
        user = (
            f"Question: {state['query']}\n\n"
            f"Retrieved context:\n{context_block}"
            f"{tool_block}\n\nWrite the answer now."
        )
        try:
            answer = llm.chat(
                [Message("system", system), Message("user", user)],
                temperature=0.1,
            )
        except Exception as e:
            answer = f"(composer error: {e})"
        citations = _extract_citations(state.get("rag_docs") or [])
        return {
            "draft_answer": answer,
            "citations": citations,
            "trace": [f"composer: draft {len(answer)} chars, {len(citations)} citations"],
        }

    return run


def _extract_citations(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse retrieved docs into one citation per article.

    A long article may surface as multiple chunks (e.g. ``article:9.1``,
    ``article:9.2``) but the user-facing citation list should show
    "Article 9" once. We key on ``(regulation, kind, top-level-number)`` and
    keep the highest-scoring chunk's metadata as the representative entry.
    Output is sorted by descending score so the most relevant articles are
    listed first in the UI.
    """
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for d in docs:
        key = (d["regulation"], d["kind"], str(d["number"]).split(".")[0])
        prev = by_key.get(key)
        if prev is None or d["score"] > prev["score"]:
            by_key[key] = d
    return sorted(by_key.values(), key=lambda x: -x["score"])


# --- verifier --------------------------------------------------------------

VERIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "grounded": {"type": "boolean"},
        "missing": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["grounded"],
}


def verifier_node(llm: LLM):
    """Build the verifier node — grounds the draft answer or triggers a retry.

    Behaviour:

    * For ``intent == "direct"`` we skip verification (no retrieval to ground
      against) and promote the draft straight to ``final_answer``.
    * Otherwise we ask the LLM whether the draft's claims are supported by
      the first retrieved context block. If grounded, the draft becomes
      ``final_answer`` and the graph terminates.
    * If not grounded and ``retries < MAX_RETRIES``, we re-queue the original
      plan in ``remaining``; the conditional edge sends the flow back to
      ``rag_invoker``. After ``MAX_RETRIES`` we accept the latest draft
      regardless and surface the verifier verdict so the UI can warn the user.
    """

    def run(state: AgentState) -> AgentState:
        if state.get("intent") == "direct":
            return {
                "verification": {"grounded": True, "missing": []},
                "final_answer": state.get("draft_answer", ""),
                "trace": ["verifier: skipped for direct intent"],
            }

        prompt = (
            "Determine whether the draft answer is grounded in the retrieved "
            "context. If the answer contains claims not supported by the "
            "context, list them in 'missing'.\n\n"
            f"Question: {state['query']}\n\n"
            f"Context:\n{(state.get('rag_contexts') or [''])[0][:2000]}\n\n"
            f"Draft answer:\n{state.get('draft_answer', '')}"
        )
        try:
            out = llm.generate_json(prompt, VERIFY_SCHEMA)
        except Exception:
            out = {"grounded": True, "missing": []}
        retries = state.get("retries", 0)
        grounded = bool(out.get("grounded", True))
        if grounded or retries >= MAX_RETRIES:
            return {
                "verification": out,
                "final_answer": state.get("draft_answer", ""),
                "trace": [f"verifier: grounded={grounded} retries={retries} -> END"],
            }
        # Signal a retry.
        return {
            "verification": out,
            "retries": retries + 1,
            "remaining": list(state.get("plan") or [state["query"]]),
            "trace": [f"verifier: not grounded, retry {retries + 1}"],
        }

    return run
