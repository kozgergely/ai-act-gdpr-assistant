"""Streamlit UI for the agentic RAG assistant.

Layout: two-column split.
  - Left: chat history (user questions + final answers + citations)
  - Right: collapsible per-turn "Agent trace" showing router / plan / RAG steps
           / tool calls / verifier outcome.

Run with:  streamlit run src/agentic_rag/ui/app.py
"""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from agentic_rag.config import settings
from agentic_rag.graph.main import build_agent_graph
from agentic_rag.llm.base import make_llm

st.set_page_config(
    page_title="EU AI Act + GDPR Compliance Assistant",
    page_icon="📜",
    layout="wide",
)


# --- session state ----------------------------------------------------------

def _init_state() -> None:
    """Seed session-scoped slots on first render so reruns can read them safely."""
    if "history" not in st.session_state:
        st.session_state.history = []  # list[dict]: query, final_answer, citations, trace, timings


@st.cache_resource(show_spinner=False)
def _build_agent_cached():
    """Process-wide singleton: load the embedding model, indices, and graph once.

    ``st.cache_resource`` keeps the returned object alive across reruns AND
    across sessions inside the same Streamlit process. The first refresh in
    a fresh container pays the ~8s cold-start (sentence-transformer load,
    Chroma open, citation graph unpickle). Every subsequent refresh — and
    every other user hitting the same container — gets the cached instance
    and renders instantly.
    """
    llm = make_llm()
    app = build_agent_graph(llm)
    return app, llm.name


def _ensure_agent() -> tuple:
    """Return ``(agent, llm_name)``, surfacing a spinner during the cold-start."""
    with st.spinner("Loading indices and embedding model — first time only..."):
        return _build_agent_cached()


def _indices_present() -> bool:
    """``True`` iff both Chroma and the citation graph exist on disk."""
    return (
        Path(settings.chroma_dir).exists()
        and Path(settings.graph_path).exists()
    )


# --- sidebar ----------------------------------------------------------------

def _sidebar() -> None:
    """Render the left-hand sidebar (setup status, sample questions, controls)."""
    with st.sidebar:
        st.markdown("## Setup")
        if not _indices_present():
            st.error(
                "Indices missing.\n\n"
                "Run `python scripts/build_indices.py` once to build the vector "
                "store and citation graph."
            )
            st.stop()
        # Load (or pull from cache) before announcing readiness, so the green
        # message reflects the actual state — model + indices both ready.
        agent, llm_name = _ensure_agent()
        st.session_state.agent = agent
        st.session_state.llm_name = llm_name
        st.success("Agent ready.")
        st.info(f"**LLM backend:** `{llm_name}`")
        st.caption(
            f"Embeddings: `{settings.embedding_model}`\n\n"
            f"Top-k vector: {settings.top_k_vector} | graph hops: {settings.graph_hops} | "
            f"top-k final: {settings.top_k_final}"
        )
        st.markdown("---")
        st.markdown("## Sample questions")
        for q in [
            "What are prohibited AI practices under the AI Act?",
            "How does GDPR Article 22 relate to AI Act high-risk systems?",
            "When does the AI Act apply to general-purpose AI models?",
            "What is a data protection impact assessment and when is it required?",
            "Which AI Act article governs risk management for high-risk systems?",
        ]:
            if st.button(q, use_container_width=True, key=f"sample_{hash(q)}"):
                st.session_state._prefill = q
                st.rerun()
        st.markdown("---")
        if st.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.rerun()


# --- rendering helpers ------------------------------------------------------

def _render_citation(c: dict) -> str:
    """Format a single citation dict as a markdown bullet line."""
    reg = c["regulation"]
    kind = c["kind"].capitalize()
    num = str(c["number"]).split(".")[0]
    title = c.get("title") or ""
    src = "+".join(c.get("sources") or [])
    suffix = f" — {title}" if title else ""
    return f"**{reg} {kind} {num}**{suffix} _(via {src}, p.{c.get('page', '?')})_"


def _render_trace(trace: list[str]) -> None:
    """Render the per-turn node trace as collapsible sections.

    Trace lines are grouped by the leading node label; the router and
    composer sections are expanded by default because they tend to contain
    the most insight into how the agent decided what to do.
    """
    grouped: dict[str, list[str]] = {}
    order: list[str] = []
    for line in trace:
        head, _, rest = line.partition(":")
        if head not in grouped:
            order.append(head)
            grouped[head] = []
        grouped[head].append(rest.strip())

    for head in order:
        entries = grouped[head]
        label = head.replace("_", " ").capitalize()
        with st.expander(f"**{label}** ({len(entries)})", expanded=(head in {"router", "composer"})):
            for e in entries:
                st.write(f"- {e}")


def _render_left(history: list[dict]) -> None:
    """Render the chat panel: user query → assistant answer → citations + meta."""
    st.subheader("Chat")
    if not history:
        st.caption("Ask a question to get started.")
    for turn in history:
        with st.chat_message("user"):
            st.write(turn["query"])
        with st.chat_message("assistant"):
            st.write(turn["final_answer"] or "_(no answer generated)_")
            cits = turn.get("citations") or []
            if cits:
                with st.expander(f"Citations ({len(cits)})", expanded=False):
                    for c in cits:
                        st.markdown("- " + _render_citation(c))
            tool_result = turn.get("tool_result")
            if tool_result:
                with st.expander("Tool result", expanded=False):
                    st.json(tool_result)
            st.caption(
                f"⏱ {turn['timings']['total_ms']:.0f} ms • "
                f"intent: `{turn.get('intent', '?')}`"
            )


def _render_right(history: list[dict]) -> None:
    """Render the agent-trace panel — one expander per chat turn."""
    st.subheader("Agent trace")
    if not history:
        st.caption("Per-turn node trace will appear here.")
        return
    # Show the most recent turn by default, others collapsed.
    for i, turn in enumerate(reversed(history)):
        idx = len(history) - i
        with st.expander(f"Turn {idx}: {turn['query'][:60]}...", expanded=(i == 0)):
            _render_trace(turn.get("trace") or [])


# --- main -------------------------------------------------------------------

def main() -> None:
    """Top-level Streamlit app entrypoint.

    Wires session state, sidebar, chat input, and the two-column layout.
    Designed to be called once per Streamlit script run; idempotent on
    reruns thanks to session-state caching.
    """
    _init_state()
    st.title("📜 EU AI Act + GDPR Compliance Assistant")
    st.caption(
        "Agentic RAG prototype — LangGraph + hybrid vector/citation-graph retrieval. "
        "Not legal advice."
    )
    _sidebar()

    prefill = st.session_state.pop("_prefill", "") if "_prefill" in st.session_state else ""
    query = st.chat_input("Ask a question about AI Act or GDPR...")
    if not query and prefill:
        query = prefill

    if query:
        t0 = time.perf_counter()
        result = st.session_state.agent.invoke({"query": query})
        elapsed_ms = (time.perf_counter() - t0) * 1000
        st.session_state.history.append(
            {
                "query": query,
                "final_answer": result.get("final_answer", ""),
                "citations": result.get("citations", []),
                "trace": result.get("trace", []),
                "intent": result.get("intent", ""),
                "tool_result": result.get("tool_result"),
                "timings": {"total_ms": elapsed_ms},
            }
        )

    left, right = st.columns([3, 2], gap="large")
    with left:
        _render_left(st.session_state.history)
    with right:
        _render_right(st.session_state.history)


if __name__ == "__main__":
    main()
