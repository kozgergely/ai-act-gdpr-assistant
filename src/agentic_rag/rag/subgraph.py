"""Hybrid RAG subgraph (LangGraph) — vector retrieval + citation-graph expansion.

Nodes:
  1. query_rewrite   — multi-query expansion: the LLM emits up to 3 alternative
                       phrasings of the user question (original always kept).
                       This is NOT HyDE — we paraphrase the question, not the
                       hypothetical answer.
  2. vector_retrieve — Chroma top-k for each rewrite
  3. graph_expand    — 1-hop neighbors of the top articles via citation graph
  4. fuse_rerank     — Reciprocal Rank Fusion, dedupe by chunk id
  5. context_assemble — pack into a prompt string with provenance headers

The subgraph is compiled once via ``build_rag_subgraph()`` and invoked by the
main agent as a regular node.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agentic_rag.config import settings
from agentic_rag.llm.base import LLM, make_llm
from agentic_rag.rag.store import get_collection, load_graph

RRF_K = 60
"""Smoothing constant in the Reciprocal Rank Fusion formula ``1/(RRF_K + rank)``.

The standard value of 60 (Cormack et al., 2009) softens the peak so that
top-ranked items dominate but lower-ranked items still contribute meaningful
mass when they appear in multiple rankings.
"""


class RAGState(TypedDict, total=False):
    """Working memory for the RAG subgraph.

    Each field is populated by a specific node:

    * ``query``: input from the parent graph.
    * ``rewritten_queries``: produced by ``query_rewrite``.
    * ``vector_hits``: produced by ``vector_retrieve`` (one list, ranks 0..N).
    * ``graph_hits``: produced by ``graph_expand`` (separate ranking).
    * ``fused``: produced by ``fuse_rerank`` (top-K final list).
    * ``context``: produced by ``context_assemble`` and returned to the caller.
    * ``trace``: per-node breadcrumbs concatenated for observability.
    """

    query: str
    rewritten_queries: list[str]
    vector_hits: list[dict[str, Any]]
    graph_hits: list[dict[str, Any]]
    fused: list[dict[str, Any]]
    context: str
    trace: Annotated[list[str], "append"]


# --- nodes -----------------------------------------------------------------

def _rewrite_node(llm: LLM):
    """Build the query-rewrite node — multi-query expansion (not HyDE).

    Asks the LLM for 2–3 alternative phrasings of the user question. The
    original query is always included in the output list so retrieval has a
    safe baseline even if the LLM produces nothing usable. We deliberately
    cap the list at three to keep the per-query cost bounded — each phrasing
    triggers an extra Chroma lookup downstream.
    """

    def run(state: RAGState) -> RAGState:
        schema = {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Up to 3 paraphrases or sub-queries.",
                }
            },
        }
        prompt = (
            "You expand a user question about EU AI Act / GDPR into 2-3 alternative "
            "phrasings or sub-queries that may match different legal phrasings. "
            "Return a JSON object with a 'queries' array including the original.\n\n"
            f"Original: {state['query']}"
        )
        try:
            out = llm.generate_json(prompt, schema)
            queries = [q for q in (out.get("queries") or []) if isinstance(q, str)]
        except Exception:
            queries = []
        if state["query"] not in queries:
            queries.insert(0, state["query"])
        queries = queries[:3]
        return {
            "rewritten_queries": queries,
            "trace": [f"rewrite: {len(queries)} queries"],
        }

    return run


def _vector_node():
    """Build the vector-retrieve node — Chroma top-K per rewritten query.

    Loads the collection once at factory time so the embedding function
    initialization (which loads the sentence-transformer model) is paid once
    per process instead of per query. Each rewritten query produces an
    independent top-K list; downstream RRF in ``fuse_rerank`` consolidates
    duplicates across queries.
    """
    col = get_collection()

    def run(state: RAGState) -> RAGState:
        queries = state.get("rewritten_queries") or [state["query"]]
        hits: list[dict[str, Any]] = []
        for q_idx, q in enumerate(queries):
            res = col.query(query_texts=[q], n_results=settings.top_k_vector)
            ids = res["ids"][0]
            docs = res["documents"][0]
            metas = res["metadatas"][0]
            dists = res["distances"][0]
            for rank, (i, d, m, dist) in enumerate(zip(ids, docs, metas, dists)):
                hits.append(
                    {
                        "id": i,
                        "text": d,
                        "meta": m,
                        "distance": dist,
                        "rank_in_query": rank,
                        "query_idx": q_idx,
                        "source": "vector",
                    }
                )
        return {
            "vector_hits": hits,
            "trace": [f"vector_retrieve: {len(hits)} hits across {len(queries)} queries"],
        }

    return run


def _graph_node():
    """Build the graph-expand node — 1-hop citation neighbours of vector seeds.

    Uses the article-level vector hits as seeds and walks ``graph_hops`` hops
    on the directed citation graph (in + out edges). Newly reached nodes
    that are not already vector hits are pulled from Chroma by chunk id and
    returned as a separate ranking. The ranking is built from a strength
    score ``1/(1 + seed_rank) * 0.5^hop`` so that:

    * neighbours of strong vector hits outrank neighbours of weak ones,
    * deeper hops are exponentially discounted,
    * a node reachable from multiple seeds keeps its best (max) strength.

    Returning a *new* ranking — rather than appending to the vector list —
    is what lets RRF treat graph-only hits as peers of vector hits.
    """
    graph = load_graph()
    col = get_collection()

    def run(state: RAGState) -> RAGState:
        if settings.graph_hops <= 0:
            return {"graph_hits": [], "trace": ["graph_expand: disabled"]}

        # Collapse vector hits to the best rank per article-level node.
        seed_ranks: dict[str, int] = {}
        for h in state.get("vector_hits") or []:
            m = h["meta"]
            base = f"{m['regulation']}:{m['kind']}:{str(m['number']).split('.')[0]}"
            seed_ranks.setdefault(base, h["rank_in_query"])

        # BFS up to graph_hops hops from each seed (out+in edges).
        # Strength = best (lowest) seed rank reaching the node, minus a per-hop penalty.
        strength: dict[str, float] = {}
        for seed, seed_rank in seed_ranks.items():
            if seed not in graph:
                continue
            frontier = {seed}
            for hop in range(settings.graph_hops):
                nxt: set[str] = set()
                for n in frontier:
                    nxt.update(graph.successors(n))
                    nxt.update(graph.predecessors(n))
                nxt -= seed_ranks.keys()
                for n in nxt:
                    # Higher is stronger; hop penalty shrinks deeper nodes.
                    s = (1.0 / (1 + seed_rank)) * (0.5 ** hop)
                    strength[n] = max(strength.get(n, 0.0), s)
                frontier = nxt

        # Rank expanded nodes by strength so RRF treats them as a proper second ranking.
        ranked = sorted(strength.items(), key=lambda kv: kv[1], reverse=True)
        expanded_rank: dict[str, int] = {n: r for r, (n, _) in enumerate(ranked)}

        hits: list[dict[str, Any]] = []
        if expanded_rank:
            chunk_ids: list[str] = []
            for node_id in expanded_rank:
                chunk_ids.extend(graph.nodes[node_id].get("chunk_ids", []) or [])
            chunk_ids = list(dict.fromkeys(chunk_ids))
            if chunk_ids:
                got = col.get(ids=chunk_ids, include=["documents", "metadatas"])
                for i, d, m in zip(got["ids"], got["documents"], got["metadatas"]):
                    base = f"{m['regulation']}:{m['kind']}:{str(m['number']).split('.')[0]}"
                    rank = expanded_rank.get(base, 999)
                    hits.append(
                        {
                            "id": i,
                            "text": d,
                            "meta": m,
                            "distance": None,
                            "rank_in_query": rank,
                            "query_idx": 0,
                            "source": "graph",
                        }
                    )
        return {
            "graph_hits": hits,
            "trace": [
                f"graph_expand: seeds={len(seed_ranks)} expanded_nodes={len(expanded_rank)} "
                f"chunks={len(hits)}"
            ],
        }

    return run


def _rrf(rank: int) -> float:
    """Reciprocal Rank Fusion score for a single ranking position."""
    return 1.0 / (RRF_K + rank)


def _fuse_node():
    """Build the fuse-rerank node — RRF over vector + graph rankings.

    Walks every hit from both source lists, scoring it by its rank in its
    own ranking. Hits that appear in both rankings (rare in practice
    because graph_expand explicitly excludes vector seeds at article level)
    accumulate their scores and end up with a ``sources`` set listing both
    origins. The resulting list is sorted by descending RRF score and
    truncated to ``settings.top_k_final``.
    """

    def run(state: RAGState) -> RAGState:
        scored: dict[str, dict[str, Any]] = {}
        for h in (state.get("vector_hits") or []) + (state.get("graph_hits") or []):
            prev = scored.get(h["id"])
            score = _rrf(h["rank_in_query"])
            if prev is None:
                scored[h["id"]] = {**h, "score": score, "sources": {h["source"]}}
            else:
                prev["score"] += score
                prev["sources"].add(h["source"])

        fused = sorted(scored.values(), key=lambda x: x["score"], reverse=True)
        fused = fused[: settings.top_k_final]
        # Serializable 'sources' field.
        for h in fused:
            h["sources"] = sorted(h["sources"])
        return {
            "fused": fused,
            "trace": [f"fuse_rerank: kept top {len(fused)}"],
        }

    return run


def _format_paragraphs(meta: dict[str, Any]) -> str:
    """Render paragraph numbers as a compact range like ``'paragraphs 1-3'``.

    Folds contiguous integers into ``a-b`` ranges and joins disjoint runs
    with commas. Returns the empty string when no paragraph metadata is
    available, which is normal for short articles, recitals, and the
    hand-written fixture chunks.
    """
    raw = (meta.get("paragraphs") or "").strip()
    if not raw:
        return ""
    nums = [n for n in raw.split(",") if n]
    if not nums:
        return ""
    if len(nums) == 1:
        return f"paragraph {nums[0]}"
    # Compact contiguous ranges (e.g. 1,2,3 -> "1-3"; 1,2,5 -> "1-2, 5").
    ints: list[int] = []
    for n in nums:
        try:
            ints.append(int(n))
        except ValueError:
            pass
    if not ints:
        return f"paragraphs {', '.join(nums)}"
    ints.sort()
    ranges: list[str] = []
    a = b = ints[0]
    for n in ints[1:]:
        if n == b + 1:
            b = n
        else:
            ranges.append(f"{a}-{b}" if a != b else f"{a}")
            a = b = n
    ranges.append(f"{a}-{b}" if a != b else f"{a}")
    return f"paragraphs {', '.join(ranges)}"


def _context_node():
    """Build the context-assemble node — formats the final prompt context block.

    Each fused chunk gets a one-line provenance header containing the
    regulation, kind/number, paragraph range (when present), title, page,
    and the merged ``sources`` flag (``vector``, ``graph``, or
    ``vector+graph``). Header + chunk text pairs are separated by a blank
    line so the downstream composer can clearly see where one source ends
    and the next begins.
    """

    def run(state: RAGState) -> RAGState:
        parts: list[str] = []
        for h in state.get("fused") or []:
            m = h["meta"]
            paragraph_part = _format_paragraphs(m)
            header_inner = f"{m['regulation']} {m['kind'].capitalize()} {m['number']}"
            if paragraph_part:
                header_inner += f" ({paragraph_part})"
            if m.get("title"):
                header_inner += f" — {m['title']}"
            header = f"[{header_inner}, p.{m['page']}; via {'+'.join(h['sources'])}]"
            parts.append(f"{header}\n{h['text']}")
        return {
            "context": "\n\n".join(parts),
            "trace": [f"context_assemble: {len(parts)} docs in context"],
        }

    return run


# --- graph builder ---------------------------------------------------------

def build_rag_subgraph(llm: LLM | None = None):
    """Compile the RAG subgraph and return a runnable LangGraph app.

    The five nodes run in order: rewrite → vector → graph → fuse → context.
    Only ``query_rewrite`` needs an LLM; the other nodes are deterministic
    given the same indices. ``llm`` defaults to :py:func:`make_llm` so the
    subgraph can be invoked standalone for testing without wiring up the
    main agent graph.
    """
    llm = llm or make_llm()
    g = StateGraph(RAGState)
    g.add_node("query_rewrite", _rewrite_node(llm))
    g.add_node("vector_retrieve", _vector_node())
    g.add_node("graph_expand", _graph_node())
    g.add_node("fuse_rerank", _fuse_node())
    g.add_node("context_assemble", _context_node())

    g.add_edge(START, "query_rewrite")
    g.add_edge("query_rewrite", "vector_retrieve")
    g.add_edge("vector_retrieve", "graph_expand")
    g.add_edge("graph_expand", "fuse_rerank")
    g.add_edge("fuse_rerank", "context_assemble")
    g.add_edge("context_assemble", END)
    return g.compile()
