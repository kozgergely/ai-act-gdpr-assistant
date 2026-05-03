"""Shared state type for the main agent graph.

LangGraph state objects are TypedDicts whose fields can be annotated with a
*reducer* — a binary callable that combines the previous value with a node's
return value. Fields without a reducer overwrite on each update; fields
annotated with ``Annotated[T, add]`` accumulate across the run.

This module is the single source of truth for the agent's working memory.
Every node consumes a partial :class:`AgentState` and returns a partial
update; LangGraph handles the merge.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict, total=False):
    """Working memory passed between nodes in the main agent graph.

    Field semantics:

    * **Inputs / scalars (overwrite on update):**

      ``query``: the original natural-language question from the user.
      ``intent``: router output, one of ``"rag"`` / ``"tool"`` / ``"direct"``.
      ``tool_name``: when ``intent == "tool"``, names the tool to invoke
      (``"web_search"`` or ``"deadline_calc"``).
      ``plan``: planner output — list of sub-questions to retrieve over.
      ``remaining``: queue of sub-questions yet to be processed by
      ``rag_invoker``; the loop reuses this field rather than tracking an
      index.
      ``tool_result``: structured payload returned by the tool executor.
      ``draft_answer``: the composer's working answer; may be replaced on
      a verifier-triggered retry.
      ``verification``: latest verifier output (``{"grounded", "missing"}``).
      ``retries``: count of verifier-triggered retries; capped by ``MAX_RETRIES``.
      ``final_answer``: the answer the verifier promoted to be returned to
      the user.
      ``citations``: deduplicated citation list extracted from ``rag_docs``.

    * **Accumulators (``Annotated[..., add]`` — concatenated on update):**

      ``rag_contexts``: per-sub-question retrieval contexts assembled by
      the RAG subgraph; appended in order.
      ``rag_docs``: every fused doc surfaced across sub-questions, used
      to compute citations and feed the verifier.
      ``trace``: human-readable per-node breadcrumbs shown in the UI and
      reports.
    """

    query: str
    intent: str
    tool_name: str
    plan: list[str]
    remaining: list[str]
    rag_contexts: Annotated[list[str], add]
    rag_docs: Annotated[list[dict[str, Any]], add]
    tool_result: dict[str, Any]
    draft_answer: str
    verification: dict[str, Any]
    retries: int
    final_answer: str
    citations: list[dict[str, Any]]
    trace: Annotated[list[str], add]
