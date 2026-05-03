"""LLM abstraction with two backends: Ollama (real) and Dummy (deterministic).

The abstraction is intentionally narrow — just enough for the agent nodes:
 * ``chat(messages)`` for free-form generation
 * ``generate_json(prompt, schema)`` for structured output

The Dummy backend lets us exercise the whole graph without a running LLM, which
matters for CI, Docker build smoke tests, and the load-test scaffolding.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from agentic_rag.config import settings


@dataclass
class Message:
    """A single chat message with a role label and textual content.

    The role mirrors the OpenAI/Ollama chat-completion API: ``"system"`` for
    framing instructions, ``"user"`` for the human turn, ``"assistant"`` for
    prior model outputs in a multi-turn dialog.
    """

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        """Serialize to the dict shape expected by the underlying chat API."""
        return {"role": self.role, "content": self.content}


class LLM(Protocol):
    """Narrow LLM interface used by every node in the agent graph.

    Two operations cover all in-graph use cases:

    * :py:meth:`chat` for free-form text generation (used by the composer node).
    * :py:meth:`generate_json` for schema-constrained structured output (used
      by router, planner, and verifier nodes).

    Implementations are interchangeable behind :py:func:`make_llm`. The
    Protocol shape lets us swap an :class:`OllamaLLM` for a :class:`DummyLLM`
    without changing any node code.
    """

    def chat(self, messages: list[Message], *, temperature: float = 0.2) -> str:
        """Run a chat-completion call and return the assistant's text reply."""
        ...

    def generate_json(
        self, prompt: str, schema: dict[str, Any], *, temperature: float = 0.0
    ) -> dict[str, Any]:
        """Generate output that conforms to ``schema`` and parse it as JSON."""
        ...

    @property
    def name(self) -> str:
        """Human-readable identifier (used in traces and reports)."""
        ...


# --- Ollama backend ---------------------------------------------------------

class OllamaLLM:
    """Real-LLM backend talking to a local Ollama HTTP server.

    The class is a thin HTTP client: the model itself runs in a separate
    Ollama runner subprocess. We never load weights inside this Python
    process. The constructor performs no I/O; the first network call happens
    on the first :py:meth:`chat` or :py:meth:`generate_json` invocation.
    """

    def __init__(self, host: str, model: str):
        """Bind to an Ollama HTTP endpoint.

        Args:
            host: Base URL of the Ollama server, e.g. ``http://localhost:11434``.
            model: Ollama model tag, e.g. ``qwen2.5:7b-instruct``.
        """
        import ollama

        self._client = ollama.Client(host=host)
        self._model = model

    @property
    def name(self) -> str:
        return f"ollama:{self._model}"

    def chat(self, messages: list[Message], *, temperature: float = 0.2) -> str:
        """Send a chat-completion request and return the assistant content."""
        resp = self._client.chat(
            model=self._model,
            messages=[m.to_dict() for m in messages],
            options={"temperature": temperature},
        )
        return resp["message"]["content"]

    def generate_json(
        self, prompt: str, schema: dict[str, Any], *, temperature: float = 0.0
    ) -> dict[str, Any]:
        """Request a JSON-shaped response that matches ``schema``.

        We rely on Ollama's ``format="json"`` option to constrain decoding so
        the output is always parseable. If the model still returns
        prose-with-JSON (rare), we extract the first ``{...}`` block as a
        best-effort fallback before re-raising.

        Args:
            prompt: Free-form instruction to the model. The schema is
                appended automatically; callers do not need to do it
                themselves.
            schema: A minimal JSON Schema describing the desired shape.
            temperature: Sampling temperature; defaults to 0 for determinism.

        Returns:
            The parsed JSON object.

        Raises:
            json.JSONDecodeError: If neither the strict nor the fallback
                parse succeeds.
        """
        system = (
            "You return ONLY a valid JSON object matching the schema. "
            "No prose, no markdown fences."
        )
        resp = self._client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": (
                        f"{prompt}\n\nRespond with JSON matching this schema:\n"
                        f"{json.dumps(schema, indent=2)}"
                    ),
                },
            ],
            format="json",
            options={"temperature": temperature},
        )
        text = resp["message"]["content"].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Best-effort recovery — grab the first JSON object we can find.
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if m:
                return json.loads(m.group(0))
            raise


# --- Dummy backend (deterministic, for CI / no-LLM environments) ------------

class DummyLLM:
    """Deterministic, network-free stand-in for a real LLM.

    The class returns plausible outputs by inspecting prompt keywords with
    simple heuristics. It satisfies the :class:`LLM` Protocol so the agent
    graph runs end-to-end without Ollama — used by the test suite, the
    Docker smoke test, and any environment where a real model is unavailable.

    The outputs are not natural language: chat responses are placeholder
    strings, and structured outputs are minimal skeletons that still type-check.
    Use this backend to validate control flow, not generation quality.
    """

    @property
    def name(self) -> str:
        return "dummy"

    def chat(self, messages: list[Message], *, temperature: float = 0.2) -> str:
        """Return a placeholder reply chosen by inspecting the system message."""
        user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        sys_ = next((m.content for m in messages if m.role == "system"), "")
        if "verify" in sys_.lower() or "grounded" in sys_.lower():
            return "GROUNDED"
        if "compose" in sys_.lower() or "citation" in sys_.lower():
            return (
                "Based on the retrieved context, the regulation addresses this topic. "
                "See the citations below for the relevant articles. "
                "(Note: this is a dummy-LLM answer.)"
            )
        return f"[dummy] received query: {user[:120]}"

    def generate_json(
        self, prompt: str, schema: dict[str, Any], *, temperature: float = 0.0
    ) -> dict[str, Any]:
        """Return a heuristic structured output derived from prompt keywords.

        Recognizes router, planner, and verifier prompt shapes and emits
        plausible payloads for each. For unknown prompts, returns an empty
        skeleton built from ``schema`` so callers can still type-check the
        response.
        """
        p = prompt.lower()
        # Isolate the user-question portion from the prompt so that keywords
        # appearing in the system-formatted instructions don't bias the dummy.
        q_match = re.search(r"question:\s*(.+?)(?:\n\n|$)", prompt, flags=re.IGNORECASE)
        question_only = (q_match.group(1) if q_match else prompt).lower()

        # Router intents
        if "intent" in p or "route" in p or "classify" in p:
            intent = "rag"
            tool_name = ""
            if any(k in question_only for k in ["deadline", "when does", "months until",
                                                 "days until", "when do the ", "when will"]):
                intent = "tool"
                tool_name = "deadline_calc"
            elif any(k in question_only for k in ["news", "recent ruling", "latest update",
                                                   "recently announced", "enforcement news"]):
                intent = "tool"
                tool_name = "web_search"
            elif len(question_only) < 40 and "?" not in question_only:
                intent = "direct"
            return {
                "intent": intent,
                "tool_name": tool_name,
                "reasoning": "heuristic dummy classifier",
            }

        # Planner — return the user question as a single, atomic sub-question.
        # The legacy heuristic split the entire prompt (system instructions
        # included) on punctuation, leaking "Decompose the following..." and
        # similar boilerplate into the sub-question list, which polluted
        # downstream retrieval. We use the same Question-extraction trick as
        # the router instead.
        if "subquestion" in p or "decompose" in p or "plan" in p:
            return {"subquestions": [question_only.strip() or prompt[:200]]}

        # Query-rewrite expansion — return only the original; the dummy is
        # not in a position to invent meaningful paraphrases.
        if "queries" in p and ("expand" in p or "phrasing" in p or "rewrite" in p):
            return {"queries": [question_only.strip() or prompt[:200]]}

        # Verifier
        if "grounded" in p or "verify" in p:
            return {"grounded": True, "missing": []}

        # Generic fallback — return empty structure based on schema
        return _empty_from_schema(schema)


def _empty_from_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Construct an empty-but-valid object conforming to a minimal JSON schema."""
    if schema.get("type") != "object":
        return {}
    out: dict[str, Any] = {}
    for k, v in (schema.get("properties") or {}).items():
        t = v.get("type")
        if t == "string":
            out[k] = ""
        elif t == "number" or t == "integer":
            out[k] = 0
        elif t == "boolean":
            out[k] = False
        elif t == "array":
            out[k] = []
        elif t == "object":
            out[k] = _empty_from_schema(v)
    return out


# --- factory ---------------------------------------------------------------

def make_llm() -> LLM:
    """Construct the configured LLM backend, falling back to Dummy on error.

    When ``settings.llm_backend == "dummy"``, always returns :class:`DummyLLM`.
    Otherwise builds an :class:`OllamaLLM` and performs a cheap ``/api/tags``
    handshake; on any exception (server down, wrong port, bad model name) we
    transparently fall back to :class:`DummyLLM` so callers never have to
    branch on backend availability. Trace output and report headers reveal
    which backend was actually used.
    """
    if settings.llm_backend == "dummy":
        return DummyLLM()
    try:
        llm = OllamaLLM(settings.ollama_host, settings.ollama_model)
        # Cheap handshake to fail fast if Ollama is down.
        llm._client.list()
        return llm
    except Exception:
        return DummyLLM()
