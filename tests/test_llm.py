"""Smoke tests for the LLM layer — focused on the Dummy backend semantics."""

from __future__ import annotations

from agentic_rag.llm.base import DummyLLM, Message


def test_dummy_chat_trivial():
    llm = DummyLLM()
    out = llm.chat([Message("user", "Hello?")])
    assert "dummy" in out.lower()


def test_dummy_router():
    llm = DummyLLM()
    out = llm.generate_json(
        "Classify the intent of the query: What are prohibited AI practices?",
        {"type": "object", "properties": {"intent": {"type": "string"}}},
    )
    assert out["intent"] in {"rag", "tool", "direct"}


def test_dummy_deadline_goes_to_tool():
    llm = DummyLLM()
    out = llm.generate_json(
        "Classify the intent: how many months until the AI Act deadline?",
        {"type": "object", "properties": {"intent": {"type": "string"}}},
    )
    assert out["intent"] == "tool"


def test_dummy_verifier_grounded_true():
    llm = DummyLLM()
    out = llm.generate_json(
        "Verify that the draft answer is grounded in the provided context.",
        {
            "type": "object",
            "properties": {
                "grounded": {"type": "boolean"},
                "missing": {"type": "array"},
            },
        },
    )
    assert out["grounded"] is True
