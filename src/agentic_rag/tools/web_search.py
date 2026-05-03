"""Web search tool — DuckDuckGo (no API key required).

Used for questions about recent guidance, enforcement news, or case law that
post-dates the static corpus. The tool is allowed to fail gracefully — an
empty list is fine for the composer node.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SearchHit:
    """One search result returned by :py:func:`web_search`.

    Attributes:
        title: Page title as reported by DuckDuckGo.
        url: Result URL.
        snippet: Short text excerpt (DuckDuckGo's "body" field).
    """

    title: str
    url: str
    snippet: str

    def to_dict(self) -> dict:
        """Serialize to a plain JSON-safe dict for the agent state."""
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


def web_search(query: str, max_results: int = 5) -> list[SearchHit]:
    """Run a DuckDuckGo text search and return up to ``max_results`` hits.

    The function never raises: a missing ``duckduckgo_search`` dependency,
    a network error, or any other failure produces an empty list. The
    composer prompt is happy to render an empty tool result, so a degraded
    web-search experience does not break the agent flow.

    Args:
        query: Free-form search query (the agent typically forwards the
            user's original question verbatim).
        max_results: Hard cap on the number of hits to return.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return []

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception:
        return []

    return [
        SearchHit(
            title=r.get("title") or "",
            url=r.get("href") or "",
            snippet=r.get("body") or "",
        )
        for r in results
    ]
