"""Unit tests for the regulation parser — focused on cross-reference extraction."""

from __future__ import annotations

from agentic_rag.ingest.parse import (
    find_cross_refs,
    find_cross_refs_detailed,
    find_paragraphs,
)


def test_article_simple():
    assert "article:9" in find_cross_refs("See Article 9 for details.")


def test_article_with_paragraph():
    # Paragraph/letter suffixes collapse to the article number.
    refs = find_cross_refs("under Article 6(1)(a) of this Regulation")
    assert "article:6" in refs


def test_plural_articles():
    refs = find_cross_refs("as provided in Articles 8 and 10")
    assert "article:8" in refs
    assert "article:10" in refs


def test_recital():
    assert "recital:26" in find_cross_refs("referred to in Recital 26")


def test_annex_roman():
    assert "annex:III" in find_cross_refs("Annex III systems")


def test_no_false_positives():
    # Standalone numbers or non-matching words should not produce edges.
    assert find_cross_refs("the year 2024 is important") == []


# --- detailed cross-references (paragraph-level preserved) -----------------


def test_detailed_keeps_paragraph_letter():
    refs = find_cross_refs_detailed("under Article 6(1)(a) of this Regulation")
    assert "Article 6(1)(a)" in refs


def test_detailed_strips_inner_whitespace():
    refs = find_cross_refs_detailed("see Article 5 (1) (a) above")
    assert "Article 5(1)(a)" in refs


def test_detailed_recital_and_annex():
    text = "as set out in Recital 26 and described in Annex III"
    refs = find_cross_refs_detailed(text)
    assert "Recital 26" in refs
    assert "Annex III" in refs


def test_detailed_dedupes_case_insensitively():
    refs = find_cross_refs_detailed("Article 5 references article 5 again")
    assert refs.count("Article 5") == 1


# --- paragraph detection ----------------------------------------------------


def test_paragraphs_simple():
    body = "1. The first paragraph says foo.\n2. The second one says bar.\n"
    assert find_paragraphs(body) == ["1", "2"]


def test_paragraphs_skip_lettered_points():
    body = (
        "1. The following are prohibited:\n"
        "(a) one thing\n"
        "(b) another thing\n"
        "2. This Article shall not apply to ...\n"
    )
    # Only numbered paragraphs, not lettered points.
    assert find_paragraphs(body) == ["1", "2"]


def test_paragraphs_dedupe():
    body = "1. First.\n1. (re-mention)\n2. Second.\n"
    assert find_paragraphs(body) == ["1", "2"]


def test_paragraphs_none_in_short_article():
    # Short articles may have no numbered paragraphs.
    body = "This Regulation shall enter into force on the day after publication."
    assert find_paragraphs(body) == []
