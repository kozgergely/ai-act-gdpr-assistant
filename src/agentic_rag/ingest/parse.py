"""Structure-aware parser for EUR-Lex regulation PDFs.

Rather than splitting on a fixed character window, the parser walks the
EUR-Lex skeleton — articles, recitals, annex references, and the
cross-references between them — and emits one :class:`Chunk` per structural
unit. Articles longer than :data:`MAX_CHUNK_CHARS` are soft-split with a
small overlap so that they still fit a vector-store record while preserving
the article-level identity in the chunk id (e.g. ``5``, ``9.1``, ``9.2``).

The downstream consumers are:

* :mod:`agentic_rag.rag.store`, which builds the Chroma vector index and the
  NetworkX citation graph from the same chunk stream.
* The fixture loader in :mod:`agentic_rag.ingest.run`, which uses the same
  :class:`Chunk` shape for the hand-written smoke fixture.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

try:
    import pymupdf  # preferred — clean text extraction on EUR-Lex PDFs
    _HAS_PYMUPDF = True
except ImportError:
    _HAS_PYMUPDF = False

from pypdf import PdfReader

# --- regex patterns ----------------------------------------------------------

# Article headers. EUR-Lex format is typically "Article 5" on its own line,
# sometimes followed by a descriptive line like "Prohibited AI practices".
ARTICLE_HDR = re.compile(r"^\s*Article\s+(\d+)\s*$", re.MULTILINE)

# Recital markers appear near the document preamble: "(1) ... (2) ..."
RECITAL_HDR = re.compile(r"(?m)^\((\d+)\)\s+")

# Numbered paragraphs inside an article body — "1.", "2.", ... at line start.
# (Not to be confused with lettered points "(a)", "(b)" which are sub-points
# beneath a numbered paragraph.)
PARAGRAPH_HDR = re.compile(r"(?m)^\s*(\d{1,3})\.\s")

# Cross-references inside the body of a paragraph.
# Matches: "Article 5", "Article 5(1)", "Article 5(1)(a)", "Articles 5 and 6",
#          "Recital 26", "Annex III", "Chapter II".
XREF_ARTICLE = re.compile(
    r"\bArticles?\s+(\d+(?:\s*\([0-9a-z]+\))*)"
    r"(?:\s*(?:and|,|\bor\b)\s*(\d+(?:\s*\([0-9a-z]+\))*))*",
    re.IGNORECASE,
)
XREF_RECITAL = re.compile(r"\bRecitals?\s+(\d+)", re.IGNORECASE)
XREF_ANNEX = re.compile(r"\bAnnex\s+([IVXLCDM]+)\b", re.IGNORECASE)


@dataclass
class Chunk:
    """A single embedded/indexed unit produced by the ingestion pipeline.

    Attributes:
        id: Globally unique chunk id, ``"{regulation}:{kind}:{number}"``. Sub-
            chunks of a long article get a ``.N`` suffix on ``number``.
        regulation: Source corpus, currently ``"AI Act"`` or ``"GDPR"``.
        kind: Structural type, ``"article"`` or ``"recital"``.
        number: Article or recital number; may include a sub-chunk suffix.
        title: Descriptive heading detected immediately after the article
            header. ``None`` when no heading was present.
        text: Plain text body of the chunk.
        page: Page number in the original PDF where this chunk starts.
        cross_refs: Article-level cross-references found in the text,
            normalized to ``"<kind>:<number>"`` (e.g. ``"article:9"``).
            These become directed edges in the citation graph.
        paragraphs: Numbered paragraph headers detected inside the chunk
            (e.g. ``["1", "2", "3"]``). Empty for recitals and very short
            articles. Used only as metadata — chunking stays at article level.
        cross_refs_detailed: Cross-references with paragraph/letter
            suffixes preserved verbatim (e.g. ``"Article 5(1)(a)"``). Used by
            the composer for human-readable citations without changing the
            graph granularity.
    """

    id: str
    regulation: str
    kind: str
    number: str
    title: str | None
    text: str
    page: int
    cross_refs: list[str] = field(default_factory=list)
    paragraphs: list[str] = field(default_factory=list)
    cross_refs_detailed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict for the JSONL chunk file."""
        return {
            "id": self.id,
            "regulation": self.regulation,
            "kind": self.kind,
            "number": self.number,
            "title": self.title,
            "text": self.text,
            "page": self.page,
            "cross_refs": self.cross_refs,
            "paragraphs": self.paragraphs,
            "cross_refs_detailed": self.cross_refs_detailed,
        }


# EUR-Lex PDFs put a publication footer at the bottom of every page:
#
#     ELI: http://data.europa.eu/eli/reg/.../oj
#     N/M
#     (1) OJ C ... (footnote citations)
#     (2) OJ ...
#
# The "(N) OJ ..." footnote markers collide with the recital "(N) ..." syntax
# and would otherwise produce duplicate recital ids. We strip the footer
# (everything from the ELI marker to the end of the page) before joining
# pages into a single document.
_PAGE_FOOTER_RE = re.compile(
    r"ELI:\s+https?://data\.europa\.eu[\s\S]*",
    re.IGNORECASE,
)


def extract_text(pdf_path: Path) -> list[tuple[int, str]]:
    """Read every page of ``pdf_path`` and return ``(page_number, text)`` pairs.

    Prefers PyMuPDF (``pymupdf``) over pypdf because EUR-Lex PDFs use a font
    setup that pypdf misinterprets — words come out fragmented (``"Ar ticle"``
    instead of ``"Article"``), which breaks every regex in this module.
    PyMuPDF extracts the text cleanly and is also faster.

    Each page is also stripped of its EUR-Lex publication footer (ELI line +
    OJ-style footnotes) so that the recital-marker regex does not match the
    footnote numbering. Falls back to pypdf if PyMuPDF is unavailable. Page
    numbers are 1-indexed to match the citations users see in the PDF reader.
    """
    out: list[tuple[int, str]] = []

    def _clean(txt: str) -> str:
        return _PAGE_FOOTER_RE.sub("", txt).rstrip() + "\n"

    if _HAS_PYMUPDF:
        doc = pymupdf.open(str(pdf_path))
        try:
            for i, page in enumerate(doc, start=1):
                try:
                    txt = page.get_text() or ""
                except Exception:
                    txt = ""
                out.append((i, _clean(txt)))
        finally:
            doc.close()
        return out

    reader = PdfReader(str(pdf_path))
    for i, page in enumerate(reader.pages, start=1):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        out.append((i, _clean(txt)))
    return out


def find_cross_refs(text: str) -> list[str]:
    """Normalize cross-references found in a text blob.

    Each ref is represented as ``"<kind>:<number>"`` — the citation graph uses
    these verbatim as node identifiers.
    """
    refs: set[str] = set()
    for m in XREF_ARTICLE.finditer(text):
        # Only capture the top-level article number; paragraph/letter suffix is
        # dropped so that graph edges are at article granularity.
        for g in m.groups():
            if not g:
                continue
            n = g.strip().split("(")[0].strip()
            if n.isdigit():
                refs.add(f"article:{n}")
    for m in XREF_RECITAL.finditer(text):
        refs.add(f"recital:{m.group(1)}")
    for m in XREF_ANNEX.finditer(text):
        refs.add(f"annex:{m.group(1).upper()}")
    return sorted(refs)


def find_cross_refs_detailed(text: str) -> list[str]:
    """Cross-references with paragraph/letter suffixes preserved verbatim.

    Returns strings like ``"Article 5(1)(a)"``, ``"Article 9"``,
    ``"Recital 26"``, ``"Annex III"``. Used for citation pretty-printing in
    the composer header — the graph still operates at article granularity.
    """
    refs: list[str] = []
    seen: set[str] = set()

    def _add(s: str) -> None:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            refs.append(s)

    for m in XREF_ARTICLE.finditer(text):
        for g in m.groups():
            if not g:
                continue
            # Normalize whitespace inside parens: "5(1) (a)" -> "5(1)(a)".
            cleaned = re.sub(r"\s*\(\s*", "(", g)
            cleaned = re.sub(r"\s*\)\s*", ")", cleaned).strip()
            if cleaned and cleaned[0].isdigit():
                _add(f"Article {cleaned}")
    for m in XREF_RECITAL.finditer(text):
        _add(f"Recital {m.group(1)}")
    for m in XREF_ANNEX.finditer(text):
        _add(f"Annex {m.group(1).upper()}")
    return refs


def find_paragraphs(text: str) -> list[str]:
    """Numbered paragraph headers found inside an article body.

    Returns the paragraph numbers in order of appearance, e.g. ["1", "2", "3"].
    Empty when the article uses no numbered paragraphs (typical for short
    articles or for hand-written fixture chunks).
    """
    out: list[str] = []
    seen: set[str] = set()
    for m in PARAGRAPH_HDR.finditer(text):
        n = m.group(1)
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _collect_full_text(pages: list[tuple[int, str]]) -> tuple[str, dict[int, int]]:
    """Concatenate page texts and record each page's start offset.

    Returns:
        A tuple of (joined text, offset → page mapping). The mapping lets us
        recover the original page number for any character position in the
        joined string via :func:`_page_at`.
    """
    offset_to_page: dict[int, int] = {}
    parts: list[str] = []
    running = 0
    for page_num, text in pages:
        offset_to_page[running] = page_num
        parts.append(text)
        running += len(text) + 1  # +1 for the newline we add between pages
    return "\n".join(parts), offset_to_page


def _page_at(offset: int, offset_to_page: dict[int, int]) -> int:
    """Return the 1-indexed page number containing ``offset`` in the joined text."""
    page = 1
    for start, p in offset_to_page.items():
        if start <= offset:
            page = p
        else:
            break
    return page


def _split_articles(full_text: str) -> list[tuple[str, int, str]]:
    """Split the document at every ``Article N`` header.

    Returns:
        A list of ``(article_number, start_offset, body)`` tuples in document
        order. The body extends until the next article header or, for the
        last article, the end of the document.
    """
    matches = list(ARTICLE_HDR.finditer(full_text))
    out: list[tuple[str, int, str]] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        body = full_text[start:end].strip()
        out.append((m.group(1), m.start(), body))
    return out


_WHEREAS_HDR = re.compile(r"^\s*Whereas\s*:\s*$", re.IGNORECASE | re.MULTILINE)
_HAS_ADOPTED_HDR = re.compile(
    r"^\s*HA(?:S|VE)\s+ADOPTED\s+THIS\s+REGULATION\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _split_recitals(full_text: str) -> list[tuple[str, int, str]]:
    """Extract the numbered ``(N)`` recital blocks from the document preamble.

    EUR-Lex preambles wrap the recitals between two header lines:

    * ``Whereas:`` — opens the recital section.
    * ``HAS ADOPTED THIS REGULATION:`` (or the plural ``HAVE ADOPTED ...``)
      — closes it; articles begin immediately after.

    Restricting the search to that explicit window prevents false positives
    from footnote-style markers like ``(1) OJ C 517, 22.12.2021, p. 56`` that
    also start with ``(N)`` at a line boundary but live in the title-page
    metadata block.
    """
    whereas = _WHEREAS_HDR.search(full_text)
    if whereas is None:
        # Fallback: original heuristic — everything before the first article.
        art1 = ARTICLE_HDR.search(full_text)
        recital_zone_start = 0
        recital_zone_end = art1.start() if art1 else len(full_text)
    else:
        recital_zone_start = whereas.end()
        end_marker = _HAS_ADOPTED_HDR.search(full_text, recital_zone_start)
        if end_marker is not None:
            recital_zone_end = end_marker.start()
        else:
            art1 = ARTICLE_HDR.search(full_text, recital_zone_start)
            recital_zone_end = art1.start() if art1 else len(full_text)

    zone = full_text[recital_zone_start:recital_zone_end]
    matches = list(RECITAL_HDR.finditer(zone))
    out: list[tuple[str, int, str]] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(zone)
        body = zone[start:end].strip()
        # Absolute offset back into the full document for page lookup.
        absolute_offset = recital_zone_start + m.start()
        out.append((m.group(1), absolute_offset, body))
    return out


def _first_line_title(body: str) -> str | None:
    """Heuristically extract the descriptive article heading.

    EUR-Lex articles are normally followed by a short title on the next
    non-blank line (e.g. ``Article 5\\nProhibited AI practices``). We accept
    any 3–120 character non-sentence line; longer or sentence-shaped lines
    are taken as body text and yield ``None``.
    """
    for line in body.splitlines():
        line = line.strip()
        if line:
            # Skip lines that look like the body's first sentence.
            if 3 <= len(line) <= 120 and not line.endswith("."):
                return line
            return None
    return None


MAX_CHUNK_CHARS = 2200
"""Soft cap on a single chunk's body length before sub-splitting kicks in."""
OVERLAP = 200
"""Character overlap between consecutive sub-chunks of a long article."""


def _soft_split(body: str) -> list[str]:
    """Split an over-long article body into overlapping character windows.

    Returns the original body unmodified when it fits in
    :data:`MAX_CHUNK_CHARS`. Otherwise produces consecutive windows of size
    :data:`MAX_CHUNK_CHARS` with :data:`OVERLAP` characters carried over so
    a sentence straddling a boundary still appears in both windows.
    """
    if len(body) <= MAX_CHUNK_CHARS:
        return [body]
    out: list[str] = []
    start = 0
    while start < len(body):
        end = min(start + MAX_CHUNK_CHARS, len(body))
        out.append(body[start:end])
        if end == len(body):
            break
        start = end - OVERLAP
    return out


def parse_regulation(pdf_path: Path, regulation: str) -> list[Chunk]:
    """Parse one regulation PDF end-to-end into a flat list of chunks.

    The output contains every recital from the preamble followed by every
    article. Long articles produce multiple sub-chunks with shared
    article-level identity (``5.1``, ``5.2``, ...). Cross-references and
    paragraph metadata are populated for each chunk so the downstream
    indices can be built without re-scanning the PDF.

    Args:
        pdf_path: Path to the EUR-Lex PDF on disk.
        regulation: Human label used as the regulation prefix in chunk ids
            (typically ``"AI Act"`` or ``"GDPR"``).

    Returns:
        Chunks in document order: recitals first, then articles.
    """
    pages = extract_text(pdf_path)
    full, offsets = _collect_full_text(pages)

    chunks: list[Chunk] = []

    for num, off, body in _split_recitals(full):
        page = _page_at(off, offsets)
        chunks.append(
            Chunk(
                id=f"{regulation}:recital:{num}",
                regulation=regulation,
                kind="recital",
                number=num,
                title=None,
                text=body,
                page=page,
                cross_refs=find_cross_refs(body),
                cross_refs_detailed=find_cross_refs_detailed(body),
            )
        )

    for num, off, body in _split_articles(full):
        page = _page_at(off, offsets)
        title = _first_line_title(body)
        pieces = _soft_split(body)
        for j, piece in enumerate(pieces):
            suffix = "" if len(pieces) == 1 else f".{j + 1}"
            chunks.append(
                Chunk(
                    id=f"{regulation}:article:{num}{suffix}",
                    regulation=regulation,
                    kind="article",
                    number=f"{num}{suffix}",
                    title=title,
                    text=piece,
                    page=page,
                    cross_refs=find_cross_refs(piece),
                    paragraphs=find_paragraphs(piece),
                    cross_refs_detailed=find_cross_refs_detailed(piece),
                )
            )

    return chunks
