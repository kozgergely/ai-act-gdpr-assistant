"""End-to-end ingestion entrypoint: PDFs → canonical JSONL chunk file.

Driver script that wires the PDF parser to the on-disk artefacts. Two
behaviours are supported:

* When the EUR-Lex PDFs are present in ``data/raw/``, parse each one and
  write the resulting chunks to ``data/processed/chunks.jsonl``.
* When no PDFs are available (sandboxed environments, fresh checkouts),
  fall back to the hand-crafted ``data/raw/fixture.jsonl`` so the rest of
  the pipeline still has a usable corpus to build indices from.

Run as ``python -m agentic_rag.ingest.run`` or ``python scripts/build_indices.py``
(which calls :py:func:`ingest_all` and continues to index building).
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from agentic_rag.config import settings
from agentic_rag.ingest.parse import Chunk, parse_regulation

console = Console()

SOURCES: dict[str, list[str]] = {
    # Accept either the simplified name (`ai_act.pdf`) or the official EUR-Lex
    # OJ filename (which is what the user gets when downloading directly from
    # the EUR-Lex web UI). The first match wins.
    "AI Act": ["ai_act.pdf", "OJ_L_202401689_EN_TXT.pdf"],
    "GDPR":   ["gdpr.pdf",   "OJ_L_2016_119_EN_TXT.pdf"],
}
"""Per-regulation list of accepted PDF filenames under ``data/raw/``."""


def _find_source(raw_dir: Path, candidates: list[str]) -> Path | None:
    """Return the first existing candidate path, or ``None`` if none match."""
    for fname in candidates:
        p = raw_dir / fname
        if p.exists():
            return p
    return None


def ingest_all() -> list[Chunk]:
    """Parse every available PDF (or fall back to the fixture) into chunks.

    Side effect: writes ``settings.processed_path`` (a JSONL file) so
    downstream index builders can stream chunks without re-running the
    parser. The output is always rewritten — incremental updates are out
    of scope for the prototype.

    Returns:
        Every chunk written to disk, in the same order as the JSONL.
    """
    raw_dir = settings.data_dir / "raw"
    chunks: list[Chunk] = []
    for regulation, candidates in SOURCES.items():
        path = _find_source(raw_dir, candidates)
        if path is None:
            console.print(
                f"[yellow]missing[/] {regulation} PDF "
                f"(looked for: {', '.join(candidates)})"
            )
            continue
        console.print(f"[bold]→[/] parsing {regulation} ({path.name})")
        regulation_chunks = parse_regulation(path, regulation)
        console.print(f"   [green]{len(regulation_chunks)}[/] chunks")
        chunks.extend(regulation_chunks)

    # Fallback: ship a hand-crafted fixture so pipeline is testable even before
    # the user has downloaded the official PDFs.
    if not chunks:
        fixture = raw_dir / "fixture.jsonl"
        if fixture.exists():
            console.print(f"[cyan]using fixture[/] {fixture}")
            for line in fixture.read_text().splitlines():
                if line.strip():
                    chunks.append(Chunk(**json.loads(line)))

    out_path = settings.processed_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    console.print(f"[green]wrote[/] {out_path} ({len(chunks)} chunks)")
    return chunks


if __name__ == "__main__":
    ingest_all()
