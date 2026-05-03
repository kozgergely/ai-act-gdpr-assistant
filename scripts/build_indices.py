"""CLI: run ingestion, then build the vector store + citation graph."""

from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console

# Allow running without installing the package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag.ingest.run import ingest_all  # noqa: E402
from agentic_rag.rag.store import (  # noqa: E402
    build_citation_graph,
    build_vector_index,
    save_graph,
)

console = Console()


def main() -> int:
    console.rule("[bold]ingest[/]")
    chunks = ingest_all()
    if not chunks:
        console.print("[red]no chunks — cannot build indices[/]")
        return 1

    console.rule("[bold]vector index[/]")
    n = build_vector_index(chunks)
    console.print(f"[green]indexed[/] {n} chunks into Chroma")

    console.rule("[bold]citation graph[/]")
    g = build_citation_graph(chunks)
    save_graph(g)
    console.print(
        f"[green]graph[/] nodes={g.number_of_nodes()} edges={g.number_of_edges()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
