"""Vector store (Chroma) + citation graph (NetworkX) construction and loading.

The two indices share a common chunk schema so the graph node ids and the
Chroma chunk ids can be joined directly during graph expansion. Both indices
are file-backed and rebuilt from scratch by ``scripts/build_indices.py``;
production deployments would persist them externally instead.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Iterable

import chromadb
import networkx as nx
from chromadb.config import Settings as ChromaSettings

from agentic_rag.config import settings
from agentic_rag.ingest.parse import Chunk

COLLECTION = "regulations"
"""Name of the single Chroma collection holding every chunk."""


# --- helpers ----------------------------------------------------------------

def _node_id(regulation: str, kind: str, number: str) -> str:
    """Compute the article-level graph node id for a chunk.

    Sub-chunks of a long article (e.g. ``9.1``, ``9.2``) collapse to a single
    graph node identified by the top-level article number. The vector store
    keeps the sub-chunks distinct via their full chunk id.
    """
    base = number.split(".")[0]
    return f"{regulation}:{kind}:{base}"


def load_chunks(path: Path | None = None) -> list[Chunk]:
    """Read the JSONL chunk file produced by the ingestion pipeline.

    Args:
        path: optional override; falls back to ``settings.processed_path``.

    Returns:
        Parsed :class:`Chunk` instances in the order they were written.
    """
    p = path or settings.processed_path
    with p.open() as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]


# --- embedding function -----------------------------------------------------

class _SentenceTransformerEmbedding:
    """Chroma-compatible embedding function backed by ``sentence-transformers``.

    Chroma's API has historically called embedding objects in two distinct
    ways: ``__call__`` for the indexing path and ``embed_query`` for the
    query path. We implement both so the same instance works for both
    ``col.add`` and ``col.query``. ``normalize_embeddings=True`` is required
    by the BGE family and lets Chroma's HNSW index treat cosine similarity
    as a simple dot product.
    """

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self._name = model_name

    def _encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    # Chroma's indexing path calls the function directly.
    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002 (chroma API)
        return self._encode(input)

    # Chroma's query path calls embed_query explicitly.
    def embed_query(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        return self._encode(input)

    def name(self) -> str:  # chroma 0.5 requires this
        return self._name


def _client() -> chromadb.api.ClientAPI:
    """Open a Chroma persistent client at ``settings.chroma_dir``."""
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(settings.chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_embedder():
    """Construct a fresh embedding-function instance from current settings."""
    return _SentenceTransformerEmbedding(settings.embedding_model)


# --- build ------------------------------------------------------------------

def build_vector_index(chunks: Iterable[Chunk], batch_size: int = 64) -> int:
    """(Re)build the Chroma collection from a stream of chunks.

    The collection is dropped and recreated so callers do not have to manage
    incremental updates. Documents are stored with a one-line provenance
    header prepended (used by retrieval traces and the composer prompt) and
    every chunk attribute that retrieval might filter on is mirrored into
    the metadata dict.

    Args:
        chunks: iterable of :class:`Chunk`. Order is preserved within batches
            but does not affect retrieval (HNSW is order-agnostic).
        batch_size: number of chunks added per ``col.add`` call.

    Returns:
        Total chunks indexed.
    """
    client = _client()
    embedder = get_embedder()
    # Fresh build each time — cheap at this corpus size.
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(
        name=COLLECTION,
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )

    batch_ids: list[str] = []
    batch_docs: list[str] = []
    batch_meta: list[dict] = []
    count = 0

    def flush() -> None:
        nonlocal count
        if not batch_ids:
            return
        col.add(ids=batch_ids, documents=batch_docs, metadatas=batch_meta)
        count += len(batch_ids)
        batch_ids.clear()
        batch_docs.clear()
        batch_meta.clear()

    for c in chunks:
        header = f"[{c.regulation} {c.kind.capitalize()} {c.number}"
        if c.title:
            header += f" — {c.title}"
        header += "]\n"
        batch_ids.append(c.id)
        batch_docs.append(header + c.text)
        batch_meta.append(
            {
                "regulation": c.regulation,
                "kind": c.kind,
                "number": c.number,
                "title": c.title or "",
                "page": c.page,
                "cross_refs": ",".join(c.cross_refs),
                "paragraphs": ",".join(c.paragraphs),
                "cross_refs_detailed": ",".join(c.cross_refs_detailed),
            }
        )
        if len(batch_ids) >= batch_size:
            flush()
    flush()
    return count


def build_citation_graph(chunks: Iterable[Chunk]) -> nx.DiGraph:
    """Build the directed citation graph from a stream of chunks.

    Nodes are keyed at article granularity (one node per
    ``(regulation, kind, top-level-number)``). Every cross-reference parsed
    out of a chunk's text becomes a directed edge from the chunk's article
    to the referenced article. References to articles missing from the
    corpus still create dangling target nodes — this surfaces coverage gaps
    instead of silently dropping them.

    Cross-regulation edges are not produced today: if AI Act text mentions
    "Article 9 of GDPR", the parser captures ``article:9`` but we resolve it
    inside the source regulation. This is documented as a known limitation
    in DEVLOG §4.
    """
    g: nx.DiGraph = nx.DiGraph()

    # Chunks may split an article across multiple rows; collapse to one node
    # per (regulation, kind, article-number).
    for c in chunks:
        nid = _node_id(c.regulation, c.kind, c.number)
        if nid not in g:
            g.add_node(
                nid,
                regulation=c.regulation,
                kind=c.kind,
                number=c.number.split(".")[0],
                title=c.title or "",
                chunk_ids=[c.id],
            )
        else:
            g.nodes[nid]["chunk_ids"].append(c.id)

    # Cross-references resolve within the same regulation. Foreign references
    # (e.g. GDPR text referencing another act) stay dangling by design.
    for c in chunks:
        src = _node_id(c.regulation, c.kind, c.number)
        for ref in c.cross_refs:
            kind, _, num = ref.partition(":")
            if not num:
                continue
            tgt = f"{c.regulation}:{kind}:{num}"
            if tgt == src:
                continue
            # Add the target node lazily if the referenced article is missing
            # from the corpus — this keeps the graph a superset of what we
            # could retrieve and surfaces coverage gaps.
            if tgt not in g:
                g.add_node(tgt, regulation=c.regulation, kind=kind, number=num, chunk_ids=[])
            g.add_edge(src, tgt)
    return g


def save_graph(g: nx.DiGraph, path: Path | None = None) -> None:
    """Persist the citation graph as pickle + GraphML.

    Pickle is used at runtime (fast load, preserves Python types). GraphML
    is emitted alongside for ad-hoc visual inspection in Gephi / yEd; node
    attributes that hold lists are flattened to comma-separated strings
    because GraphML cannot represent collections natively.
    """
    p = path or settings.graph_path
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(g, f)
    # Also emit GraphML for ad-hoc inspection in Gephi / yEd.
    gml_path = p.with_suffix(".graphml")
    graphml_copy = g.copy()
    for _, data in graphml_copy.nodes(data=True):
        if "chunk_ids" in data:
            data["chunk_ids"] = ",".join(data["chunk_ids"])
    nx.write_graphml(graphml_copy, gml_path)


def load_graph(path: Path | None = None) -> nx.DiGraph:
    """Read the pickled citation graph from disk."""
    p = path or settings.graph_path
    with p.open("rb") as f:
        return pickle.load(f)


def get_collection():
    """Open the Chroma collection bound to the current embedding function."""
    return _client().get_collection(name=COLLECTION, embedding_function=get_embedder())
