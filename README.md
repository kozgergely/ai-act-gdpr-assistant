# Agentic RAG — EU AI Act + GDPR Compliance Assistant

Senior AI Engineer take-home prototype. An **agentic RAG chatbot** built with LangGraph, a **hybrid vector + citation-graph retrieval** subsystem, and a fully open-source / local-first LLM stack (Ollama + `qwen2.5:7b-instruct`, with a deterministic `DummyLLM` fallback). Packaged with Docker + docker-compose.

- Development log (chronological steps, design decisions, alternatives — *Hungarian*): **[`DEVLOG.md`](./DEVLOG.md)**
- Post-prototype roadmap (Phase 2 / 3 with GCP + Azure mappings — *English*): **[`ROADMAP.md`](./ROADMAP.md)**

---

## 1. Problem statement

EU companies face two major, interlocking regulatory frameworks: the **GDPR** (Regulation 2016/679, in force since 2018) and the **AI Act** (Regulation 2024/1689, entering application in staged phases between 2 Feb 2025 and 2 Aug 2027). Compliance teams need fast, citation-grounded answers to questions like:

- *"Is this use case prohibited under Article 5 of the AI Act?"*
- *"When does my general-purpose AI model with systemic risk need to comply?"*
- *"Does our automated decisioning pipeline trigger both GDPR Article 22 **and** AI Act Article 6?"*

### Why agentic RAG (and not plain RAG)?

Legal texts are **dense, precise, and heavily cross-referenced**. A single question like *"What are the risk management requirements for high-risk AI systems?"* requires Article 9 (risk management), Article 6 (which systems count as high-risk), Article 10 (data governance), and Annex III (use-case list). A plain semantic search typically surfaces one or two of these; the rest arrive only via the **explicit citation graph** embedded in the text ("Testing shall be conducted in accordance with Article 10"). The agentic layer adds:

- **Router**: classifies queries as retrieval / tool / direct — a *"when does this apply?"* question doesn't need RAG at all, it needs a date tool.
- **Planner**: decomposes multi-part questions into sub-queries.
- **Verifier**: checks the composed answer is grounded in retrieved context; re-queries if not.

### User need

Compliance officers, legal counsel, engineering leads, and auditors who need (a) a **fast first pass** over the regulations, (b) **clickable citations** they can verify, (c) explicit acknowledgment of **when the answer is speculative or out-of-corpus** (that's why the verifier matters).

---

## 2. Architecture

```
                      ┌──────────────────────────────────────┐
                      │             Streamlit UI             │
                      │  (chat left, agent trace right)      │
                      └──────────────────┬───────────────────┘
                                         │
                                  ┌──────▼───────┐
                                  │   router     │  (LLM, conditional routing)
                                  └──┬─────┬─────┴──┐
                        intent=rag   │     │ tool   │ direct
                                     │     ▼        ▼
                             ┌───────▼──┐  ┌──────────────┐
                             │ planner  │  │tool_executor │  → web_search / deadline_calc
                             └───┬──────┘  └──────┬───────┘
                                 │                │
                             ┌───▼───────────┐    │
                             │ rag_invoker   │    │
                             │  (loop over   │    │
                             │  subquestions)│    │
                             └───┬───────────┘    │
                                 │                │
                      ┌──────────▼────────────┐   │
                      │    RAG subgraph       │   │
                      │ ┌──────────────────┐  │   │
                      │ │ query_rewrite    │  │   │
                      │ │ vector_retrieve  │  │   │
                      │ │ graph_expand     │  │   │
                      │ │ fuse_rerank      │  │   │
                      │ │ context_assemble │  │   │
                      │ └──────────────────┘  │   │
                      └──────────┬────────────┘   │
                                 │                │
                               ┌─▼────────────────▼┐
                               │     composer      │  (LLM, citations + disclaimer)
                               └────────┬──────────┘
                                        │
                                  ┌─────▼────┐  retry ──┐
                                  │ verifier │──────────┘
                                  └─────┬────┘
                                        │ grounded
                                        ▼
                                       END
```

### Main graph — 6 nodes (requirement: ≥5)

| Node | Type | Responsibility |
|---|---|---|
| `router` | LLM (structured output) | Classify intent → `rag` / `tool` / `direct`; choose tool if applicable |
| `planner` | LLM (structured output) | Decompose the question into 1–3 sub-questions |
| `rag_invoker` | Subgraph invocation | Call the RAG subgraph once per sub-question, accumulate context |
| `tool_executor` | Deterministic | Run `web_search` (DuckDuckGo) or `deadline_calc` |
| `composer` | LLM | Write the final answer with inline citations + "not legal advice" disclaimer |
| `verifier` | LLM (structured output) | Check grounding; loop back to `rag_invoker` on failure (max 1 retry) |

**Conditional routing** (autonomous decisions): `router` → 3-way; `verifier` → retry-or-end.
**State**: `TypedDict` carries query, intent, plan, retrieved docs, tool result, draft answer, verification, citations, trace.

### Tools (requirement: ≥2; ≥1 non-retrieval)

1. **RAG subgraph** — retrieval (see below).
2. **`deadline_calc`** — non-retrieval. Maps a question to AI Act Article 113 application phases (prohibitions / GPAI / high-risk / Annex I) and returns days/months remaining (or since application).
3. **`web_search`** — non-retrieval. DuckDuckGo, no API key. For recent guidance/enforcement news not yet in the static corpus.

### RAG subgraph — 5 nodes (dedicated subgraph, not counted in the main graph)

```
query_rewrite → vector_retrieve → graph_expand → fuse_rerank → context_assemble
```

- **`query_rewrite`** — multi-query expansion: the LLM produces up to 3 alternative phrasings of the user question (the original is always kept), each is then embedded separately. Bridges the gap between informal user phrasing and the formal legal vocabulary in the corpus. *(Not HyDE: we do not generate hypothetical answer documents — only paraphrases of the question.)*
- **`vector_retrieve`** — Chroma cosine search with `BAAI/bge-small-en-v1.5` embeddings (normalized).
- **`graph_expand`** — graph-based augmentation. From each top vector hit, walk one hop on the citation graph (in + out edges) to pull in directly cross-referenced articles. Example: a vector search for *"prohibited AI practices"* surfaces Article 5; this node then also brings in Article 9 because Article 5's text says *"See also Article 9 on risk management"*. Captures structural relationships that semantic search misses.
- **`fuse_rerank`** — merges the vector ranking and the graph ranking into one final list with Reciprocal Rank Fusion (k=60), deduplicated by chunk id, top-K=6. Both rankings contribute as peers so a graph-only hit can outrank a weak vector hit.
- **`context_assemble`** — pack selected chunks with provenance headers for the composer.

### Data ingestion

- **Download**: `scripts/download_data.py` attempts to fetch the AI Act + GDPR PDFs from EUR-Lex (may require running from a non-sandboxed IP due to CloudFront WAF; manual-download fallback is documented in the script).
- **Parse**: `pypdf` text extraction → regex-based structure detection (Article / Recital headers, cross-references). Long articles soft-split into 2200-char windows with 200-char overlap.
- **Cross-references**: regex extracts `Article N`, `Article N(M)(x)`, `Recital N`, `Annex X` → normalized to `{regulation}:{kind}:{number}` node ids.
- **Paragraph-aware metadata**: each article chunk also stores the list of numbered paragraphs (`1.`, `2.`, ...) found in its body and a detailed cross-reference list that preserves paragraph/letter suffixes (e.g. `"Article 5(1)(a)"`). The graph stays at article granularity; the composer header reads, e.g., `[AI Act Article 5 (paragraphs 1-3) — Prohibited AI practices, p.48]`.
- **Fixture**: `data/raw/fixture.jsonl` — 19 hand-crafted chunks (11 AI Act + 8 GDPR), 39 real cross-reference edges. Ensures the full pipeline runs end-to-end **without** the PDFs, for CI and smoke testing.

### Models and trade-offs

| Component | Choice | Rationale |
|---|---|---|
| LLM | `qwen2.5:7b-instruct` via Ollama | Best-in-class 7-8B for `format="json"` tool output; ~40–60 tok/s on Apple Silicon Metal |
| LLM fallback | `DummyLLM` (deterministic) | Enables CI, Docker smoke tests, offline dev without Ollama |
| Embeddings | `BAAI/bge-small-en-v1.5` (384d) | Best quality/latency for CPU; < 50 ms/query |
| Vector store | Chroma (persistent, file-backed) | No separate service; pip-install; metadata filtering |
| Graph store | NetworkX (in-process, pickled) | Lightweight; deterministic cross-ref graph needs no LLM |
| UI | Streamlit | Single-file prototype UI; trivial to run |

See [`DEVLOG.md`](./DEVLOG.md) (Hungarian) for full decision rationale including alternatives considered.

---

## 3. Evaluation results

### Functional eval — 15-question gold set

See [`eval/questions.yaml`](./eval/questions.yaml) for the full set. Three scoring layers (per D12 in [`DEVLOG.md`](./DEVLOG.md), Hungarian):

| Layer | Metric | DummyLLM (542 chunk) | Real LLM (qwen2.5:7b, 542 chunk) |
|---|---|---|---|
| Retrieval | Recall@K of expected chunks | 66.7% | **80.0%** |
| Citation | Precision (cited → retrieved) | 100% | **100%** (no hallucinated citations) |
| Routing | Intent accuracy | 100% | **93.3%** (1 miss on a deadline question) |
| Quality | LLM-as-judge (1–5) | n/a | **3.62 / 5** |
| Latency | Mean per query | ~42 ms | ~29 sec |

The real-LLM run shows the impact of the LLM-driven query rewrite (multi-query expansion): three paraphrases per question give the vector retriever broader coverage of legal phrasings, lifting recall by 13 percentage points over the deterministic baseline. Citation precision stays at 100% — the verifier loop catches any draft whose citations aren't grounded in retrieval. The single intent miss is on q06 (deadline question routed to RAG instead of `deadline_calc`); the final answer is still correct because retrieval surfaces Article 113, but the routing metric flags it.

**Graph contribution (ablation study).** Setting `GRAPH_HOPS=0` disables the citation-graph expansion entirely. Running the same eval suite under both LLMs, with and without the graph:

| Setup | Recall | Answer quality |
|---|---|---|
| DummyLLM, vector + graph | 60% | n/a |
| DummyLLM, vector only | **73.3%** | n/a |
| Real LLM, vector + graph | **80%** | **3.62** |
| Real LLM, vector only | **80%** | 3.53 |

Two findings:
1. With the deterministic Dummy LLM, the graph **hurts** recall (–13 pp) because graph hits displace relevant vector hits in the RRF fusion (a tunable RRF-weight issue queued for Phase 2).
2. With the real LLM and its query rewrite, the graph is **neutral on recall** but **lifts answer quality by ~0.1 LLM-judge points** — the graph functions as a context-enricher rather than a retrieval booster, giving the composer richer cross-references for synthesis.

**Per-question highlights (real-LLM run):**
- Pure factual (q01, q02, q07, q12, q13, q15): 100% recall.
- Multi-article cross-reference (q03, q08, q09, q11): 50% recall — one expected article surfaces, the other doesn't (graph expansion + RRF doesn't always pull both at once).
- **q10 (GDPR Article 22 ↔ AI Act Article 6)**: 0% recall — the citation graph currently has **no cross-regulation edges**. Documented limitation; two fix paths queued in [`DEVLOG.md §4`](./DEVLOG.md) (Hungarian).

Reproduce:

```bash
# Deterministic baseline (no LLM cost, ~5s for 15 queries):
LLM_BACKEND=dummy python eval/runner.py --no-judge

# Full quality pass with real LLM (~15-20 min for 15 queries on M1 Max):
LLM_BACKEND=ollama OLLAMA_MODEL=qwen2.5:7b-instruct python eval/runner.py
```

Reports land in `eval/reports/report_<timestamp>.md`.

### Load test — 100 queries

Ran on the dev machine (M1 Max 64GB) with `DummyLLM` to isolate pipeline overhead:

| Scenario | p50 (ms) | p95 | p99 | QPS | Errors |
|---|---|---|---|---|---|
| sequential (1 worker) | **46** | 54 | 74 | 23.3 | 0 |
| 5 workers | 200 | 210 | 215 | **30.2** | 0 |
| 10 workers | 461 | 771 | 830 | 23.4 | 0 |

**Bottleneck analysis:**

With a real LLM (`qwen2.5:7b` on Metal, ~1–2s per call), the **composer node** dominates end-to-end latency (router + composer + verifier = 3 LLM calls per query). The pipeline makes ~3 LLM calls × 1–2s = 3–6s per query. With `DummyLLM`, the bottleneck is the **sentence-transformer embedding forward pass** (~20–40 ms CPU) and Chroma HNSW lookup (~5 ms) — these run inside the `vector_retrieve` node.

At concurrency, the sweet spot is ~5 workers; beyond that, Chroma's per-collection lock and the shared embedding model cause contention and p99 blows up.

**Recommended optimizations** (full list in `load_test/reports/`):

1. **Cache the router decision** (normalized-query → intent) — repeated UI questions skip one LLM call.
2. **Compress the composer context** — cap to top-3 chunks after retrieval quality is validated; halves composer token count.
3. **Stream the composer output** and run the verifier overlapped — amortizes verification cost instead of adding it serially.
4. **Parallelize per-sub-question RAG calls** — when the planner decomposes, run sub-RAGs concurrently.

---

## 4. Setup and run

### Prerequisites

- **Python 3.11**
- **[`uv`](https://github.com/astral-sh/uv)** package manager (much faster than pip; `pip install -e .` also works as a fallback). On macOS: `brew install uv`. On Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- *Optional:* **[Ollama](https://ollama.com/)** for real-LLM answers. On macOS: `brew install --cask ollama-app`. The pipeline runs without it via the `DummyLLM` fallback.
- *Optional:* **Docker Desktop** for the containerized path.

### Quickstart — the simplest happy path (~5 min on M1 Max)

```bash
# 1. Clone and enter the repo
git clone <repo-url> pwc && cd pwc

# 2. Create the venv and install
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

# 3. Build indices from the shipped fixture (real PDFs optional, see below)
python scripts/build_indices.py

# 4. Run the UI with the deterministic Dummy LLM
LLM_BACKEND=dummy streamlit run src/agentic_rag/ui/app.py
# → open http://localhost:8501
```

This works fully offline and produces placeholder answers. To get real natural-language answers, swap the LLM (next section).

### Setup matrix — pick a row

| Where the app runs | LLM backend | When to use |
|---|---|---|
| Native Python | DummyLLM | Fastest setup; placeholder answers but full pipeline + UI works |
| Native Python | Host Ollama | Best demo experience on Mac; Metal GPU, real answers |
| Docker | DummyLLM | Validates the container build; no LLM needed |
| Docker | Host Ollama | Container app + GPU-accelerated host Ollama (recommended on Apple Silicon) |
| Docker | Container Ollama | CPU-only on Mac (slow); fine on Linux + NVIDIA |

The next sections give the exact commands for each row.

### One-time prep (any setup)

```bash
# 1. Install dependencies
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

# 2a. (Optional) Download the real EUR-Lex PDFs.
#     If EUR-Lex's CloudFront WAF blocks the request, the script prints
#     manual download URLs you can open in a browser.
python scripts/download_data.py

# 2b. Build the vector store + citation graph.
#     Falls back to the shipped fixture (data/raw/fixture.jsonl)
#     if no PDF is present — the pipeline still works end to end.
python scripts/build_indices.py
```

The build takes ~30 s on the fixture (19 chunks) and ~60 s on the full PDFs (542 chunks). The first run downloads the bge-small-en-v1.5 embedding model from HuggingFace (~67 MB).

### Run native + DummyLLM

No external services required. Useful for quick UI smoke testing.

```bash
LLM_BACKEND=dummy streamlit run src/agentic_rag/ui/app.py
# → http://localhost:8501
```

### Run native + host Ollama (recommended on Mac for real answers)

One-time install of Ollama on macOS (Apple Silicon native, Metal GPU):

```bash
brew install --cask ollama-app
open -a Ollama                         # starts the menubar agent + server
ollama pull qwen2.5:7b-instruct        # ~4.7 GB, one-time download
```

Verify the server is reachable:
```bash
curl -sf http://localhost:11434/api/tags | head -3
```

Run the UI:
```bash
LLM_BACKEND=ollama OLLAMA_MODEL=qwen2.5:7b-instruct \
  streamlit run src/agentic_rag/ui/app.py
```

A typical query takes ~30–60 s end-to-end on M1 Max.

### Run Docker + DummyLLM

Self-contained — only Docker Desktop required. The `app` image bundles the
fixture-built indices, so no host data mount is needed for a smoke run.

```bash
docker compose build app                     # ~3–5 min first time
LLM_BACKEND=dummy docker compose up app
# → http://localhost:8501
```

If port 8501 is busy:
```bash
APP_PORT=8502 LLM_BACKEND=dummy docker compose up app
```

### Run Docker + host Ollama (recommended on Apple Silicon)

Apple Silicon Metal GPU is **not** exposed to Linux containers, so the
in-compose `ollama` service would fall back to slow CPU. Best path: keep
Ollama running natively on the host and point the container at it via the
special `host.docker.internal` DNS name.

```bash
# 1. Make sure host Ollama is up (see "Run native + host Ollama" above)
curl -sf http://localhost:11434/api/tags >/dev/null && echo "ollama ok"

# 2. Start the container app pointing at the host LLM
OLLAMA_HOST=http://host.docker.internal:11434 \
LLM_BACKEND=ollama \
OLLAMA_MODEL=qwen2.5:7b-instruct \
  docker compose up app
# → http://localhost:8501
```

### Run Docker + in-compose Ollama (Linux / CUDA hosts only)

```bash
docker compose --profile with-llm up -d
docker compose exec ollama ollama pull qwen2.5:7b-instruct
LLM_BACKEND=ollama docker compose up app
```

On Apple Silicon this works but inference is CPU-only and noticeably slower
than the host-Ollama variant above.

### Other useful commands

```bash
# Unit + smoke tests (~40 s, all 27 with the DummyLLM)
LLM_BACKEND=dummy python -m pytest tests/ -v

# Eval suite, deterministic baseline (~10 s; no LLM judging)
LLM_BACKEND=dummy python eval/runner.py --no-judge

# Eval suite with real LLM-as-judge (~15–20 min on M1 Max; needs Ollama running)
LLM_BACKEND=ollama OLLAMA_MODEL=qwen2.5:7b-instruct python eval/runner.py

# Ablation: vector-only retrieval (graph disabled)
GRAPH_HOPS=0 LLM_BACKEND=dummy python eval/runner.py --no-judge

# Load test (~5 s with DummyLLM, 3 concurrency scenarios)
LLM_BACKEND=dummy python load_test/run.py --n 100 --workers 1,5,10

# Stop everything
docker compose down                          # containers
pkill -f "streamlit run"                     # native Streamlit
osascript -e 'quit app "Ollama"'             # host Ollama (optional)
```

Reports land in `eval/reports/report_<timestamp>.md` and `load_test/reports/report_<timestamp>.md`. Each is a self-contained markdown summary with per-question breakdown and latency tables.

### Verify the install end-to-end

```bash
# All 27 unit tests should pass:
LLM_BACKEND=dummy python -m pytest tests/ -q

# Quick CLI sanity check on a single query:
LLM_BACKEND=dummy python -c "
import sys; sys.path.insert(0, 'src')
from agentic_rag.graph.main import run_query
out = run_query('What are prohibited AI practices under the AI Act?')
print('intent:', out['intent'])
print('citations:', len(out.get('citations') or []))
print('answer:', out.get('final_answer', '')[:200])
"
```

If both succeed, the pipeline is healthy. From here, switch to a real LLM (Ollama section above) for natural-language answers.

---

## 5. Project layout

```
.
├── src/agentic_rag/
│   ├── config.py            # env-driven settings
│   ├── ingest/              # PDF → structured chunks
│   ├── rag/
│   │   ├── store.py         # Chroma + NetworkX (build + load)
│   │   └── subgraph.py      # 5-node RAG subgraph
│   ├── graph/
│   │   ├── state.py         # AgentState TypedDict
│   │   ├── nodes.py         # router / planner / rag_invoker / tool_executor / composer / verifier
│   │   └── main.py          # StateGraph wiring
│   ├── llm/base.py          # Ollama + DummyLLM backends
│   ├── tools/
│   │   ├── deadline.py      # non-retrieval: AI Act phase dates
│   │   └── web_search.py    # non-retrieval: DuckDuckGo
│   └── ui/app.py            # Streamlit UI
├── scripts/
│   ├── download_data.py     # EUR-Lex PDF fetcher (with manual fallback)
│   └── build_indices.py     # ingest → vector + graph
├── eval/
│   ├── questions.yaml       # 15-question gold set
│   ├── runner.py            # 3-layer scoring + markdown report
│   └── reports/             # timestamped eval reports (gitignored)
├── load_test/
│   ├── run.py               # 100-query scenarios with concurrency sweep
│   └── reports/             # timestamped load reports (gitignored)
├── tests/                   # pytest: parser, LLM layer, subgraph, main graph, tools
├── data/
│   ├── raw/fixture.jsonl    # shipped 19-chunk corpus for smoke testing
│   ├── raw/*.pdf            # downloaded EUR-Lex PDFs (gitignored)
│   ├── processed/           # JSONL of normalized chunks
│   ├── chroma/              # Chroma persistent client
│   └── graph/               # NetworkX pickle + GraphML
├── Dockerfile               # multi-stage, slim runtime, uv
├── docker-compose.yml       # app + optional ollama (with-llm profile)
├── DEVLOG.md                # chronological steps + decision journal (Hungarian)
└── ROADMAP.md               # Phase 2 / Phase 3 + GCP + Azure mapping (English)
```

---

## 6. Known limitations / next steps

- **Cross-regulation graph edges** — currently only within-regulation (fix queued, see DEVLOG §4 — Hungarian).
- **Stateless LLM calls (no conversational memory)** — every query is sent to the LLM in isolation. The chat history visible in the UI is purely cosmetic; the agent does not see prior turns, so pronoun follow-ups like *"What does it reference?"* do not resolve correctly. Targeted fix in Phase 2: a query-rewrite pre-step that folds the last N turns into a self-contained question before the router.
- **Reranker** — cross-encoder reranker (e.g. `bge-reranker-base`) would lift retrieval precision ~5–10pp for a ~150 ms cost; targeted for Phase 2.
- **Entity / GraphRAG-style graph** — deferred pending eval signal; current citation-graph approach already closes most cross-reference gaps deterministically.
- **Streaming UI** — composer answer currently renders on completion; streaming would shave perceived latency.

Full roadmap: [`ROADMAP.md`](./ROADMAP.md).

---

_This assistant does not provide legal advice. Always consult a qualified counsel for actual compliance decisions._
