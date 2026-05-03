# Roadmap — From prototype to production

This document describes the phases following the current **Phase 1 (prototype)**, with a concrete **GCP** and **Azure** mapping for every capability. Updated continuously as the prototype teaches us more.

> Last updated: 2026-05-03

---

## Phase overview

| Dimension | Phase 1 — Prototype (current) | Phase 2 — Pilot | Phase 3 — Production |
|---|---|---|---|
| **Users** | 1 (the developer) | 5–50 internal pilot users | 1k–10k+, multi-tenant |
| **Deployment** | Local `docker compose up` | Single managed region | Multi-region, multi-env (dev/stage/prod) |
| **LLM** | Local Ollama `qwen2.5:7b` | Managed inference (GPU endpoint) | SLA-backed, auto-scaled, A/B model routing |
| **Data sources** | 2 PDFs (AI Act + GDPR) + fixture | + interpretive guidance (EDPB, EDPS, AI Office) | Continuous EUR-Lex CELEX delta ingestion, versioning |
| **Indices** | Local Chroma + pickled NetworkX | Managed vector DB + Graph DB | Multi-region, sharded, tiered storage |
| **Eval** | 18 questions, offline | 50–100 questions + LLM-as-judge + monthly human review | Continuous eval, canary release, drift detection |
| **Governance** | Disclaimer in the answer | PII redaction + access log | Audit trail (GDPR Art. 30 style), versioned answers, e-discovery |
| **Auth** | None | SSO (OIDC) + RBAC | Multi-tenant isolation, fine-grained ACL down to documents |

---

## Phase 2 — Pilot (3–6 months after prototype)

**Goal:** serve real internal / limited external users on managed cloud. Priority is pipeline quality and feedback loops.

### Functional deltas vs. Phase 1

1. **Retrieval quality upgrade**
   - Cross-encoder reranker (e.g. `bge-reranker-large`) downstream of `fuse_rerank`.
   - LLM-driven entity extraction → **hybrid citation + entity graph** (alongside the current regex-based cross-references, not replacing them).
   - Layout-aware PDF parsing (table-aware, e.g. LayoutLMv3 or Unstructured).
   - **Conversational memory** — a query-rewriting pre-step before the router that folds the last 2–3 turns into a self-contained query. The Phase 1 prototype passes stateless calls to the LLM (history is UI-only), so pronoun follow-ups like *"What does it reference?"* do not resolve. Details in DEVLOG §4.
   - **RRF tuning + graph re-weighting** — the Phase 1 ablation showed that the citation-graph chunks compete equally with vector hits in the current Reciprocal Rank Fusion, occasionally displacing high-relevance vector hits. Phase 2 will down-weight graph hits or score them by query-cosine before fusing. See DEVLOG §4.4.
2. **Eval & observability**
   - Per-node trace (input, output, latency, token count) into a central tracing system.
   - 50–100 question golden set + LLM-as-judge + monthly human review on ~20 % of answers.
   - Drift monitor: embedding-space distance shift after corpus updates.
3. **Data**
   - EDPB + EDPS guidance + AI Office Q&A ingestion.
   - Versioning: every chunk gains `effective_from` and `superseded_by` fields; answers cite with timestamps.
4. **Security & compliance**
   - PII redaction when logging user queries.
   - Prompt injection filter (input sanitization + a Rebuff-style detector).
   - Response safety classifier (avoid concrete legal advice; redirect to counsel).
5. **Infrastructure**
   - Stateless app service, managed DB, secrets in a managed vault.
   - CI/CD: full eval pipeline runs on every PR; regression blocks merge.

### Phase 2 — capability → cloud mapping

| Capability | GCP | Azure |
|---|---|---|
| **LLM inference** | Vertex AI Model Garden (Llama-3.3-70B / Gemini 1.5 Pro); or self-hosted GKE GPU + vLLM | Azure AI Foundry (Llama-3.3 / Mistral Large); Azure OpenAI Service (GPT-4o / o1-mini) |
| **Embeddings** | Vertex AI `text-embedding-005`; or self-hosted bge on GKE | Azure OpenAI `text-embedding-3-large`; or self-hosted bge on AKS |
| **Vector DB** | Vertex AI Vector Search (managed ANN, Matching Engine) **or** AlloyDB + pgvector | Azure AI Search (vector + semantic ranker built-in) **or** Cosmos DB for PostgreSQL + pgvector |
| **Graph DB** | Spanner Graph (GQL, globally consistent) **or** Neo4j AuraDB from GCP Marketplace | Cosmos DB for Apache Gremlin **or** Neo4j AuraDB from Azure Marketplace |
| **Orchestration / API** | Cloud Run (LangGraph server + Streamlit as separate services) | Azure Container Apps (LangGraph + Streamlit) |
| **Document ingestion** | Cloud Storage bucket → Cloud Run Job (parser) → Pub/Sub → index worker | Blob Storage → Azure Functions → Service Bus → index worker |
| **Secrets** | Secret Manager | Azure Key Vault |
| **Auth (SSO)** | Identity Platform / IAP | Microsoft Entra ID + APIM policy |
| **Observability** | Cloud Trace + Cloud Logging + Managed Service for Prometheus | Application Insights + Azure Monitor + Log Analytics |
| **Tracing (LLM-specific)** | LangSmith / Langfuse self-hosted on Cloud Run | LangSmith / Langfuse self-hosted on Container Apps |
| **Eval CI** | Cloud Build + Artifact Registry | Azure DevOps / GitHub Actions + Azure Container Registry |
| **PII redaction** | Cloud DLP API | Microsoft Purview DLP + Azure AI Language (PII detection) |

### Phase 2 trade-offs

- **Vertex AI Vector Search vs. AlloyDB pgvector** — Vector Search delivers <50 ms p99 latency but is less flexible for metadata filters. AlloyDB pgvector wins when complex SQL-style filtering is needed (regulation × effective date). **Recommendation:** AlloyDB / pgvector while < 10M chunks; Vector Search above that.
- **Azure AI Search vs. pgvector** — Azure AI Search ships a high-quality semantic ranker out of the box and supports BM25 + vector hybrid search. It's the better default on Azure when the primary use case is RAG.
- **Gemini / GPT-4o vs. self-hosted** — Managed APIs are simpler but cost scales linearly with traffic. Self-hosted vLLM on GKE/AKS is fixed cost but adds SRE overhead. **Phase 2 recommendation:** managed API for fastest time-to-market; revisit at the break-even (~50 k queries/day for a 7B class model).
- **Spanner Graph vs. Cosmos Gremlin** — Spanner Graph is newer (2024), with a smaller driver ecosystem but strong consistency and SQL-style joins on graph queries. Cosmos Gremlin is more mature and multi-model. Both are viable; follow the existing cloud preference.

---

## Phase 3 — Production (6–12 months after pilot)

**Goal:** multi-tenant, enterprise-grade, SLA-backed, audited service. The focus shifts to **versioning**, **governance**, and **continuous improvement**.

### Functional deltas vs. Phase 2

1. **Multi-tenancy**
   - Per-tenant vector namespaces and per-tenant fine-tuned prompt templates.
   - Tenant-selectable embedding model / LLM (e.g. EU-only vs. global).
2. **Governance & audit**
   - Full query + retrieval + answer + model-version audit log on immutable storage, isolated per tenant.
   - "Provenance badge" — every citation surfaces the EUR-Lex CELEX id + version + effective date.
   - GDPR Article 17 (right to erasure) support for user chat logs.
3. **Continuous evaluation**
   - 500+ question regression run on every release with a regression threshold.
   - Human-in-the-loop: random 1 % of answers reviewed by a lawyer, results fed back into prompt and retrieval fine-tuning.
   - Shadow traffic for new models / indices before production rollout.
4. **SLA / DR**
   - Multi-region active/active for reads at minimum; active/passive on the write path.
   - RPO < 5 min for document ingestion; RTO < 15 min for inference.
5. **Cost governance**
   - Per-tenant cost attribution (token counting + vector search quotas).
   - Small-model fallback (Gemini Flash / GPT-4o-mini) for simple queries; the router decides.
6. **Versioning & lifecycle**
   - Always answer against a specific AI Act / GDPR consolidated text version (date-stamped).
   - Automatic re-ingest + delta-eval when a new consolidated version is published.

### Phase 3 — capability → cloud mapping

| Capability | GCP | Azure |
|---|---|---|
| **LLM routing** | Vertex AI Model Garden + Vertex AI Prediction Router (Gemini Pro for hard, Flash for easy) | Azure AI Foundry + Prompt Flow router (GPT-4o vs. 4o-mini) |
| **Vector DB (scale)** | Vertex AI Vector Search (multi-region replication) | Azure AI Search (S3 HD / L2 SKU, multi-region replication) |
| **Graph DB (scale)** | Spanner Graph (global, regional replicas) | Cosmos DB for Gremlin (multi-region, multi-write) |
| **Agent orchestration** | Vertex AI Agent Builder + custom LangGraph agents (GKE) | Azure AI Foundry Agent Service + LangGraph (AKS) |
| **Ingestion pipeline** | Cloud Composer (Airflow 2) + Dataflow (Beam) for batch; Pub/Sub + Cloud Functions for delta | Azure Data Factory + Databricks for batch; Event Grid + Functions for delta |
| **Document versioning** | BigQuery + versioned Cloud Storage; Dataplex metadata | Data Lake Storage Gen2 + Purview catalog |
| **Audit log (immutable)** | Cloud Logging → BigQuery sink with bucket-locking and retention policies | Log Analytics + Azure Storage immutable blob (WORM) |
| **Multi-tenant identity** | Identity Platform + per-tenant JWT custom claims; IAM Conditions | Entra ID External ID (B2B/B2C) + APIM subscription per tenant |
| **RBAC (row-level)** | BigQuery row-level security + per-tenant vector namespaces | Azure AI Search security filters + RLS on the metadata DB |
| **Secrets + KMS** | Cloud KMS + Secret Manager + CMEK throughout | Key Vault + Azure Disk Encryption + CMK throughout |
| **Continuous eval** | Vertex AI Experiments + custom GKE jobs | Azure AI Foundry Evaluation + custom AKS jobs |
| **A/B + shadow traffic** | Cloud Run traffic splitting + Vertex Endpoints traffic split | Container Apps revisions + API Management canary |
| **Cost attribution** | Billing export → BigQuery + tenant tag | Cost Management + resource tags |
| **Data loss prevention (PII)** | Cloud DLP + VPC-SC perimeter | Microsoft Purview + Private Endpoints |
| **Responsible AI** | Vertex AI Safety Filters + Model Armor | Azure AI Content Safety + Prompt Shields |

### Phase 3 trade-offs and notes

- **GCP vs. Azure ecosystem:** in legal / compliance segments, customers tend to prefer **Azure** (Entra ID, Purview offer more mature governance), unless there is a clear Google affinity. If the Gemini / Vertex AI Agent Builder stack is the right fit, GCP is equally capable.
- **EU data residency:** both clouds offer EU regions (europe-west3/4 on GCP; West Europe / Sweden Central on Azure). **Azure OpenAI** requires EU Data Boundary to be explicitly enabled; **Vertex AI Gemini** EU residency is now available but the model region must be set explicitly.
- **Self-hosted vs. managed LLM in Phase 3:** above ~10 k queries/day, self-hosted (vLLM or TGI on AKS/GKE) is the cost optimum but SRE-heavy. Managed services (GPT-4o, Gemini Pro) win until the break-even point, or where compliance demands the model run on owned infrastructure.
- **Graph DB choice:** if the citation + entity graph exceeds ~100M edges, managed Neo4j Enterprise is preferable to Cosmos Gremlin / Spanner Graph because the Cypher ecosystem and the graph algorithms (PageRank, community detection) are more mature.
- **Migration path:** Phase 1 → Phase 2 does not touch the domain logic thanks to the app-level isolation — only `rag/store.py`, `llm/base.py`, and `config.py` change. Phase 2 → Phase 3 is **breaking** because of multi-tenancy and versioning (state and storage models change) — plan a migration window.

---

## Cross-cutting concerns (all phases)

### Cost bands (rough estimates, 2026 prices)

| Phase | Order of magnitude (USD/month) | Main drivers |
|---|---|---|
| Phase 1 | 0 | Local machine, no API |
| Phase 2 | 2k–10k | Managed LLM (GPT-4o / Gemini) + vector DB + compute |
| Phase 3 | 30k–100k+ | LLM tokens (~70 %), storage (~10 %), compute (~15 %), other |

### Observability / evaluation baseline

The same **eval and trace pipeline** applies regardless of cloud:
- **Offline golden set** versioned in git, executed on every PR.
- **Online trace** of every node (LangSmith / Langfuse / custom).
- **Feedback loop**: 👍 / 👎 buttons in the UI → bad examples → weekly review → prompt or retrieval fix → regression test.

### Compliance-specific requirements

- **EU AI Act Article 50 (transparency):** the chatbot must clearly disclose that the user is talking to an AI — to be wired in at Phase 2.
- **Article 10 (data governance):** if we ever fine-tune, training data must be documented, representative, and bias-analyzed.
- **Article 13 (instructions for use):** documentation for the deployer covering limitations and intended use — part of the README and the technical design doc.
- **Internal GDPR:** keep query logs at most 30 days; PII redaction is mandatory.

---

## Migration path summary

1. **Phase 1 → Phase 2:** flip the backend flag in `.env` (Ollama → Vertex AI / Azure OpenAI), swap the Chroma client in `rag/store.py` for a managed vector DB client. The LangGraph itself is unchanged. Estimated effort: ~2 weeks + eval harness + CI.
2. **Phase 2 → Phase 3:** introduce multi-tenancy (per-tenant index), the versioning data model, and the audit log. Domain logic is unaffected, but a **schema migration** is required. Estimated effort: ~2 months + compliance review.

---

## Open questions

- **Q1.** Which cloud for the pilot? The answer drives which managed service stack we validate more deeply.
- **Q2.** Do we need OCR in the Phase 2 ingestion pipeline (scanned PDFs), or is text-first EUR-Lex enough for now?
- **Q3.** When do we add an entity-level graph (GraphRAG / LightRAG style) alongside the current citation graph? Eval-driven decision, see DEVLOG §4.4.
