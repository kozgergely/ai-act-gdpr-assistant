# Roadmap — Prototípustól a produkcióig

Ez a dokumentum a jelenlegi **Phase 1 (prototípus)** utáni következő fázisokat írja le, minden képességhez egy-egy **GCP** és **Azure** megoldás-leképezéssel. Folyamatosan frissül, ahogy a prototípusból tanulunk.

> Utolsó frissítés: 2026-04-23

---

## Fázis-áttekintés

| Dimenzió | Phase 1 – Prototype (most) | Phase 2 – Pilot | Phase 3 – Production |
|---|---|---|---|
| **Felhasználók** | 1 (a fejlesztő) | 5–50 belső pilot user | 1k–10k+, multi-tenant |
| **Deployment** | `docker compose up` lokálisan | Egyetlen managed region | Multi-region, multi-env (dev/stage/prod) |
| **LLM** | Ollama `qwen2.5:7b` lokálisan | Managed inference (GPU endpoint) | SLA-s, auto-scaled, A/B-s modell routing |
| **Adatforrás** | 2 PDF (AI Act + GDPR) + fixture | +szakmai guidance (EDPB, EDPS, AI Office) | Folyamatos EUR-Lex CELEX delta ingestion, versioning |
| **Index** | Chroma helyben + pickled NetworkX | Managed vector DB + Graph DB | Multi-region, sharded, tiered storage |
| **Eval** | 15 kérdés, offline | 50–100 kérdés + LLM-as-judge + human review | Continuous eval, canary release, drift detection |
| **Governance** | Disclaimer a válaszban | PII redaction + access log | Audit trail (GDPR Art. 30 style), versioned answers, e-discovery |
| **Auth** | Nincs | SSO (OIDC) + RBAC | Multi-tenant isolation, fine-grained ACL a dokumentumokig |

---

## Phase 2 — Pilot (3–6 hónap a prototípus után)

**Célja:** Hiteles belső/limitált külső felhasználók kiszolgálása, managed cloudban. A pipeline minőségét és a tanulási hurkokat priorizáljuk.

### Funkcionális delták a Phase 1-hez képest

1. **Minőségi retrieval upgrade**
   - Cross-encoder reranker (pl. `bge-reranker-large`) a `fuse_rerank` mögé.
   - LLM-alapú entity extraction → **hibrid citation+entity graph** (a jelenlegi regex-cross-ref MELLETT, nem helyette).
   - Layout-aware PDF parsing (table-érzékeny, LayoutLMv3 vagy Unstructured).
   - **Conversational memory** — query-rewriting pre-step a router elé, ami az utolsó 2–3 turn-t önmagában értelmes query-vé fordítja; a Phase 1 prototípus stateless LLM-hívásokkal dolgozik (history csak a UI-ban él), így a *"What does it reference?"* típusú pronoun-follow-upok jelenleg nem oldódnak fel. Részletesen DEVLOG §4.
2. **Eval & observability**
   - Full trace minden noderől (input, output, latency, token count) → központi tracing system.
   - 50–100 kérdéses golden set + LLM-as-judge + havi human review 20%-nyi válaszra.
   - Drift monitor: embedding-space distance változás új dokumentumok után.
3. **Adatok**
   - EDPB + EDPS guidance + AI Office Q&A ingestion.
   - Versioning: minden chunkhoz `effective_from`, `superseded_by` mező; a válaszok időbélyeggel idéznek.
4. **Biztonság & compliance**
   - PII redaction a user query-k logolásakor.
   - Prompt injection szűrő (pl. reguláris input sanitization + Rebuff-szerű detector).
   - Response safety classifier (kikerülje a konkrét jogi tanácsot; irányítsa ügyvédhez).
5. **Infrastruktúra**
   - Stateless app service, managed DB, titkos adatok managed vaultban.
   - CI/CD: PR-ra teljes eval pipeline fut, regressziót blokkol.

### Phase 2 — képesség → cloud leképezés

| Képesség | GCP | Azure |
|---|---|---|
| **LLM inferencia** | Vertex AI Model Garden (Llama-3.3-70B / Gemini 1.5 Pro); vagy saját GKE GPU + vLLM | Azure AI Foundry (Llama-3.3 / Mistral Large); Azure OpenAI Service (GPT-4o/o1-mini) |
| **Embedding** | Vertex AI `text-embedding-005` vagy saját bge GKE-n | Azure OpenAI `text-embedding-3-large` vagy saját bge AKS-en |
| **Vector DB** | Vertex AI Vector Search (managed ANN, Matching Engine) **vagy** AlloyDB + pgvector | Azure AI Search (vector + semantic ranker beépítve) **vagy** Cosmos DB for PostgreSQL + pgvector |
| **Graph DB** | Spanner Graph (GQL, globally consistent) **vagy** Neo4j AuraDB GCP Marketplace-ből | Cosmos DB for Apache Gremlin **vagy** Neo4j AuraDB Azure Marketplace-ből |
| **Orchestration / API** | Cloud Run (LangGraph server + Streamlit külön service) | Azure Container Apps (LangGraph + Streamlit) |
| **Document ingestion** | Cloud Storage (bucket) → Cloud Run Job (parser) → Pub/Sub → index worker | Blob Storage → Azure Functions → Service Bus → index worker |
| **Secrets** | Secret Manager | Azure Key Vault |
| **Auth (SSO)** | Identity Platform / IAP | Microsoft Entra ID + APIM policy |
| **Observability** | Cloud Trace + Cloud Logging + Managed Prometheus | Application Insights + Azure Monitor + Log Analytics |
| **Tracing (LLM-specific)** | LangSmith / Langfuse self-hosted Cloud Run-on | LangSmith / Langfuse self-hosted Container Apps-ban |
| **Eval CI** | Cloud Build + Artifact Registry | Azure DevOps / GitHub Actions + Azure Container Registry |
| **PII redaction** | Cloud DLP API | Microsoft Purview DLP + Azure AI Language (PII detection) |

### Phase 2 trade-offok

- **Vertex AI Vector Search vs. AlloyDB pgvector** — a Vector Search < 50ms p99 latency-t ad, de kevésbé rugalmas a metaadat-szűrésre. AlloyDB pgvector jobb, ha komplex SQL-szerű filtering kell (regulation × effective date). **Ajánlás:** alloyDB/pgvector, amíg < 10M chunk; felette Vector Search.
- **Azure AI Search vs. pgvector** — Az Azure AI Search-ben a semantic ranker out-of-the-box jó minőségű, és BM25 + vector hibrid keresés támogatott. Jobb első választás Azure-on, mint a pgvector, ha a fő use case RAG.
- **Gemini/GPT-4o vs. self-hosted** — Managed API egyszerűbb, de havi költség lineárisan skálázódik a lekérdezésekkel. Self-hosted (GKE/AKS) vLLM-mel fix cost, de SRE-teher. **Ajánlás Phase 2-re:** managed API (gyorsabb piacra jutás), és méretet figyelni a váltási ponton (~50k query/nap ≈ self-host break-even).
- **Spanner Graph vs. Cosmos Gremlin** — Spanner Graph újabb (2024), relatíve limitált driver-ökoszisztéma, viszont erős konzisztencia + SQL join graph-query-kkel. Cosmos Gremlin érettebb, multi-model. Mindkettő viable; a meglévő cloud preferenciát kövesse.

---

## Phase 3 — Production (pilot után 6–12 hónap)

**Célja:** Multi-tenant, enterprise-grade, SLA-s, audited szolgáltatás. Ennek központjában a **versioning**, a **governance** és a **continuous improvement** áll.

### Funkcionális delták a Phase 2-höz képest

1. **Multi-tenancy**
   - Per-tenant vector namespace + per-tenant fine-tuned prompt templatek.
   - Tenantonként külön embedding model / LLM választhatóság (pl. EU-only vs. globális).
2. **Governance & audit**
   - Teljes query + retrieval + answer + model-verzió audit log, immutable storage-on, tenant-szinten izolálva.
   - "Provenance badge": minden citation mellé a EUR-Lex CELEX+verzió+effective date.
   - GDPR Article 17 (right to erasure) támogatás a user-chatlog-okra.
3. **Continuous evaluation**
   - Minden release-en 500+ kérdés futása, regressziós küszöbbel.
   - Human-in-the-loop: random 1% válaszra jogász review, eredmény visszacsatorva prompt + retrieval finetune-hoz.
   - Shadow traffic új modellre / indexre a production rollout előtt.
4. **SLA / DR**
   - Multi-region active/active (legalább read), active/passive a write path-ra.
   - RPO < 5 perc a dokumentum-ingestionra; RTO < 15 perc az inference-re.
5. **Cost governance**
   - Per-tenant cost attribution (token counting + vector search quota).
   - Kis modell (Gemini Flash / GPT-4o-mini) fallback egyszerű kérdésekre; router dönt.
6. **Verzió & lifecycle**
   - Mindig konkrét AI Act / GDPR **verzióra** válaszol (consolidated text dátummal).
   - Új consolidated verzió publikálásakor automatikus re-ingest + delta-eval.

### Phase 3 — képesség → cloud leképezés

| Képesség | GCP | Azure |
|---|---|---|
| **LLM routing** | Vertex AI Model Garden + Vertex AI Prediction Router (Gemini Pro nagy, Flash kicsi) | Azure AI Foundry + Prompt Flow router (GPT-4o vs. 4o-mini) |
| **Vector DB (scale)** | Vertex AI Vector Search (multi-region replication) | Azure AI Search (SKU S3 HD / L2, multi-region replication) |
| **Graph DB (scale)** | Spanner Graph (global, regional replicas) | Cosmos DB for Gremlin (multi-region, multi-write) |
| **Agent orchestration** | Vertex AI Agent Builder + saját LangGraph custom agents (GKE) | Azure AI Foundry Agent Service + LangGraph (AKS) |
| **Ingestion pipeline** | Cloud Composer (Airflow 2) + Dataflow (Beam) batch; Pub/Sub + Cloud Functions delta | Azure Data Factory + Databricks batch; Event Grid + Functions delta |
| **Document versioning** | BigQuery + Cloud Storage verzionálva, Dataplex metadata | Data Lake Storage Gen2 + Purview catalog |
| **Audit log (immutable)** | Cloud Logging → BigQuery sink bucket policy-vel (lock + retention) | Log Analytics + Azure Storage immutable blob (WORM) |
| **Multi-tenant identity** | Identity Platform + per-tenant JWT custom claims; IAM Conditions | Entra ID External ID (B2B/B2C) + APIM subscription per tenant |
| **RBAC (row-level)** | BigQuery row-level security + per-tenant vector namespace | Azure AI Search security filters + RLS a metadata DB-n |
| **Secrets + KMS** | Cloud KMS + Secret Manager + CMEK mindenhol | Key Vault + Azure Disk Encryption + CMK mindenhol |
| **Continuous eval** | Vertex AI Experiments + custom job GKE-n | Azure AI Foundry Evaluation + custom job AKS-en |
| **A/B + shadow traffic** | Cloud Run traffic splitting + Vertex Endpoints traffic split | Container Apps revisions + API Management canary |
| **Cost attribution** | Billing export → BigQuery + tenant tag | Cost Management + Resource tags |
| **Data loss prevention (PII)** | Cloud DLP + VPC-SC perimeter | Microsoft Purview + Private Endpoints |
| **Responsible AI** | Vertex AI Safety Filters + Model Armor | Azure AI Content Safety + Prompt Shields |

### Phase 3 trade-offok + megjegyzések

- **GCP vs. Azure ökoszisztéma:** Jogi / compliance szegmensben az ügyfelek **Azure**-t preferálják (Entra ID, Purview érettebb governance), hacsak nincs explicit Google-affinity. Azonban, ha erős a Gemini / Vertex AI Agent Builder választás, GCP is tökéletes.
- **EU data residency:** Mindkét cloud EU-régiót ad (europe-west3/4 GCP, West Europe / Sweden Central Azure). **Azure OpenAI** EU Data Boundary kötelezően aktiválandó; **Vertex AI Gemini** EU residency most már elérhető, de a modell-régió kifejezetten konfigurálandó.
- **Self-hosted vs. managed LLM Phase 3-ra:** 10k+ napi query esetén self-hosted (vLLM vagy TGI AKS/GKE-n) cost-optimum, de SRE-igényes. Managed szolgáltatások (GPT-4o, Gemini Pro) jobbak, amíg nem éri el a break-even-t, vagy compliance miatt saját infra kell.
- **Graph DB választás:** Ha a citation+entity graph > 100M él, Neo4j enterprise (managed) jobb mint a Cosmos Gremlin / Spanner Graph, mert a Cypher ökoszisztéma + graph algoritmusok (PageRank, community detection) érettebbek.
- **Migration path:** Phase 1 → Phase 2 az app-szintű elszigetelésnek köszönhetően nem érinti a domain logikát, csak az `rag/store.py` + `llm/base.py` + `config.py`-t. Phase 2 → Phase 3 a multi-tenancy és a versioning bevezetése miatt **breaking** (state, storage modell változik) — tervezzünk migration windowt.

---

## Közös megfontolások minden fázisra

### Költség-sávok (saccolt, 2026 árak)

| Fázis | Nagyságrend (USD/hó) | Fő tételek |
|---|---|---|
| Phase 1 | 0 | Saját gép, nulla API |
| Phase 2 | 2k–10k | Managed LLM (GPT-4o/Gemini) + vector DB + compute |
| Phase 3 | 30k–100k+ | LLM token (~70%), storage (~10%), compute (~15%), egyéb |

### Observability / evaluation baseline

Függetlenül a cloudtól ugyanaz a **eval és trace pipeline** kell:
- **Offline golden set** verziózva (git), PR-ra futtatva.
- **Online trace** minden node-ról (LangSmith / Langfuse / custom).
- **Feedback loop**: 👍/👎 gombok a UI-n → bad examples → weekly review → prompt vagy retrieval fix → regresszió-teszt.

### Compliance-specifikus követelmények

- **EU AI Act Article 50 (transparency):** A chatbot world jelzi, hogy AI-val beszélget → már Phase 2-ben beépítendő.
- **Article 10 (data governance):** Training data (ha majd finetune-olunk) dokumentálva, representative, bias-elemzéssel.
- **Article 13 (instructions for use):** Dokumentáció a deployer-nek a korlátokról, intended use-ról → README + tech design doc része.
- **Saját GDPR:** Query logokat maximum 30 napig; PII redaction kötelező.

---

## Migration path összefoglaló

1. **Phase 1 → Phase 2:** `.env`-ben backend-flag-et flippelni (Ollama → Vertex AI / Azure OpenAI), `rag/store.py`-ban a Chroma cliens → managed vector DB cliens. A LangGraph maga változatlan. Becsült effort: ~2 hét + eval harness + CI.
2. **Phase 2 → Phase 3:** Multi-tenancy (per-tenant index), versioning adatmodell, audit log. Ez a domain-logikát nem érinti, de **schema migration**-t igényel. Becsült effort: ~2 hónap + compliance review.

---

## TODO / nyitott kérdések

- Q1. Pilotban melyik cloud? (A válasz befolyásolja, melyik managed szolgáltatás-stacket mélyebben validáljuk.)
- Q2. A Phase 2 dokumentum-ingestionben kell-e már OCR (scanned PDF-ek) — vagy a text-first EUR-Lex elég egy darabig?
- Q3. Mikor nyomjuk be az entitás-szintű graphot (GraphRAG/LightRAG stílus) a jelenlegi citation graph mellé? Eval-driven döntés.
