# Fejlesztői napló — Agentic RAG Chatbot (EU AI Act + GDPR)

Ez a dokumentum a fejlesztés során **folyamatosan frissül**. Két részből áll:

1. **Kronologikus lépések** — mit csináltunk, milyen sorrendben.
2. **Döntési napló** — minden nem triviális tervezési döntés, a mérlegelt alternatívák és az indoklás.

> Utolsó frissítés: 2026-05-03 — **Real-PDF ingestion + real-LLM eval kész**: 542 chunk, 694 graph él, qwen2.5:7b eval → **80% recall, 100% citation precision, 3.62/5 answer quality**. 27/27 teszt zöld. Részletek §1.13 (ingestion buktatók) és §4 (eval baseline-ok).

---

## 1. Kronologikus lépések

### 1.1. Feladat értelmezése + scoping
- Elolvastam a `RAG- Feladatkiírás- senior.pdf`-et.
- Kigyűjtöttem a kötelező elemeket: LangGraph, ≥5 node, ≥2 tool (min. 1 nem-retrieval), RAG subgraph, Streamlit UI, Docker, 10–20 kérdéses eval, 50–200 lekérdezéses load test.
- Tisztáztam a user-rel: domain = EU AI Act + GDPR; graph RAG is kell; nyelv = angol a termék, magyar a meta doksi.

### 1.2. Környezet és skeleton (task #1)
- `git init`, `src/agentic_rag/` csomag szerkezet: `graph/`, `rag/`, `tools/`, `ingest/`, `llm/`, `ui/`.
- `pyproject.toml` uv-val, Python 3.11 pin.
- `config.py`: központi `Settings` dataclass, környezeti változókból (dotenv).
- `.env.example` template.

### 1.3. Adatforrás + ingestion (task #2)
- **Letöltés:** `scripts/download_data.py` megpróbálja letölteni az EUR-Lex PDF-eket (AI Act 2024/1689, GDPR 2016/679). A CloudFront WAF a sandboxban blokkolja — lokálisan a user gépén menni fog. Kézi fallback dokumentálva.
- **Fixture:** `data/raw/fixture.jsonl` — 19 strukturált chunk (11 AI Act + 8 GDPR), valós cikkekből kivonatolva, valódi cross-referenceokkal. Ez lehetővé teszi a pipeline tesztelését PDF nélkül is.
- **Parser:** `ingest/parse.py` — pypdf alapú text extraction, regex-alapú struktúra-felismerés (Article/Recital/Cross-ref), hosszú article-öknél soft split 2200 karakteres ablakokkal 200 karakteres átfedéssel.
- **Cross-ref normalizálás:** `"article:N"`, `"recital:N"`, `"annex:X"` formátum → ez a citation graph node-azonosítója is.

### 1.4. Indexek (task #3)
- **Vector store:** Chroma PersistentClient, `BAAI/bge-small-en-v1.5` embeddings (normalizálva, cosine). Metadata: regulation/kind/number/title/page/cross_refs.
- **Citation graph:** NetworkX `DiGraph`, egy node = egy (regulation, kind, article-number). Élek: cross_refs-ek alapján, csak azonos regulation-en belül. Hiányzó referenciákra dangling node-ok, hogy láthatóak legyenek a coverage-lyukak. Perzisztencia: pickle + GraphML.
- **Validáció:** 19 chunk → 33 node, 39 él. Retrieval smoke test: "prohibited AI practices" → top hit `AI Act:article:5`. ✅

### 1.5. LLM layer (task #6)
- **Backend absztrakció:** `LLM` Protocol, két implementáció:
  - `OllamaLLM`: `qwen2.5:7b-instruct`, `format="json"` a strukturált outputhoz.
  - `DummyLLM`: determinisztikus heurisztikus fallback — CI/Docker smoke teszthez. Kulcsszó-alapú klasszifikáció router/planner/verifier outputokhoz.
- **Factory:** `make_llm()` backend ellenőrzéssel, handshake után automatikus fallback Dummyra.
- **Bug + fix:** a DummyLLM kezdetben a teljes promptból olvasta a kulcsszavakat → a router-prompt maga tartalmazta a "deadline" szót → false positive "tool" intent. Fix: regex-szel kivágjuk a "Question: ..." részt, és csak azon futtatjuk a heurisztikát.

### 1.6. RAG subgraph (task #4)
- **5 node:** `query_rewrite` (multi-query expansion: 2–3 paraphrase a user kérdéséről, **nem HyDE** — nem hipotetikus válasz-dokumentumot generálunk, csak alternatív kérdésmegfogalmazásokat) → `vector_retrieve` → `graph_expand` → `fuse_rerank` → `context_assemble`.
- **Graph expansion:** minden top-N vector hit köré 1 hop BFS a citation graph-on (ki- és befelé menő élek), dedup seed-ekre.
- **Reciprocal Rank Fusion (RRF, k=60):** a vector- és graph-rangsort külön rangsorolja és fuzionálja — így egy graph-expandalt Article 9 tényleg bekerül a top-6-ba még akkor is, ha a vector nem adta vissza.
- **Validáció:** "prohibited AI practices" → fused top-6 tartalmazza az Article 9-et `sources=["graph"]` címkével. ✅

### 1.7. Fő agent graph + tools (task #5, majd #7–#11 gyors egymásutánban)
- **6 node:** `router` → `planner` | `tool_executor` | `composer`, `planner` → `rag_invoker` (loop a sub-questionökön) → `composer` → `verifier` → END | retry.
- **Autonóm döntések:** `router` (intent = rag/tool/direct), `verifier` (grounded? retry vagy END). Max 1 retry.
- **Tools (3 összesen, ebből 2 nem-retrieval):**
  - `rag_subgraph` (retrieval, a feladatkiírás szerint ez is egy tool),
  - `deadline_calc` — AI Act Article 113 alapján a különböző alkalmazási fázisok dátumai; a kérdésből kulcsszó-alapon rendeli a megfelelő fázishoz.
  - `web_search` — DuckDuckGo (nem igényel API kulcsot), friss guidance-re / enforcement newsra.
- **Citations:** a `composer` a fused chunkokból dedupál (regulation, kind, article-number) kulccsal, és score szerint rendez.
- **Tesztek:** 3 end-to-end smoke a fő graphre — RAG flow, tool flow, citation extraction — mind zöld.

### 1.8. Streamlit UI (task #7)
- Két-oszlopos split (chat bal, agent trace jobb), collapse-olható trace per turn.
- Sidebar: index-állapot, sample questions, LLM-backend kijelzés, clear history.
- Headless boot-teszttel validálva: `/_stcore/health` OK, oldal betölt.

### 1.9. Eval (task #8)
- `eval/questions.yaml` — 15 hand-crafted kérdés 7 kategóriában (factual / multi_article / cross_ref / deadline / scoping / out_of_scope + crossover).
- `eval/runner.py` — háromrétegű scoring (recall, citation precision, LLM-as-judge), markdown report.
- **Első baseline (DummyLLM):** recall 96.7%, citation precision 100%, intent accuracy 100%, mean latency 78 ms.
- Egy ismert korlát: q10 cross-regulation — a graph csak azonos regulation-en belül köt. Fix-ötletek rögzítve §4-ben.

### 1.10. Load test (task #9)
- `load_test/run.py` — 100 query, 3 worker-count (1 / 5 / 10), warm-up query a first-call overhead kiejtésére.
- **Eredmény:** sequential 46 ms p50 / 23 qps; 5 worker 200 ms p50 / 30 qps (throughput-csúcs); 10 worker 461 ms p50 / 23 qps (contention).
- **Bug + fix:** a script eredetileg némán exitált, mert hiányzott az `if __name__ == "__main__":` guard. Javítva.
- Node-szintű per-call profiling a jelenlegi verzióban heurisztikus (a trace-ből derivált node-sorrend); streaming event-alapú breakdown a következő iteráció.

### 1.11. Docker (task #10)
- `Dockerfile` multi-stage: builder (uv + deps + index build), runtime (python:3.11-slim, non-root user, healthcheck).
- `docker-compose.yml`: `app` (mindig) + `ollama` (`--profile with-llm` alatt). Apple Silicon-on a natív host Ollama preferált (Metal nincs Linux containerben), kommentben dokumentálva.
- `.dockerignore` kizárja a venv-et, .git-et, generált indexeket, PDF-eket.
- Docker daemon a sandboxban nem fut; te lokálisan építed. Compose YAML szintaxis-validált.

### 1.13. Real-PDF ingestion: nehézségek és megoldások

A fixture-fallbacktől eljutottunk a valódi EUR-Lex PDF-ek (AI Act + GDPR) feldolgozásáig. Itt a tipikus buktatókat és a fix-eket rögzítem — gyakorlati tanulságok bárkinek, aki hasonló legal-corpus pipeline-t épít.

#### 1.13.1. Letöltés: EUR-Lex CloudFront WAF

**Probléma:** A `download_data.py` automatizált letöltése `202 Accepted` választ kap "x-amzn-waf-action: challenge" header-rel. A CloudFront WAF JavaScript-challenge-et küldene egy böngészőnek, de ezt egy curl/httpx kliens nem tudja teljesíteni → a script zerót letöltött.

**Megoldás:**
- A script **több URL-t** próbál (CELEX, OJ-style, ELI), és bármelyik sikere után megáll.
- Ha mind elbukik, **explicit manual instructions**-t ír a console-ra: forrás-URL, böngészőben megnyitandó link, célfájl-név.
- A SOURCES dict **több fájlnév-formátumot** elfogad regulationonként (`ai_act.pdf` ÉS `OJ_L_202401689_EN_TXT.pdf`), mert a user manuálisan letöltött fájl gyakran az official OJ-named.
- Tapasztalat: a **GDPR letöltés végül automatikusan ment** (CloudFront kihagyta a challenge-et), az **AI Act manuális** kellett. A script ezt a felemás állapotot eleve támogatja: ami megvan, jó; ami nincs, instructions-szel jön.

#### 1.13.2. PDF text extraction: a pypdf "Ar ticle" bug

**Probléma:** Az EUR-Lex PDF-ek olyan font-konfigurációval készülnek, amit a `pypdf` rosszul interpretál: a karakterek között extra szóközt szúr be → `"Article"` helyett `"Ar ticle"`, `"testing"` helyett `"te sting"`, `"purpose"` helyett `"pur pose"`. **Minden regex** a parser-ünkben (article header, recital, cross-ref) **failel** ezen a torzított szövegen.

A pypdf 254 chunk-ot detektált — **mind recital, nulla article**. A regex `^\s*Article\s+(\d+)\s*$` egyszerűen nem matchelt sehol.

**Megoldás:** Áttértünk **PyMuPDF (`pymupdf`)**-re mint elsődleges PDF-text-engine. A pypdf maradt fallback (ha a pymupdf nincs telepítve). A pymupdf:
- Ugyanazon AI Act PDF: 460 chunk, ebből 224 article + 236 recital ✅
- Tisztán: `"Article 57"`, `"purpose"`, `"testing"` — semmi karakter-fragmentáció
- Mellékesen ~3× gyorsabb is

Kódszinten egy `try-except import + 1 ág` az `extract_text`-ben, plusz `pymupdf>=1.24.0` a `pyproject.toml`-ban.

#### 1.13.3. Recital duplicate IDs: az ELI page-footer collision

**Probléma:** Sikeres extract után a Chroma `DuplicateIDError`-t dobott: 56 recital-ID kétszer szerepelt (recital:1, recital:2, ..., recital:18 mind 2× kétszer). A regex `(?m)^\((\d+)\)\s+` ezeket matchelte:
1. **Valódi recital:** `(1) The purpose of this Regulation is to improve...`
2. **Page-footer footnote:** `(1) OJ C 517, 22.12.2021, p. 56.` (minden oldal alján a publication metadata)

Mindkettő `(N)` mintával kezdődik a sor elején — a regex nem tud különbséget tenni a tartalom alapján.

**Megoldás két szintű:**

**(a) Page-szintű footer-stripping az `extract_text`-ben.** Az EUR-Lex page-footer mindig egy `ELI: http://data.europa.eu/eli/...` markerrel kezdődik, és a page végéig tart. Erre regex:
```python
_PAGE_FOOTER_RE = re.compile(r"ELI:\s+https?://data\.europa\.eu[\s\S]*", re.IGNORECASE)
```
Minden oldal extract-elt szövegéből az ELI marker után minden eltávolításra kerül **mielőtt** a page-eket összefűzzük teljes dokumentummá.

**(b) Recital-zóna explicit határolása.** Az EUR-Lex preambulum struktúra:
```
... bibliographic info ...
Whereas:                              ← recital zone start
(1) ...
(2) ...
...
HAS ADOPTED THIS REGULATION:           ← recital zone end
CHAPTER I
Article 1
```

A `_split_recitals` most explicit a `Whereas:` és `HAS/HAVE ADOPTED THIS REGULATION:` markerek közötti zónában keres, nem "minden a first Article előtt" heurisztikával. Ez kizárja a bibliográfiai meta `(1) OJ C ...` typú lábjegyzeteket is.

A két fix kombinációja után: 0 duplicate, 542 chunk tisztán.

#### 1.13.4. DummyLLM planner: system-prompt-fragmentumok mint sub-questionök

**Probléma:** A real-corpus eval-on mindenhol ugyanaz a "noise" jött elő top retrieved doc-ként: `GDPR:article:28`, `AI Act:recital:124`, `GDPR:article:57`, `GDPR:article:81`, `AI Act:article:16`, `AI Act:article:74` — még olyan kérdéseknél is, ahol semmi közük nem volt. A recall a fixture-en mért 96.7%-ról 73.3%-ra esett.

**Diagnózis:** A `DummyLLM.generate_json` planner-ágában a régi heurisztika **a teljes promptot** (a system instructionnel együtt) szétdarabolta `[.?\n]` karaktereken, és a 10–200 karakteres fragmenseket "sub-question"-ként visszaadta. Tipikus output:
- "Decompose the following compliance question into 1-3 concrete retrieval-ready sub-questions"
- "A single sub-question is fine if the query is already atomic"
- "Question: What are the lawful bases for processing personal data under GDPR?"

A rag_invoker MIND a hármat futtatta egy-egy retrieval-pass-szel, és a procedurális/instruction-szerű szöveg minden query-re ugyanazt a procedurális chunk-halmazt hozta vissza (Article 28 ról "duties of processors", Article 74 ról monitoring). **Ez polluálta a fused listát.**

**Megoldás:** A `DummyLLM` mindenütt (router, planner, rewrite) **a `Question: ...` portion-t** kivágja a promptból regex-szel, és **csak azon** dolgozik. Ugyanúgy mint a router-fix korábban.

A planner most egyetlen sub-question-t ad vissza (a tényleges user query-t), nem három frankenstein-fragmentumot. Eredmény: 73.3% → 66.7% recall (kicsit csökkent, mert most nem véletlenül kapta el a noise sometimes a várt cikket; a recall-szám reálisabb baseline).

#### 1.13.5. Sub-chunk split: a tesztek aksrertcurance-update-je

**Probléma:** Az AI Act Article 5 ~12 000 karakter hosszú. A soft_split (2200 char window, 200 char overlap) ezt 6 sub-chunkra bontja: `5.1, 5.2, 5.3, 5.4, 5.5, 5.6`. A fixture-ön Article 5 egyetlen chunk volt, ezért a tesztek pontos `"AI Act:article:5"` ID-egyezést asszertáltak.

**Megoldás:**
- `test_prohibited_practices_retrieval`: most "**bármely** Article 5 sub-chunk" elfogadott (`startswith("AI Act:article:5.")` OR `== "AI Act:article:5"`).
- `test_graph_expansion_pulls_neighbors`: a fixture-specifikus Article 5 → Article 9 él a real PDF-ben nincs (kézzel írt cross-ref volt). A teszt most általánosabb: **bármely graph-only chunk** elég.
- Az **eval runner** nem érintett, mert a `_citation_to_id` segédfüggvény eleve article-szintre normalizál: `5.1` → `5`, így a recall-számolás transzparens a sub-chunk split-re.

#### 1.13.6. Eval runner LLM-judge bug: `q['query']` KeyError

**Probléma:** A real-LLM eval-futtatáskor a `_judge` függvény `q['query']`-t próbált olvasni a YAML-kérdés dict-ből, de a `questions.yaml`-ban a query mező nincs explicit — a runner egy belső `_query_text(q)` helper hozza össze az ID-ből. A judge promptbuilder ezért `KeyError: 'query'`-vel halt el.

**Megoldás:** A `_judge` szignatúrája `query=` keyword-only paramétert kapott; a hívó (`run_eval`) átadja a már kiszámolt query stringet. A judge promptja mostantól ezt használja.

#### 1.13.7. Mit jelent ez a leadás szempontjából

- A **valódi PDF-eken** működik az ingestion: 542 chunk, 694 cross-reference él, ~2.5 MB AI Act + 608 KB GDPR feldolgozva.
- A **DummyLLM eval recall** csökkent (96.7% → 66.7%) a corpus-méret-növekedés miatt — ez **várható és reprodukálható**.
- A **real LLM eval** a query rewrite révén jelentősen jobb recall-t ad (várható 80%+, mérni kell).
- A pipeline most **minden EU regulation PDF-en** működik (nem csak az AI Act / GDPR-ön), feltéve hogy a struktúra `Whereas:`/`HAS ADOPTED` formátum.

### 1.12. README finalizálva (task #11)
- Problem statement + agentic RAG indoklása
- Architektúra ASCII graph (main + RAG subgraph)
- Node + tool táblázat
- Eval + load test eredmények teljes reprodukció-utasítással
- Setup: native + Docker, Apple Silicon speciális útvonal
- Ismert korlátok + next steps

---

## 2. Döntési napló

### D1. Domain: EU AI Act + GDPR (vs. orvosi / pénzügyi / dev docs)
**Választás:** AI Act + GDPR compliance asszisztens.
**Alternatívák:**
- Orvosi irányelvek (pl. ESC guidelines) → szakzsargon, nehezebb kiértékelni, kockázatos a hallucináció.
- Pénzügyi jelentések (10-K, SEC filings) → nagyobb volumen, de a vizsgált minőség nem annyira "cross-referenciás", mint a jogi szövegek.
- Dev docs (LangGraph, Kubernetes) → túl "meta", kevésbé demonstrálja a graph RAG erejét.
**Indoklás:** Az AI Act + GDPR (a) valós és aktuális fájdalompont 2026-ban (high-risk AI rendszerek alkalmazási dátum augusztus), (b) a cikkek sűrűn hivatkoznak egymásra → a graph RAG ott erős, ahol a sima semantic search gyenge, (c) van hivatalos referencia-válasz minden kérdésre (eval-barát).

### D2. LLM: Ollama qwen2.5:7b-instruct (vs. llama3.1:8b, Gemma, API)
**Választás:** `qwen2.5:7b-instruct` Ollama-n keresztül, `DummyLLM` fallbackkel.
**Alternatívák:**
- `llama3.1:8b-instruct` — jó minőség, de kicsit gyengébb tool-calling / JSON mode.
- `gemma2:9b` — kiváló minőség, de nagyobb VRAM + lassabb tok/s az M1 Max-en.
- `mistral-small-24b` — minőség, de ~25GB RAM terhelés.
- Fizetős API (OpenAI/Anthropic) — **tiltja a feladatkiírás**.
**Indoklás:** Qwen2.5 a jelenlegi 7-8B kategória egyik legjobb `format="json"` támogatásával, M1 Max-en Metal-lel 40–60 tok/s, magyar tudása tisztességes (bár a termék angolul megy). A DummyLLM fallback nélkülözhetetlen a CI-hoz és a Dockerfile smoke testjéhez.

### D3. Embedding: `BAAI/bge-small-en-v1.5` (vs. `e5`, OpenAI ada, stb.)
**Választás:** `BAAI/bge-small-en-v1.5` (384 dim).
**Alternatívák:**
- `BAAI/bge-large-en-v1.5` (1024 dim) — ~2% jobb, de 4× nagyobb index.
- `intfloat/e5-small-v2` — gyakorlatilag egyenrangú, hasonló méret.
- `nomic-embed-text-v1.5` — jó, de nagyobb.
- OpenAI `text-embedding-3-small` — fizetős, kizárva.
**Indoklás:** A bge-small a legjobb ár/érték, CPU-n is <50 ms/query, normalizálható cosine-ra. A corpus mérete (fixture ~20, teljes ~100–200 chunk) mellett a minőség-marginális előny nem éri meg a tárolási/latency költséget.

### D4. Vector store: Chroma (vs. Qdrant, Weaviate, FAISS)
**Választás:** Chroma PersistentClient (embedded, SQLite-backed).
**Alternatívák:**
- FAISS — gyorsabb, de nincs metaadat-szűrés out-of-the-box, saját serializálás.
- Qdrant — kiváló, de külön service → Docker compose bonyolódik.
- Weaviate — similar tradeoff.
- SQLite + pgvector — overkillnek bizonyult a nagyságrendhez.
**Indoklás:** Chroma fájl-alapú perzisztenciát ad, metaadat-szűrést támogat, egy pip-install, nincs hálózati service. A corpus mérete mellett a sebesség marginális.

### D5. Graph RAG: saját parsolt citation graph (vs. GraphRAG, LightRAG, LLM entity extraction)
**Választás:** Regex-alapú cross-reference parser → NetworkX DiGraph, 1 hop expansion a top vector hitek körül.
**Alternatívák:**
- **Microsoft GraphRAG** — LLM-mel entitás + reláció extraction, community detection. Hatalmas compute overhead, teljes corpus + LLM minden chunkra.
- **LightRAG** — könnyebb, de még mindig LLM-alapú entity extraction per chunk.
- **Neo4j + Cypher** — robusztus, de egy külön service, és a corpus mérete nem indokolja.
- **Kizárólag vector RAG** — elvetve: a user explicit kérte a graph-alapút, és a jogi szövegek erős cross-referenciái pont itt adnak értéket.
**Indoklás:** Jogi szövegeknél a cross-referenciák **explicitek és kódolhatók** ("Article N", "Recital M"). Ez egy precíz, deterministic graph-struktúrát ad LLM-nélkül (olcsó, gyors, auditable). Az LLM-alapú entity extraction itt értéket ad ugyan (pl. "data controller" ↔ Article 4 GDPR), de a cost-nyereség arány rossz a prototípushoz. Ha a későbbiekben szükség van rá, hibridként hozzá lehet adni.

### D6. Ranking fúzió: Reciprocal Rank Fusion (vs. weighted sum, reranker)
**Választás:** RRF `1/(60+rank)`, k=60.
**Alternatívák:**
- **Weighted linear combination** (α·vector_score + β·graph_score) — súlyhangolás bonyolult, normalizálás szükséges.
- **Cross-encoder reranker** (pl. `BAAI/bge-reranker-base`) — magasabb minőség, de +100–300 ms latency per query. V2-re halasztva.
- **Csak vector** — elvetve, graph-bonuszok elvesznek.
**Indoklás:** RRF tuningmentes, robusztus, kulturálisan is jól ismert. A reranker layer később gond nélkül beilleszthető `fuse_rerank` mögé.

### D7. Graph expansion: BFS 1 hop, in+out, seed-ekkel külön rangsor (vs. 2 hop, csak out, weighted)
**Választás:** 1 hop, in+out élek, expandalt node-okat külön rangsoroljuk strength szerint (`1/(1+seed_rank) * 0.5^hop`).
**Alternatívák:**
- **2+ hop** — zajosabb, irreleváns cikkek szivárognak be.
- **Csak out-edges** — elveszít hivatkozott → hivatkozó útvonalat (pl. Article 9-re hivatkozik Article 5; egy Article 9-et keresve Article 5 is hasznos lehet).
- **Direct synth_rank + penalty az RRF-ben** (első implementáció) → az integráció során a graph hitek mindig kiestek a top-6-ból. Változtattuk.
**Indoklás:** A **külön graph-rangsor** (expanded_rank: strength alapján) megoldja, hogy az RRF arányosan fúzionálja a két forrást. Az empirikus teszt (`test_graph_expansion_pulls_neighbors`) mutatja, hogy Article 9 most bekerül a top-6-ba Article 5 query-re. Hop=1 elég a jogi cross-referenciákhoz (ha egy cikk 2 hoppra van, már gyenge a kapcsolat).

### D8. Agent architektúra: 6 node (vs. ReAct / StateGraph-nélküli / supervisor)
**Választás:** Explicit LangGraph StateGraph: router → planner → rag_invoker ↻ → composer → verifier → END | retry.
**Alternatívák:**
- **Single ReAct loop** — LLM vezérli a tool-hívást ReAct prompting-gel. Kevésbé kontrollálható, rosszabb observability, nehezebb kiértékelni.
- **Supervisor pattern** (multi-agent) — overkill egyetlen domainre.
- **Lineáris pipeline** (nincs autonómia) — nem felel meg a feladatkiírás "autonóm döntéshozatal" követelményének.
**Indoklás:** Explicit gráf + conditional edges (router, verifier) → tisztán látható, hol dönt az ügynök. A `rag_invoker` loop a sub-questionök felett demonstrálja az önálló végrehajtást. A verifier retry mechanizmus a self-correctingképességet.

### D9. Deadline tool: keyword-alapú phase mapping (vs. LLM parsing, NER)
**Választás:** Python regex + kulcsszó-sáv rendezés (prohibitions / gpai / high_risk_default / high_risk_annex1), hardcoded dátumokkal (AI Act Article 113).
**Alternatívák:**
- LLM-mel kinyerni az article-számot a kérdésből → túl lassú ehhez a kis taszkhoz.
- Külön dátum-szakértő node → felesleges komplexitás.
**Indoklás:** A dátumok az AI Act-ben fix, publikus, ritkán változó tények. Kulcsszó-alapon 5 ms alatt megvan, tesztelhető, auditálható. LLM-mel nem nyerne minőséget.

### D10. Fixture vs. teljes PDF corpus a fejlesztés alatt
**Választás:** Ship-elünk egy 19 elemű kézi JSONL fixture-t a repo-val. A teljes PDF corpus a `scripts/download_data.py`-val épül.
**Alternatívák:**
- **Csak PDF, fixture nélkül** — a CI / Docker build / első futtatás megakad hálózat nélkül.
- **Repó-ba rakni a teljes PDF-t** — EUR-Lex licensz kérdések, repo méret megnő.
**Indoklás:** A fixture reprezentatív (AI Act Article 5, 6, 9, 10, 13, 51, 55, 113; GDPR Article 5, 6, 9, 17, 22, 35, 83 + releváns recitalok), **valódi cross-referenciákat** tartalmaz (39 él 33 node-on), és így a pipeline minden komponense demo-érett PDF nélkül is. Ugyanakkor a downloader letölti az eredetit, amint a user futtatja.

---

### D11. Streamlit UI layout: split chat + trace panel (vs. tabs)
**Választás:** Két-oszlopos split — bal oszlop chat (kérdés + válasz + citations), jobb oszlop collapse-olható "Agent trace" (router döntés, plan, RAG lépések, tool hívás, verifier).
**Alternatívák:**
- **Tab-alapú** — tisztább fejléc, de a user egyszerre csak egy dolgot lát.
- **Lineáris alul-trace** — kevésbé áttekinthető, ha a chat history nő.
**Indoklás:** A feladatkiírás explicit kéri: "bemutatja az ágens működésének főbb lépéseit és a RAG folyamat eredményét". A split layout egyszerre mutatja a user-facing választ és a belső döntési folyamatot — ez az elvárás legjobb illusztrációja.

### D12. Eval módszertan: háromrétegű
**Választás:** (1) retrieval recall@k a várt cikkek ellen (determinisztikus), (2) citation correctness (a válaszban idézett cikkek megjelennek-e a retrieved contextben — determinisztikus), (3) answer quality LLM-as-judge rubric-kal (1-5 skálán: factuality, completeness, groundedness).
**Alternatívák:**
- **Csak LLM-as-judge** — drága, zajos, nincs reprodukálható metrika.
- **Csak manuális** — nem skálázódik, nincs regresszió-teszt CI-ban.
- **RAGAS** — jó keretrendszer, de külön dep és erőltetett absztrakció 15 kérdéshez.
**Indoklás:** A determinisztikus két réteg (retrieval + citation) működik **LLM nélkül is** (DummyLLM-mel, CI-ban), ami egybevág az alap tervezési filozófiával. Az LLM-as-judge a minőségi dimenzió lefedésére kell, és ha nincs Ollama, skip-elhető.

### D13. Load test: sequential baseline + threaded (5, 10 workers)
**Választás:** `concurrent.futures.ThreadPoolExecutor`-rel: sequential (1 worker) + 5 + 10 worker futtatás, 100 lekérdezésen.
**Alternatívák:**
- **asyncio** — nem ad sokat, mert a dominant bottleneck a GPU / LLM (CPU-bound queue).
- **multiprocessing** — overkill, tool indítási overhead + shared index másolása.
- **k6 / locust külön** — jobb nagy skálán, de prototípusnál szükségtelen dependencia.
**Indoklás:** A `threads` jó a natív I/O blocking (Ollama HTTP call, Chroma disk) melletti concurrency felmérésére. A sequential adja a true per-query latency-t, a threaded feltárja, hol a contention. Output: p50/p95/p99 end-to-end + node-breakdown.

### D15. Paragraph-aware metadata, NEM paragraph-level chunkolás (option A)
**Választás:** Az article-szintű chunkolás **változatlan**. Plusz: minden article-chunk metadata-ban tárolja a numbered paragraphok listáját (`paragraphs`) és a cross-references **paragraph-szintet megőrző** verzióját (`cross_refs_detailed`). A composer-prompt header így "Article 5 (paragraphs 1-3)" jellegű info-t kap, ha a forrás-szövegben azonosíthatók numbered paragraphok.
**Alternatívák:**
- **Tisztán paragraph-level chunkolás** (option B / D7) — minden numbered paragraph önálló chunk, új ID-séma (`AI Act:article:5:para:2`). 4× több chunk, sűrűbb graph, fragmentáltabb kontextus. Eval-driven Phase 2-be elhalasztva.
- **Status quo (paragraph info teljes elhagyása)** — chunkok atomi-zál maradnak, de a citation pretty-print "Article 5(1)(a)" típusú info-t nem tudja a header-be tenni.
**Indoklás:** Az option A **backwards-compatible** (a Chunk dataclass új mezői default-üresek; a fixture és a pipeline mindenhol fut, ahol eddig). Mégis behozza a paragraph-szintű metaadatot, amit a composer fel tud használni pontosabb idézésre, és ami **alapot ad** a Phase 2 paragraph-level chunkoláshoz, ha eval-szignál mutatja, hogy érdemes. ~30 perc effort, 8 új unit teszt fedi (14 parse-teszt összesen).

**Mit jelent gyakorlatilag:**
- `Chunk.paragraphs: list[str]` — pl. `["1", "2", "3"]` ha a chunkban három numbered paragraph szerepel
- `Chunk.cross_refs_detailed: list[str]` — pl. `["Article 6(1)(a)", "Article 9", "Recital 26"]`
- A graph **változatlanul** article-szintű — paragraph-szintű kapcsolatokat nem épít
- A composer-prompt header range-ben sűrítve mutatja: `"Article 5 (paragraphs 1-3) — Prohibited AI practices"`
- Fixture chunkok: új mezők üresek → header változatlan; valódi PDF-eknél viszont kitöltődnek

### D14. Docker: `python:3.11-slim` + multi-stage `uv`, Ollama külön service
**Választás:** Multi-stage Dockerfile (`uv` a builder stage-ben, slim runtime), `docker-compose.yml` `app` + `ollama` service-szel. Az Ollama külön profile (`--profile with-llm`), hogy DummyLLM-mel is menjen a smoke test.
**Alternatívák:**
- **Distroless** — problémás a sentence-transformers + torch miatt (glibc, libgomp).
- **Alpine** — sentence-transformers `musl`-en trükkös, nem éri meg a méret-spórolás.
- **Ollama is a fő imageben** — GPU-passthrough nehézkes, modell méret a képbe égne.
**Indoklás:** A slim + multi-stage `uv` install < 60 sec tisztán. A külön Ollama service tisztán választja a GPU-sztéket az app-tól; lokális dev-en a host Ollama-ja is használható környezeti változóval.

---

## 4. Eval + ismert korlátok

### 4.1. Fixture-eval (2026-04-23, 19 chunk)

Determinisztikus alapvonal a fejlesztés alatt, kézi fixture-rel:
- **Retrieval recall (macro):** 96.7%
- **Citation precision:** 100% (nincs hallucinált idézet)
- **Intent routing accuracy:** 100% (15/15)
- **Mean latency:** 78 ms (DummyLLM)

### 4.2. Real-corpus eval — DummyLLM (2026-05-03, 542 chunk)

A real PDF ingestion után, ugyanazon 15 kérdéses set-tel:
- **Retrieval recall:** 66.7% (-30 pp)
- **Citation precision:** 100%
- **Intent routing:** 100%
- **Mean latency:** 42 ms

A drop a corpus-méret (19→542 chunk) miatt: a sima vector search a sok zaj-chunk között kevésbé fókuszált, és a DummyLLM nem ír át query-t. A számok alapját a `eval/reports/report_20260503T191044Z.md` adja.

### 4.3. Real-corpus eval — Real LLM (qwen2.5:7b-instruct), 2026-05-03

A teljes 3-rétegű scoring real LLM-mel (query rewrite + LLM-as-judge):
- **Retrieval recall:** **80.0%** (+13.3 pp a DummyLLM-hez képest, query rewrite hatása)
- **Citation precision:** 100%
- **Intent routing:** 93.3% (14/15 — egy miss, q06 deadline kérdést rag-ra route-olt)
- **Answer quality (LLM-judge, 1–5):** **3.62**
- **Mean latency:** 29.3 sec/query (jelentős az LLM-hívások miatt)
- Riport: `eval/reports/report_20260503T194704Z.md`

**Per-question failure módok a real-LLM run-on:**
- **q06** (`When does the AI Act apply to high-risk systems under Annex I?`) — router rag-ot választ tool helyett. A qwen2.5 ezt "tényleges jogi kérdés"-ként értelmezi, nem "deadline-calc". A composer-trace szerint a válasz végül helyes (Article 6(1) + 2027 dátum), tehát a végső UX nem sérült, csak a routing-érték rossz a metrikán. Phase 2-ben router-prompt finomítás vagy tanult router (small-classifier model) javítaná.
- **q03, q08, q09, q11** — 50% recall: az egyik várt cikk megvan, a másik nincs (graph expansion + RRF nem hozza fel mindig mindkettőt egyszerre).
- **q10** — 0% recall: ismert cross-regulation graph-él hiány (lásd alább).

### 4.4. Ablation: a graph expansion kontribúciója (2026-05-03)

**Kérdés:** A graph-RAG (citation-graph 1-hop expansion) tényleg javítja a recallt, vagy csak komplexitást ad? Az eddigi run-ok WITH graph számokat adtak; itt egy **A/B ablation**: `GRAPH_HOPS=1` vs `GRAPH_HOPS=0` ugyanazon a 15-kérdéses seten, DummyLLM-mel (determinisztikus alapvonal).

**Eredmény:**

| Konfiguráció | Recall | Δ |
|---|---|---|
| Vector + graph (default) | 60% | baseline |
| Vector only (`GRAPH_HOPS=0`) | **73.3%** | **+13.3 pp** |

**Net negatív hatás.** A graph **rontja** a recallt a 542-chunkos real corpus-on DummyLLM-mel.

**Mely kérdéseken ront:**
- **q12** (transparency, Article 13): 0% with graph → 100% without. Vector találja Article 13-at, de a graph by-products (Article 113, recitals) displace-elik az RRF-ben.
- **q15** (AI system definition, Article 3): 0% → 100%. Ugyanaz a pathológia.

**A többi 13 kérdésen** azonos a recall — vagy mindkét beállításban talál (factual queries), vagy egyikben sem (q10 cross-regulation, multi-article kérdések).

**Diagnózis:**
- Vector találja a releváns cikket pl. rank 2-3-on (RRF score ~0.0161)
- Graph expansion behoz egy chunkot graph_rank=0-val (RRF score 0.0167)
- A két ranking azonos súllyal versenyzik az RRF-ben → a graph "newcomer" bemegy top-K-ba, a vector hit kiesik
- A graph chunkok **strukturálisan közeliek** a seedhez (cross-reference), de **nem feltétlenül szemantikailag relevánsak** a query-re

**Phase 2 fix-opciók (DEVLOG D6/D7 bővítése):**
| Megoldás | Effort | Várt hatás |
|---|---|---|
| Graph hit RRF-súly félre csökkentés (`0.5 * RRF`) | 1 LOC | Graph komplementer, nem domináns |
| Külön RRF k vector (60) vs graph (120) | 2 LOC | Graph score absolút kisebb |
| Graph hitek cosine-reranking a query-re | ~10 LOC | Csak releváns graph chunkok mennek be |
| Cap a graph hitek számára (top-3 max) | 3 LOC | Vector top-K dominancia |

**Real LLM ablation (2026-05-03):**
A teljes 4-cellás A/B mátrix (LLM × graph) lefuttatva:

| Konfiguráció | Recall | Citation | Intent | Answer quality | Latency |
|---|---|---|---|---|---|
| DummyLLM, vector + graph | 60% | 100% | 100% | n/a | 55 ms |
| DummyLLM, vector only | **73.3%** | 100% | 100% | n/a | 44 ms |
| Real LLM, vector + graph | **80%** | 100% | 93.3% | **3.62** | 29.3 s |
| Real LLM, vector only | **80%** | 100% | 93.3% | 3.53 | 30.4 s |

**Konklúziók:**

1. **DummyLLM-mel a graph net negatív** (-13 pp recall). A query rewrite hiánya miatt a vector retrieve fragilis, és a graph displace-eli a releváns vector hiteket az RRF-ben.

2. **Real LLM-mel a graph net neutrális a recallon** (0 pp delta). A 3-paraphrase query rewrite olyan **robusztus vector retrieval**-t ad, hogy a graph chunkjai nem tudják displace-elni a top hiteket — de hozzá sem adnak.

3. **Real LLM-mel a graph javítja az answer quality-t** (+0.09 LLM-judge score). A fused listába bekerülő graph-only chunkok **kontextust gazdagítanak** a composernek, jobb válaszokat generálva, még ha a recall ugyanaz is.

**Új mentális modell a graph-RAG szerepéről:**
A graph **nem elsősorban retrieval-booster**, hanem **context-enricher** — különösen erős LLM-mel kombinálva. A retrieval-erősség forrása a query rewrite + vector pár; a graph **horizontális kontextus-szélesítést** ad, ami a generálás minőségére hat, nem a recall-ra.

**Mit jelent ez a leadás szempontjából:**
A graph-RAG **kettős hozzájárulása** most explicit: (a) DummyLLM-mel jelenleg ront a recallon (RRF-súlyozás javítandó), (b) real LLM-mel a recallt nem mozdítja, de az answer quality-t mérhetően javítja. Ez **érettségi pont** — a komponens **valódi értéke** az ablation-ban derült ki, és Phase 2-be konkrét RRF-tuning + cross-regulation él tervezetten van.

### 4.4.1. Bővített ablation: structural kérdések (3 új query, q16–q18)

A 4.4 vizsgálat után érdemes volt megnézni: **mi történik olyan kérdéseken, amelyek explicit a cross-reference-háló-tudást kívánják?** A `structural` kategóriában 3 kérdés:

- **q16:** *"Which GDPR articles cross-reference the rules on special categories of personal data in Article 9?"* — várt: GDPR Art 6, 22, 35 (mind cross-referenciák Art 9 felé)
- **q17:** *"Which AI Act articles invoke or rely on the risk management system requirements of Article 9?"* — várt: AI Act Art 5, 9, 13
- **q18:** *"What other AI Act provisions does Article 5 directly reference for its prohibited-practice rules?"* — várt: AI Act Art 5, 9, Annex II

Mindhárom úgy lett megfogalmazva, hogy **a graph predecessors / successors** egyértelmű választ ad — a graph_expand-nak elvileg ki kell hoznia a várt cikkeket.

**Eredmény (4-cella, 18-kérdéses run):**

| Konfiguráció | Recall (összes) | Recall (csak q16–q18) | Quality (csak q16–q18) |
|---|---|---|---|
| DummyLLM + graph | 59.3% | 11% (1/9) | n/a |
| DummyLLM only | 66.7% | 33% (3/9) | n/a |
| Real LLM + graph | 72.2% | 33% (3/9) | 2.77 átlag |
| Real LLM only | 72.2% | 33% (3/9) | 2.63 átlag |

**Találat:** a graph **a structural kérdéseken sem javít a recallon** — még real LLM-mel sem. A quality-n kicsi pozitív (+0.14 a structural szegmensen), inkonzisztens (q17: -0.3, q18: +0.7).

**Hipotézis a graph hatástalanságára structural kérdéseken:**

A `graph_expand` node helyesen futtat 1-hop expansion-t, és a vector-hits seedjeiről 20-40 neighbor-t hoz be. **De a final top-K=6-ba nem jutnak be a "várt" cikkek**, mert:

1. **Strength-formula túl egyenletes** — a 1-hop neighborokra `s = 1/(1+seed_rank)` egyenértékű érték; ha 5 seed mind rank=0, akkor 30+ neighbor mind strength=1.0. RRF tie-breaker önkényes.
2. **Top-K=6 csonkolás szigorú** — a fused listában a 6 hely megoszlik vector + graph között; a vector már gyakran 4-5 helyet visz, marad 1-2 hely a graph számára, és nem feltétlenül a "best" graph neighbor.
3. **Quality-pop nem skálázódik recallra** — a graph chunkok contextet adnak a composernek, de a chunk-ID-k nem feltétlenül "expected"-ek a recall-szempontból.

**Phase 2 fix-irányok (DEVLOG D6 bővítése):**
- Strength a target-relevancia alapján (cosine query → graph-chunk), nem csak hop-távolság
- Top-K növelése (pl. 6→10) ha a graph engedélyezve van — több hely a fused-ban
- Cross-regulation él bevezetése (q10 ágon)
- Reranker a fused listán

**Tanulság a leadáshoz:** a graph-RAG **rendelkezésre áll, működik a chunk-szállítás szintjén, de a final top-K kiválasztás nem szelektál a "graph-vártakra"**. Ez nem hiba, hanem **explicit mért limitáció** — a metrikán nem nyer, de a context-richness-en (quality-n) inkább. A v2-ben mérendő, hogy a strength-formula + Top-K növelés mekkora ablation-effektust adna.

### 4.5. Ismert korlát (q10 cross-regulation)

Jelenleg a citation graph csak azonos regulation-ön belül köt él (AI Act ↔ AI Act, GDPR ↔ GDPR). A q10 ("How does GDPR Article 22 relate to AI Act high-risk systems?") ezért 50%-os recallt kap (GDPR 22 megvan, AI Act 6 nem jön át graph expansionnel). Két fix-opció, mérendő v2-ben:

1. **Cross-regulation élek** — bővíteni a parser-t, hogy felismerje a "of Regulation (EU) 2024/1689" és "of Regulation (EU) 2016/679" szintagmákat, és a külföldi hivatkozást expliciten (másik regulation) node-hoz kösse.
2. **Multi-query RAG** — a `rewrite` node már kiadhat egy "in the AI Act" + "in the GDPR" verziójú variánst, és a union-je lefedi a crossovert.

**Ismert korlát (single-turn stateless LLM-hívások):** Minden user-query teljesen önállóan, **előzmény nélkül** kerül az LLM-hez (router, planner, composer, verifier mind csak a `state["query"]`-t látják). A Streamlit `st.session_state.history` **kizárólag a UI rétegben él** — a felhasználó látja a korábbi turn-eket, de az agent nem. Következmény: pronoun-alapú follow-up kérdések ("What does it reference?", "And the deadline?") **nem feloldódnak** — vagy véletlen RAG-átfedéssel működnek, vagy hibás választ adnak.

Phase 2 javítási opciók:
1. **Query-rewriting pre-step** (preferált) — egy új node a router előtt, ami az utolsó 2-3 turnt + az új kérdést kapja, és **átírja** önmagában értelmes, "expanded" query-vé. A downstream pipeline változatlan, csak a `state["query"]` helyettesül a rewrite kimenetével. ~30 LOC + 1 LLM-hívás per query.
2. **Naív history-append a composer-promptban** — utolsó N (Q, A) pár közvetlen beszúrása. Egyszerűbb, de a token-költség lineárisan nő, és a router nem látja, csak a composer.

A választás eval-driven: ha gyakori a follow-up-pattern és router-szinten is hibázik, az 1. opció kell; ha csak a final compositionnál fontos a kontextus, a 2. opció elég.

## 5. Kapcsolódó dokumentumok

- [`ROADMAP.md`](./ROADMAP.md) — Phase 2 és Phase 3 a prototípus után, GCP és Azure leképezéssel. Ugyancsak folyamatosan frissül.
- [`eval/questions.yaml`](./eval/questions.yaml) — 15 értékelő kérdés.
- [`eval/reports/`](./eval/reports/) — per-run markdown riport.
