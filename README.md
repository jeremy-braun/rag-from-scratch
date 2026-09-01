# RAG from Scratch

Vollständiges Retrieval-Augmented Generation System — ohne LangChain oder andere Abstraktionsschichten. Jede Komponente selbst implementiert: Chunking, Embedding, Vektorspeicherung, Retrieval, LLM-Generierung und Groundedness-Evaluation.

**Wissenschaftliche Basis:** Lewis et al. 2020 — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* ([arXiv:2005.11401](https://arxiv.org/abs/2005.11401))

---

## Warum from Scratch

LangChain kann jeder in 20 Minuten aufsetzen. Was dabei verloren geht: das Verständnis was darunter passiert.

Dieses Projekt implementiert jede Komponente manuell — nicht weil Frameworks schlecht sind, sondern weil man nur so erklären kann warum Cosine Similarity als Distanzmetrik funktioniert, was Chunk Overlap löst, oder warum ein LLM ohne RAG bei domänenspezifischen Fragen halluziniert.

---

## Architektur

```
[PDF-Dokumente]
      ↓
[chunking.py]     — Text in überlappende Chunks aufteilen
      ↓
[embedder.py]     — Chunks → 384-dim Vektoren (all-MiniLM-L6-v2)
      ↓
[ChromaDB]        — Vektoren persistent speichern (Cosine Space, HNSW)
      ↓
[Query]
      ↓
[retriever.py]    — Query → Vektor → Top-K Chunks per Cosine Similarity
      ↓
[rag_pipeline.py] — Chunks + Query → Prompt → llama3.1:8b
      ↓
[Antwort mit Grounding im Kontext]
      ↓
[evaluate.py]     — Groundedness-Check per LLM-as-Judge + manueller Review
```

---

## Tech Stack

| Komponente | Tool | Begründung |
|---|---|---|
| Embeddings | `sentence-transformers` / `all-MiniLM-L6-v2` | Lokal, kostenlos, 384-dim, ~90 MB |
| Vektordatenbank | `ChromaDB` | Lokal, persistent, HNSW intern |
| LLM | `llama3.1:8b` via Ollama | Lokal, kostenlos, keine API-Abhängigkeit |
| PDF-Parsing | `pypdf` | Lightweight, keine externe Abhängigkeit |

Vollständig lokal und kostenlos — keine OpenAI API, keine Cloud-Kosten.

---

## Schnellstart

**Voraussetzungen:** Python 3.13, [Ollama](https://ollama.com) installiert

```bash
# Repository klonen
git clone https://github.com/jeremybraun/AI-902.git
cd AI-902/rag-from-scratch

# Virtuelle Umgebung
python3 -m venv venv
source venv/bin/activate

# Abhängigkeiten
pip install sentence-transformers chromadb tiktoken numpy jupyter ollama pypdf

# LLM herunterladen
ollama pull llama3.1:8b
```

**Eigene Dokumente indexieren:**

1. PDFs in `examples/papers/` ablegen
2. In `src/rag_pipeline.py` `modus = "index"` setzen
3. Ausführen:

```bash
python src/rag_pipeline.py
```

**Query stellen:**

1. `modus = "query"` setzen, Frage anpassen
2. Ausführen:

```bash
python src/rag_pipeline.py
```

**Evaluation:**

```bash
python src/evaluate.py
```

---

## Komponenten im Detail

### `chunking.py`
Zwei Chunking-Strategien mit konfigurierbarem Overlap:

- **Fixed-size:** Text nach Wortanzahl aufteilen. Schnell, vorhersehbar, aber ignoriert Satzgrenzen.
- **Sentence-based:** Text nach Sätzen aufteilen. Respektiert natürliche Sprachgrenzen, erzeugt variable Chunk-Längen.

Overlap existiert weil relevante Information an Chunk-Grenzen sonst verloren geht. Ein Gedanke der über zwei Chunks verteilt ist, ist in beiden Chunks durch Overlap vorhanden.

### `embedder.py`
Lädt `all-MiniLM-L6-v2` und wandelt Chunks per Batch-Processing in 384-dimensionale Vektoren um. Speichert Text, Vektor und ID in ChromaDB mit Cosine Similarity als Distanzmetrik. Läuft vollständig auf CPU — keine GPU nötig.

### `retriever.py`
Wandelt die Query als Ganzes (kein Chunking) in einen Vektor um und sucht per HNSW Approximate Nearest Neighbor die nächsten Nachbarn im Vektorraum. Kombiniert Top-K als Obergrenze mit Score-Threshold als Relevanzfilter. Standardwerte: `n_results=5`, `distance_threshold=0.8` (empirisch für wissenschaftliche Paper kalibriert).

### `rag_pipeline.py`
Zwei Modi:

- **index:** Liest alle PDFs aus `examples/papers/`, chunked, embedded und speichert in ChromaDB. Einmalig ausführen.
- **query:** Retrievet relevante Chunks, baut Prompt mit expliziter Grounding-Instruktion, ruft llama3.1:8b via Ollama auf.

Der System-Prompt erzwingt Groundedness: das LLM wird instruiert ausschließlich auf Basis des Kontexts zu antworten und parametrisches Wissen nicht hinzuzufügen.

### `evaluate.py`
Groundedness-Evaluation per LLM-as-Judge: llama3.1:8b bewertet seine eigene Ausgabe gegen die übergebenen Chunks und gibt einen Score zwischen 0.0 und 1.0 plus Begründung zurück. Ergänzt durch manuelles Human Review weil kleine Modelle bei Selbstbewertung zu nachsichtig tendieren.

---

## Wissensbasis

Die Wissensbasis besteht aus drei wissenschaftlichen Paper zu Child-AI Co-Creative Storytelling:

- Zhang et al. (2022) — *StoryDrawer: A Child–AI Collaborative Drawing System*. CHI 2022.
- Fan et al. (2025) — *From Words to Wonder: Designing and Evaluating an AI-Empowered Creative Storytelling System*. IDC 2025.
- CHI 2026 — *Towards Understanding Children's Collaborative Interaction Patterns in Child-AI Co-creative Interfaces*.

Domänenspezifischer Inhalt den llama3.1:8b ohne RAG nicht beantworten kann — genau der Anwendungsfall für den RAG konzipiert ist.

---

## Evaluation

Groundedness wurde auf drei Testfragen evaluiert:

| Frage | LLM-Judge Score | Manueller Score |
|---|---|---|
| Design principles for child-AI storytelling | 0.5 | 0.7 |
| Children's interaction with AI vs. human partners | 0.5 | 0.4 |
| Limitations of child-AI co-creative systems | 0.5 | 0.3 |

**Beobachtung:** llama3.1:8b tendiert zur Selbstbewertung von 0.5 unabhängig vom tatsächlichen Grounding — ein bekanntes Problem bei kleinen Modellen. Das manuelle Review zeigt eine grössere Varianz die das Modell nicht korrekt abbildet.

---

## Limitations & Production Considerations

**Bekannte Limitationen dieser Implementierung:**

**PDF-Tabellen nicht abrufbar.** pypdf extrahiert tabellarischen Content nicht oder fehlerhaft. Spezifische Statistiken in Tabellen (z.B. "3.42 kreative Blockaden pro Session") sind im Index nicht vorhanden und damit nicht retrievebar. Verifiziert durch direkten ChromaDB-Query.

**Sprachinkonsistenz.** llama3.1:8b wechselt zwischen Deutsch und Englisch abhängig von Query-Sprache und Chunk-Sprache. In Produktion: Query-Sprache normalisieren oder Prompt-Sprache explizit vorgeben.

**Selbstbewertungsbias.** LLM-as-Judge mit demselben Modell das die Antwort generiert hat ist inhärent problematisch. In Produktion: stärkeres Judge-Modell oder regelbasierte Groundedness-Prüfung per Textüberlappung als Baseline.

**Distance Threshold empirisch gesetzt.** 0.8 wurde durch manuelle Tests für wissenschaftliche Paper kalibriert. Für anderen Content-Typ (z.B. Prosa, Code) muss der Threshold neu kalibriert werden.

**Was in Produktion anders wäre:**

- **Hybrid Search:** BM25 (Keyword) + Dense Retrieval kombiniert per Reciprocal Rank Fusion — robuster bei spezifischen Begriffen und Eigennamen
- **Cross-Encoder Reranking:** zweistufiges Retrieval — bi-encoder für Kandidaten, cross-encoder für präzises Re-Ranking der Top-K
- **Query Rewriting:** HyDE (Hypothetical Document Embeddings) oder Query Expansion um semantische Lücken zwischen Query und Dokumentsprache zu schliessen
- **PDF-Parsing:** `pymupdf` statt `pypdf` für bessere Tabellen- und Layoutextraktion
- **Retrieval Quality Gating:** automatische Prüfung ob retrievte Chunks überhaupt relevant genug sind bevor der LLM-Call gemacht wird

---

## Verwandte Projekte

[ReAct Agent Reimplementation](../react-reimplementation/) — eigene Implementierung des Reasoning+Acting Loop (Yao et al. 2022) ohne Agent-Frameworks.
