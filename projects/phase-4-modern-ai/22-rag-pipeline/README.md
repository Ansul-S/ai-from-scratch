# Project 22 — RAG Pipeline from Scratch
### Daedalus: Grounded Question Answering over a Notebook Corpus

> **Difficulty:** 🔴 Hard
> **Time to complete:** ~15–20 hours
> **Phase:** Phase 4 — Modern AI & Applied LLMs
> **Tags:** `rag` `embeddings` `vector-search` `qdrant` `bge-m3` `cross-encoder-reranking` `cosine-similarity` `chunking` `grounded-generation` `ollama` `qwen3`

---

## 📌 What You'll Learn

By the end of this project you will be able to:

- [ ] Explain what RAG actually solves — and why fine-tuning is the wrong tool for "the model doesn't know my documents"
- [ ] Build a structure-aware ingestion pipeline that treats code and prose differently, and explain why that distinction decides retrieval quality
- [ ] Choose `max_words` and `sentence_overlap` from a measured sweep instead of copying a blog post
- [ ] Explain what a text embedding is, why cosine similarity is the right comparison, and why normalising vectors turns cosine into a dot product
- [ ] Index 230 vectors in Qdrant with metadata payloads rich enough to cite the exact source cell
- [ ] Explain the bi-encoder vs cross-encoder tradeoff and why a two-stage retrieve → rerank pipeline beats either one alone
- [ ] Write a grounding prompt that stops an LLM from answering a question in the wrong domain entirely
- [ ] Separate **retrieval failure** from **generation failure** when an answer comes back wrong — the single most useful debugging skill in applied LLM work
- [ ] Write 11 pytest unit tests that catch corpus corruption before a single vector is created
- [ ] Explain every design decision in a technical interview

---

## 🆕 New Concepts vs Phase 1

This is the first project where the "model" is not something you train. Nothing here has weights you fit — the intelligence is in **what you retrieve** and **what you put in the prompt**.

| Concept | Projects 01–04 (Classical ML) | Project 22 (RAG) |
|---------|------------------------------|------------------|
| What you build | A model with learned parameters | A pipeline with pretrained components |
| Training | Gradient descent over epochs | None — indexing replaces training |
| "Fitting" | `w ← w − α∇w` | Embed the corpus once, store the vectors |
| Input | A numeric feature matrix | Free text: markdown and Python source |
| The hard part | Optimisation and regularisation | Chunking, retrieval quality, grounding |
| Error metric | Accuracy, R², F1 | Was the right chunk retrieved? Did the answer stay grounded? |
| Hyperparameters | learning rate, degree, k | `max_words`, `sentence_overlap`, `top_k`, `final_top_k`, temperature |
| Failure mode | Overfitting / underfitting | Retrieval failure vs generation failure |
| What "wrong" looks like | A number that's off | A fluent, confident, completely fabricated paragraph |
| Cost of a mistake | A bad prediction | A bad prediction *that sounds authoritative* |

> The uncomfortable shift: in Phase 1, a broken model looks broken. In Phase 4, a broken system produces beautiful prose. Every safeguard in this project exists because of that.

---

## 📦 The Corpus — Our "Dataset"

**Source:** `corpus/Retrival_Ext_QnA_Complete.ipynb` — a 223-cell teaching notebook on retrieval-based document QnA (embeddings, FAISS, BERT readers).
**Why a notebook and not a PDF:** a notebook has *typed* cells. That type — markdown or code — is free, high-quality structural metadata, and using it is what separates a corpus that retrieves well from one that doesn't.

### Corpus Composition

| Property | Value |
|----------|-------|
| Total cells | 223 |
| Markdown cells | 118 (52.9%) |
| Code cells | 105 (47.1%) |
| Total characters | 149,049 |
| Words per cell — median | 65 |
| Words per cell — mean | 75.1 |
| Longest cell | 536 words (code) |
| Cells above the 250-word budget | 6 |
| Cells under 20 words | 22 |

### What This Corpus Is Good For

- **It is nearly half code.** A code chunk that loses its indentation is worthless as evidence — this corpus forces the cleaning pipeline to be correct rather than convenient.
- **It is a teaching document.** Cells are short, self-contained and topical, so cell boundaries are genuine semantic boundaries.
- **It is a domain the retriever can be graded on.** You know the material, so you can tell whether a retrieved chunk is actually relevant — the hardest thing to automate in RAG evaluation.

> ⚠️ **Corpus Reality Check:** This corpus is a well-structured, single-author notebook. Real corpora are 40,000 PDFs with broken OCR, three versions of the same policy document, tables that extract as word salad, and scanned pages with no text layer at all. In production, corpus preparation *is* the project — model selection is an afternoon. This project teaches the pipeline; real projects teach data wrangling.

---

## 💡 Intuition First

### The Problem RAG Solves

You have an LLM. It is excellent at language and knows nothing about *your* documents. Three ways to fix that:

**1. Fine-tune the model on your documents.** Expensive, slow, and wrong for this job. Fine-tuning teaches *behaviour and style*, not facts. Facts baked into weights cannot be updated without retraining, cannot be cited, and cannot be deleted when a document is retracted. Worse: the model will still confidently answer about documents you never gave it.

**2. Paste everything into the prompt.** Works until it doesn't. A 149,049-character corpus is roughly 40k tokens — that fits in a modern context window, but 40,000 documents do not. And even when it fits, the model's attention is diluted across mostly irrelevant text: performance on a specific question is *better* with 5 relevant chunks than with the whole corpus.

**3. Retrieve first, then generate.** Search the corpus for the handful of passages that actually answer this question, hand only those to the model, and instruct it to answer strictly from them. That is RAG.

### The Real-World Analogy

An open-book exam.

A student who memorised the textbook (fine-tuning) can be out of date and can't show you where an answer came from. A student handed the entire library (long context) wastes the exam flipping pages. A student who knows how to find the three right pages, reads them, and answers with the book open (RAG) — that student is fast, current, and can point at the page.

**The librarian is the retriever. The student is the generator. Most RAG systems fail because the librarian brought the wrong book — not because the student is stupid.**

### Why Two Retrieval Stages

Vector search compares a question to 230 chunks *cheaply*, because each chunk was turned into a vector once, at index time. The price of that speed is precision: the question and the chunk were never actually read together — just compared as two points in space.

A cross-encoder reads the pair — question *and* chunk, together, in one forward pass — and judges relevance properly. That is far more accurate and far too slow to run over an entire corpus.

So you use both: the fast, imprecise model shortlists 10 candidates; the slow, precise model ranks those 10 and keeps 5.

> **Interview-ready line:** *"Retrieval is a funnel. Each stage is more expensive and more accurate than the last, and each stage only sees what the previous one shortlisted."*

---

## 🔢 How It Works — The Math

### 1. Text Embeddings

An embedding model $E(\cdot)$ maps text to a fixed-length vector:

$$\mathbf{v} = E(\text{text}), \qquad \mathbf{v} \in \mathbb{R}^{1024}$$

BGE-M3 produces 1024 dimensions. The individual numbers mean nothing to a human; what matters is *relative position*: texts with similar meaning land near each other, even when they share no words.

That last part is the whole point. Keyword search for "disadvantages" misses a passage that says "limitations". Their embeddings are neighbours.

### 2. Cosine Similarity

Relevance is measured by the angle between the question vector and the chunk vector:

$$\text{sim}(\mathbf{q}, \mathbf{c}) = \cos\theta = \frac{\mathbf{q} \cdot \mathbf{c}}{\|\mathbf{q}\| \, \|\mathbf{c}\|} \in [-1, 1]$$

**Why the angle and not the distance?** Euclidean distance is dominated by vector *magnitude*, and magnitude in embedding space tracks things like text length, not meaning. A 30-word chunk and a 200-word chunk about the same topic should score alike. The angle ignores length; the distance does not.

### 3. Normalisation Makes Cosine a Dot Product

If every vector is scaled to unit length ($\|\mathbf{v}\| = 1$), the denominator disappears:

$$\text{sim}(\mathbf{q}, \mathbf{c}) = \mathbf{q} \cdot \mathbf{c}$$

This is why the pipeline passes `normalize_embeddings=True` and then verifies the norms are exactly 1.0. Cosine similarity across the whole index collapses into one matrix multiply — and the vector store can use the cheaper operation internally.

### 4. Top-k Retrieval

$$\text{Retrieve}(q, D) = \underset{c \in D}{\arg\text{top-}k} \; \big( E(q) \cdot E(c) \big)$$

The cost model is the entire argument for RAG. With $N$ chunks and a reader that costs $T$ per chunk:

$$T_{\text{brute}} = N \times T_{\text{reader}} \qquad\text{vs}\qquad T_{\text{rag}} = T_{\text{search}} + k \times T_{\text{reader}}$$

$T_{\text{search}}$ grows sub-linearly in $N$ with an approximate index, and $k$ is a constant you choose. Brute force scales with the corpus. RAG does not.

### 5. Bi-encoder vs Cross-encoder

| | Bi-encoder (BGE-M3) | Cross-encoder (bge-reranker-v2-m3) |
|---|---|---|
| Input | question **or** chunk | question **and** chunk together |
| Score | $E(q) \cdot E(c)$ | $f(q, c)$ — one forward pass per pair |
| Chunk vectors | precomputed at index time | impossible — depends on the question |
| Cost per query | 1 encode + a vector search | one model call **per candidate** |
| Measured here | 230 chunks embedded in 50.17s (once) | 10 pairs scored in 5.11s (every query) |
| Accuracy | good | better |

The cross-encoder cannot precompute anything, because its representation of a chunk depends on the question. That is exactly why it is more accurate — and why it can only ever see a shortlist.

### 6. The Math-to-Code Table

| Concept | Code |
|---------|------|
| $\mathbf{v} = E(\text{chunk})$ | `model.encode(chunk_texts, convert_to_numpy=True)` |
| $\|\mathbf{v}\| = 1$ | `normalize_embeddings=True` |
| store as float32 | `chunk_embeddings.astype("float32")` |
| $\cos\theta$ metric | `VectorParams(size=1024, distance=Distance.COSINE)` |
| $\mathbf{q} = E(\text{question})$ | `embedder.encode_query(question)` |
| $\arg\text{top-}k \; \mathbf{q}\cdot\mathbf{c}$ | `client.query_points(query=query_vector, limit=top_k)` |
| $f(q, c)$ cross-encoder score | `reranker.predict([[question, chunk_text], ...])` |
| rank by reranker score | `scored_chunks.sort(key=..., reverse=True)` |
| evidence block | `build_evidence_context(reranked_chunks)` |
| grounded prompt | `build_prompt(question, evidence_context)` |
| deterministic answer | `"options": {"temperature": 0.1}` |

---

## 🔄 Pipeline Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      INDEX TIME  (run once per corpus)                   │
│                                                                          │
│  corpus/*.ipynb  ──►  NotebookLoader  ──►  TextCleaner  ──►  Chunker     │
│  223 cells            cell + type          code ≠ prose      cell-aware  │
│  149,049 chars        metadata kept        fences stashed    250 words   │
│                                                              overlap 2   │
│                                                                   │      │
│                                                                   ▼      │
│                          ChunkEmbedder (BAAI/bge-m3)                     │
│                     ┌──────────────────────────────────────┐             │
│                     │  230 chunks → (230, 1024) float32    │             │
│                     │  normalized → every ‖v‖ = 1.0        │             │
│                     │  50.17 s on CPU                      │             │
│                     └──────────────────────────────────────┘             │
│                                                                   │      │
│                                                                   ▼      │
│                     QdrantIndex — collection "daedalus_chunks"           │
│                     230 points · COSINE · payload = full metadata        │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                      QUERY TIME  (run per question)                      │
│                                                                          │
│  question ──► clean ──► encode_query ──► Qdrant search (top_k = 10)      │
│                         same model!      cosine, ~ms                     │
│                                                    │                     │
│                                                    ▼                     │
│                     ChunkReranker (bge-reranker-v2-m3)                   │
│                     10 (question, chunk) pairs → scores → keep 5         │
│                     5.11 s on CPU                                        │
│                                                    │                     │
│                                                    ▼                     │
│                     build_evidence_context(5 chunks)                     │
│                     [Source: ...] [Cell: 54] [Cell Type: markdown]       │
│                                                    │                     │
│                                                    ▼                     │
│                     build_prompt(question, evidence)  ← 8 grounding rules│
│                                                    │                     │
│                                                    ▼                     │
│                     Ollama · qwen3:8b · temperature 0.1                  │
│                     8.17 s → grounded answer + citable evidence          │
└──────────────────────────────────────────────────────────────────────────┘
```

**The three rules visible in this diagram:**
1. The **same** embedding model encodes chunks and questions. Different models = two unrelated coordinate systems = meaningless scores.
2. `top_k` (10) is always larger than `final_top_k` (5). The reranker exists to *reorder*; give it nothing to choose from and it does nothing.
3. Question and evidence are built together, in one function call, from the same `question` argument. They can never drift apart.

---

## 🛠️ Prerequisites & Structure

### Install Dependencies

```bash
pip install -r requirements.txt
```

The generation stage talks to a local [Ollama](https://ollama.com) server:

```bash
ollama pull qwen3:8b
ollama serve                 # http://localhost:11434
```

Embedding and reranking weights download from Hugging Face on first run (BGE-M3 ≈ 2.2 GB, reranker ≈ 2.2 GB). Pass `--local-files-only` to load them from the local cache and never touch the network.

### Project Directory

```
22-rag-pipeline/
├── corpus/
│   └── Retrival_Ext_QnA_Complete.ipynb   ← the study material (223 cells)
├── daedalus_qdrant/                      ← local vector store (gitignored)
├── results/
│   ├── cell_type_distribution.png
│   ├── cell_word_counts.png
│   ├── cleaning_impact.png
│   ├── chunk_size_distribution.png
│   └── chunking_hyperparameter_sweep.png
├── tests/
│   ├── __init__.py
│   └── test_solution.py                  ← 11 tests, no models needed
├── 01_eda.ipynb                          ← corpus exploration + chunking sweep
├── 02_implementation.ipynb               ← the full pipeline, built step by step
├── solution.py                           ← all classes live here
├── train.py                              ← CLI entry point
├── requirements.txt
└── README.md
```

> ⚠️ **`daedalus_qdrant/` is a build artifact, not source.** It is regenerated by `python train.py` in about a minute and is gitignored. Committing a vector store means committing a binary blob that goes stale the moment the chunker changes.

---

## 🏗️ Build From Scratch

### Step 1 — Load the Corpus (`NotebookLoader`)

A notebook is an ordered list of typed cells. Flattening it to a string throws away the two most valuable pieces of metadata you will ever get for free: **where** a passage lives and **what kind** of content it is.

```python
class NotebookLoader:
    def load(self, verbose: bool = True) -> List[Dict[str, Any]]:
        import nbformat

        with open(self.notebook_path, "r", encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=4)

        cells_data = []
        for cell_index, cell in enumerate(notebook.cells):
            text = cell.source
            cells_data.append({
                "source": Path(self.notebook_path).name,
                "source_type": "ipynb",
                "cell_number": cell_index + 1,
                "cell_type": cell.cell_type,      # markdown / code / raw
                "text": text,
                "word_count": len(text.split()),
                "character_count": len(text),
            })
        return cells_data
```

**Why validate the path in the constructor?** Because the alternative — picking the corpus by list index — is a bug waiting to happen:

```python
code_file_path = os.listdir()[1]   # ← silently selects .DS_Store, or the store, or this notebook
```

`os.listdir()` order is arbitrary. `NotebookLoader._validate` selects **by name** and rejects a missing file, a directory, a non-`.ipynb` file, and the degenerate case of a notebook ingesting itself.

---

### Step 2 — Clean, by Cell Type (`TextCleaner`)

This is the stage that decides whether your code chunks are usable. There are **two** cleaning paths, and mixing them up destroys the corpus.

```python
    @classmethod
    def clean(cls, text, cell_type="markdown") -> str:
        # ---------- CODE: conservative ----------
        if cell_type == "code":
            cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
            cleaned = "\n".join(line.rstrip() for line in cleaned.split("\n"))
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
            return cleaned.strip()          # trailing whitespace only — indentation stays

        # ---------- MARKDOWN: aggressive, but fences are protected ----------
        fenced_blocks = []
        def _stash_fence(match):
            fenced_blocks.append(match.group(0))
            return f"<FENCEDBLOCK{len(fenced_blocks) - 1}>"

        cleaned = re.sub(r"```.*?```", _stash_fence, cleaned, flags=re.DOTALL)
        ...                                  # 10 prose normalisation rules
        for i, block in enumerate(fenced_blocks):
            cleaned = cleaned.replace(f"<FENCEDBLOCK{i}>", block)
```

**What the prose path removes:** tokenizer artefacts (`[CLS]`, `[SEP]`, `<pad>`), invisible Unicode (zero-width spaces, BOM, non-breaking spaces), words broken across lines (`trans-\nformer` → `transformer`), repeated whitespace, spaces before punctuation, padding inside brackets.

**What it must never touch:** Python indentation. Rule 7 alone — `re.sub(r"[ \t]+", " ", cleaned)` — flattens every code block it reaches. Hence the fence stashing: fenced blocks leave before the prose rules run and come back untouched afterwards.

**Measured impact on this corpus:** cleaning removed **67 characters (0.04%)** and changed **15 of 223 cells**. That number is deliberately unimpressive. Cleaning is not there to shrink the corpus — it is there to make sure that the 15 cells with artefacts don't embed garbage, without damaging the other 208.

---

### Step 3 — Split into Units (`split_into_sentences`)

```python
        if cell_type == "code":
            return [text.strip()]            # a code cell is ONE unit
```

**Why code is never split on punctuation:** `model.fit(X, y)`, `0.90`, `df.shape` — a period inside code is not a sentence boundary. Splitting there produces fragments that are neither valid Python nor meaningful English.

For prose, abbreviations are protected before the split so `e.g.` and `et al.` don't end a sentence:

```python
        sentence_candidates = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\(\[])', protected_text)
```

The lookahead matters: a sentence ends when punctuation is followed by whitespace *and* something that looks like a new sentence start.

---

### Step 4 — Chunk, Cell by Cell (`NotebookChunker`)

Two invariants:

**1. A chunk never spans two cells.** The cell boundary is a real semantic boundary in a teaching notebook, and keeping it means every chunk can cite exactly one location.

**2. Consecutive chunks of the same cell overlap by `sentence_overlap` units.** A definition split across a boundary survives whole in at least one chunk.

```python
                if current_units and current_word_count + unit_word_count > self.max_words:
                    chunks_data.append(self._make_chunk(global_chunk_id, cell, current_units))
                    global_chunk_id += 1

                    # carry the tail of the closed chunk into the next one
                    current_units = current_units[-self.sentence_overlap:].copy()
                    current_word_count = sum(len(u.split()) for u in current_units)

                    # if the overlap itself leaves no room, drop the oldest units
                    while current_units and current_word_count + unit_word_count > self.max_words:
                        current_word_count -= len(current_units.pop(0).split())
```

Over-long units are handled by type:

```python
    def _split_oversized_code(self, unit: str) -> List[str]:
        """Split on LINE boundaries so every stored chunk is still readable Python."""
```

**Result on this corpus:** 223 cells → **230 chunks** (120 markdown, 110 code), mean 73.64 words, median 65.5, max exactly 250.

```bash
python train.py --dry-run
# Total cells found: 223
# Total chunks created: 230
# Words   mean/median: 73.64 / 65.5
```

---

### Step 5 — Choose the Chunking Hyperparameters

Do not copy `chunk_size=500` from a tutorial. Sweep it (`01_eda.ipynb`):

| `max_words` | Chunks created | Mean words/chunk | Largest chunk |
|---|---|---|---|
| 50 | 488 | 37.35 | 50 |
| 100 | 287 | 61.17 | 100 |
| 150 | 241 | 70.20 | 150 |
| 200 | 233 | 72.14 | 200 |
| **250** | **230** | **73.64** | **250** |
| 300 | 226 | 74.46 | 300 |
| 400 | 226 | 74.46 | 400 |
| 600 | 222 | 75.43 | 536 |

The curve is flat above 200 because **only 6 of 223 cells exceed the budget at all** — the corpus is written in short cells. Below 150 the index inflates fast (488 chunks at 50 words) and long code cells get shredded into fragments that don't stand alone.

| `sentence_overlap` | Chunks | Total stored words | Duplicated text |
|---|---|---|---|
| 0 | 229 | 16,745 | 0.00% |
| 1 | 229 | 16,761 | 0.10% |
| **2** | **230** | **16,938** | **1.15%** |
| 3 | 231 | 17,240 | 2.96% |
| 5 | 231 | 17,256 | 3.05% |

**The decision: `max_words=250`, `sentence_overlap=2`.** Both sit on the flat part of their curve — a larger budget buys nothing, a smaller one doubles the index, and 1.15% duplicated text is a cheap insurance premium against boundary loss.

> ⚠️ **These numbers are corpus-specific.** A corpus of long legal PDFs would put the elbow somewhere else entirely. The transferable part is the method, not the 250.

---

### Step 6 — Embed the Chunks (`ChunkEmbedder`)

```python
        chunk_embeddings = self.model.encode(
            chunk_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,      # ‖v‖ = 1 → cosine becomes a dot product
            show_progress_bar=verbose,
        ).astype("float32")                 # half the memory of float64, no quality loss
```

**Measured:** 230 chunks → `(230, 1024)` float32 in **50.17 seconds on CPU**. Verify normalisation immediately:

```python
norms = np.linalg.norm(chunk_embeddings, axis=1)
# Minimum norm: 1.0   Maximum norm: 1.0   Average norm: 1.0
```

If those norms aren't 1.0, cosine scores across your index are silently wrong.

**Why the embedding index is stored in metadata, not the vector:**

```python
for idx, chunk in enumerate(chunks_data):
    chunk["embedding_index"] = idx
```

Keeping 1024 floats per chunk inside the metadata list would multiply its size for no benefit — the vectors already live in Qdrant.

**Why an empty chunk gets a placeholder:** an empty string embeds to a meaningless vector that can still rank. Substituting `"empty chunk"` keeps row indices aligned with `chunks_data` — that alignment is what makes `PointStruct(id=idx, ...)` correct.

---

### Step 7 — Index in Qdrant (`QdrantIndex`)

```python
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)   # ← makes re-runs work

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )
```

The payload carries the **full** chunk metadata, so a search result is self-describing — `source`, `cell_number`, `cell_type`, `chunk_text`, `preview`, and the chunking parameters that produced it. No second lookup into an in-memory list is needed to cite a result, which matters the moment the querying process is not the process that built the index.

**Search refuses to run against an empty collection:**

```python
        points_count = self.count()
        if points_count == 0:
            raise RuntimeError(
                f"Qdrant collection '{self.collection_name}' contains 0 vectors. "
                "Upload the chunk embeddings before retrieving."
            )
        top_k = min(top_k, points_count)
```

Capping `top_k` against `len(chunks_data)` instead looks equivalent and is not: the in-memory list says nothing about what was actually uploaded. See Failure 4.

---

### Step 8 — Rerank (`ChunkReranker`)

```python
        pairs = [[cleaned_question, chunk.get("chunk_text", "")] for chunk in retrieved_chunks]
        reranker_scores = self.model.predict(pairs, show_progress_bar=verbose)

        scored_chunks.sort(key=lambda c: c["reranker_score"], reverse=True)
```

**Measured, on the question *"What is the role of embeddings in the retrieval process?"*:**

| Vector rank | Vector score | Reranker score | Final rank | Chunk |
|---|---|---|---|---|
| 2 | 0.6449 | **0.9263** | **1** | "Why Embeddings Help Retrieval" (cell 79) |
| 5 | 0.6128 | 0.5307 | 2 | Instructor answers (cell 122) |
| 1 | **0.6471** | 0.4996 | 3 | "Bridge to Next Section" (cell 146) |
| 4 | 0.6149 | 0.3190 | 4 | Instructor answers (cell 196) |
| 6 | 0.6106 | 0.2801 | 5 | "Bridge to Next Section" (cell 100) |

Read the first and third rows together. Vector search put a generic "Bridge to Next Section" cell **first** (0.6471) and the chunk actually titled *"Why Embeddings Help Retrieval"* **second** (0.6449) — a gap of 0.0022, which is noise. The cross-encoder, reading each pair properly, separates them decisively: 0.9263 vs 0.4996.

**That is the entire argument for reranking.** Bi-encoder scores in the top-10 are packed into a narrow band where ordering is close to arbitrary. The reranker spreads them out.

---

### Step 9 — Ground the Generation

Two functions, deliberately separate from the model call:

```python
    @staticmethod
    def build_evidence_context(chunks):
        # every block is labelled, so the answer can be traced back
        f"[Source: {chunk['source']}]\n[Cell: {chunk['cell_number']}]\n..."

    @staticmethod
    def build_prompt(question, evidence_context):
        """8 rules — the ones that matter:
        1. The study material is the ONLY source of truth.
        2. Do NOT use your general knowledge.
        3. Do NOT reinterpret the question using another field or domain.
        7. If the evidence is insufficient, say so explicitly.
        """
```

**Rule 3 is not paranoia.** Asked *"What is the difference between retrieval failure and generation failure?"* with **no** evidence, Qwen3-8B produced a fluent, well-structured, confident answer about **memory psychology** — encoding specificity, tip-of-the-tongue states, Broca's and Wernicke's aphasia. Every word plausible. Entirely the wrong domain. The corpus has a section on exactly this question in a RAG sense, and the ungrounded model never went near it.

**Then the whole pipeline in one call:**

```python
    def answer(self, question, retrieve_top_k=10, final_top_k=5, temperature=0.1):
        retrieved_chunks = self.retrieve(question, top_k=retrieve_top_k)
        reranked_chunks = self.rerank(question, retrieved_chunks, final_top_k=final_top_k)
        evidence_context = self.build_evidence_context(reranked_chunks)
        prompt = self.build_prompt(question, evidence_context)
        generation = self.generator.generate(prompt, temperature=temperature)
```

Every stage takes the **same `question` argument**. This is a correctness property, not a style choice — see Failure 5.

**Why `temperature=0.1`:** a grounded answer should reproduce its evidence, not improvise around it. Creativity is the enemy here.

**Measured end to end (CPU):** question embedding ~0.2s + rerank 5.11s + generation 8.17s ≈ **13.5 seconds per question**, on top of a one-time 50s index build.

---

### Step 10 — CLI (`train.py`)

```bash
python train.py --dry-run                                    # ingest + chunk only, no models
python train.py                                              # build the vector index
python train.py --question "What is a text embedding?"       # build, then answer
python train.py --no-build --question "Why is brute-force QnA not scalable?"
python train.py --max-words 150 --overlap 3                  # re-index with new chunking
python train.py --top-k 20 --final-top-k 8                   # widen the funnel
```

`--dry-run` is the one to reach for while iterating on chunking: it exercises loading, cleaning and chunking in about a second and loads no models at all.

---

### Step 11 — Unit Tests (`tests/test_solution.py`)

There are no accuracy metrics to assert here, so the tests protect the stages that *silently* corrupt a corpus:

```python
def test_code_cell_indentation_is_preserved():
    assert "\n    model.fit" in cleaned

def test_fenced_block_in_markdown_is_restored_verbatim():
    assert "\n    loss = train_one_epoch(   model )" in cleaned   # fence untouched
    assert "That is the whole idea." in cleaned                   # prose normalised

def test_consecutive_chunks_overlap():
    assert chunks[0]["chunk_text"].split("\n")[-2:] == chunks[1]["chunk_text"].split("\n")[:2]

def test_corpus_chunking_is_reproducible():
    assert len(cells) == 223
    assert len(chunks) == 230        # a change here changes every stored vector
```

```bash
pytest tests/ -v
# 11 passed in 1.07s ✅
```

None of them load a transformer or touch Qdrant, which is what keeps the suite honest and fast enough to run on every edit.

---

## 📊 Visualization Deep Dive

### `cell_type_distribution.png`

118 markdown (52.9%) vs 105 code (47.1%). This single plot justifies the entire two-path cleaning design. If the corpus were 95% prose, one cleaning path would be defensible. At a near 50/50 split, treating code like prose would corrupt nearly half the index — and the corruption would be invisible until someone reads a retrieved chunk and finds a 500-word one-liner.

### `cell_word_counts.png`

Left: word-count histograms for both cell types with the 250-word budget marked. Right: cell length across the notebook, coloured by type.

Three things to read:
1. **The mass is far left of the budget.** Median 65 words, mean 75.1. Chunking is mostly *packing*, not *splitting*.
2. **Only 6 cells cross 250 words** — but the longest is 536 (code). Those 6 are the only cells where the splitting rule matters, and they're exactly where a naive splitter does its damage.
3. **22 cells are under 20 words.** Bridge sentences, one-line imports. They still get indexed: a short code cell is a legitimate answer to "how do I open a PDF with PyMuPDF?"

### `cleaning_impact.png`

Left: raw vs cleaned characters per cell, with the identity line. Right: the histogram of characters removed.

Almost every point sits **on** the identity line — 208 of 223 cells are unchanged, and total shrinkage is 0.04%. That is the correct outcome. A cleaning stage that removes 20% of your corpus is not cleaning, it is deleting. The value is concentrated in the 15 cells that *did* change, where invisible Unicode and tokenizer artefacts would otherwise have been embedded as content.

### `chunk_size_distribution.png`

Left: words per chunk by type, with the mean (73.64) and the 250-word cap. Right: how many chunks each cell produced.

The right panel is the informative one: **the overwhelming majority of cells produce exactly one chunk.** 223 cells → 230 chunks means only a handful split. This is what a well-structured corpus looks like, and it is why the retrieval scores in this project are trustworthy — most chunks are whole, coherent thoughts written by a human, not arbitrary 500-character windows.

### `chunking_hyperparameter_sweep.png`

Left: `max_words` against index size (red) and mean chunk size (grey), with the chosen 250 marked. Right: duplicated text as a function of `sentence_overlap`.

The left curve has a sharp knee below 150 and is flat above 200 — the classic "choose the elbow" shape from Project 02's degree sweep, in a completely different domain. The right curve shows overlap costs rising roughly linearly to 3% at overlap 5; overlap 2 costs 1.15%.

> **The transferable lesson:** RAG hyperparameters are as sweepable as a polynomial degree. Most systems get them by imitation; measure them instead.

---

## 🔬 Why the Two-Stage Design Works

### Vector Search Compresses; the Cross-encoder Compares

A bi-encoder must squash a whole chunk into 1024 numbers **without knowing what will be asked**. That compression is lossy in a question-dependent way: the aspect of a chunk that answers *your* question may not survive it. The consequence shows up in the score distribution — the top-10 vector scores in the example above span 0.6106 to 0.6471, a range of 0.037. Ordering within that band is close to noise.

The cross-encoder has no compression bottleneck: it reads question and chunk in a single forward pass, with attention flowing between them. Its scores for the same 10 candidates spanned 0.2801 to 0.9263 — a range of 0.65, more than **17× wider**. That separation is what "more accurate" concretely means.

### Why the Funnel Ratio Matters

`top_k=10 → final_top_k=5` is a deliberate 2:1. The reranker can only reorder what it is given:

- `top_k = final_top_k` → the reranker changes nothing but the order of what you were already showing.
- `top_k` too large → cost grows linearly (each pair is a full model call: 10 pairs ≈ 5.11s on CPU) with diminishing returns, since chunks ranked 50–100 by the bi-encoder are rarely relevant.

### Retrieval Failure vs Generation Failure

The most important diagnostic distinction in applied RAG:

| | Retrieval failure | Generation failure |
|---|---|---|
| Symptom | Answer is about the wrong thing | Answer contradicts the evidence shown |
| Where to look | `result["reranked_chunks"]` | `result["prompt"]` |
| Diagnostic | Is the correct chunk in the evidence? | It's there — did the model use it? |
| Typical cause | Bad chunking, wrong `top_k`, vocabulary mismatch | Weak prompt, high temperature, evidence too long |
| Fix | Re-chunk, widen `top_k`, add reranking | Tighten the grounding rules, lower temperature |

This is why `answer()` returns `retrieved_chunks`, `reranked_chunks`, `evidence_context` **and** `prompt` alongside the answer. A RAG system that returns only a string cannot be debugged.

---

## 💥 Failure Analysis — What Broke and What We Learned

Every failure below happened while building this pipeline.

### Failure 1: `os.listdir()[1]` Selected the Wrong File

**What happened:**

```python
code_file_names = os.listdir()
code_file_path = code_file_names[1]      # ← "the second file"
```

The directory also contained `.DS_Store`, the `daedalus_qdrant/` store, and the notebook itself. Index 1 pointed at whichever the OS happened to list second.

**Why it happened:** `os.listdir()` returns entries in arbitrary order. The code encoded a fact about one machine at one moment.

**The fix:** select by name and validate — file exists, is a file, has `.ipynb`, is not this notebook.

**The lesson:** never address a file by position in a directory listing. Positional selection is fine for arrays and catastrophic for filesystems.

---

### Failure 2: Prose Cleaning Destroyed Code Indentation

**What happened:** the markdown cleaner ran `re.sub(r"[ \t]+", " ", cleaned)` over cells containing fenced ```` ```python ```` blocks. Every code block inside a markdown cell came out flattened to single spaces.

**Why it matters:** in a corpus that is 47% code, indentation *is* the information. A retrieved snippet with no indentation is not evidence — it's noise that a language model will confidently interpret anyway.

**The fix:** stash fenced blocks behind placeholders before any prose rule runs, restore them verbatim afterwards.

```python
cleaned = re.sub(r"```.*?```", _stash_fence, cleaned, flags=re.DOTALL)
...
for i, block in enumerate(fenced_blocks):
    cleaned = cleaned.replace(f"<FENCEDBLOCK{i}>", block)
```

**The lesson:** a text cleaner needs to know what *kind* of text it's cleaning. "Normalise whitespace" is correct for a paragraph and destructive for everything else. This is now covered by `test_fenced_block_in_markdown_is_restored_verbatim`.

---

### Failure 3: Long Code Cells Became One-Liners

**What happened:** the oversized-unit fallback split *everything* by words:

```python
piece = " ".join(words[start:start + max_words])   # ← for code too
```

**11 of 110 code chunks** came out as unreadable single lines — every newline gone.

**Why it happened:** `" ".join(text.split())` is the standard way to normalise prose. Applied to code it is a newline-deleting machine.

**The fix:** split long code on **line** boundaries so every stored piece is still valid, readable Python:

```python
    def _split_oversized_code(self, unit: str) -> List[str]:
        for line in unit.split("\n"):
            if piece_lines and piece_words + len(line.split()) > self.max_words:
                pieces.append("\n".join(piece_lines))
                ...
```

**The lesson:** 11 of 110 is 10% of your code corpus silently ruined — and nothing errors. Bugs in a data pipeline don't crash, they degrade. Covered by `test_long_code_cell_keeps_its_line_structure`.

---

### Failure 4: Retrieval Returned `[]` and Printed Nothing

**What happened:** the first retrieval call ran **before** the upload cell. It queried an empty collection, returned an empty list, printed nothing, and raised no error. Ten minutes went into debugging the embedding model.

**Why it happened:** `top_k` was capped against the in-memory list:

```python
top_k = min(top_k, len(chunks_data))    # ← says nothing about what's in Qdrant
```

`chunks_data` had 230 entries. The collection had 0. The two were never checked against each other.

**The fix:** cap against the collection and refuse to run against an empty index.

```python
points_count = self.count()
if points_count == 0:
    raise RuntimeError(f"Qdrant collection '{...}' contains 0 vectors. Upload first.")
top_k = min(top_k, points_count)
```

**The lesson:** an empty result is not an error condition your code gets to ignore. When "no results" and "nothing indexed" look identical from the outside, make the second one raise loudly.

---

### Failure 5: The Prompt Asked Question X with Evidence for Question Y

**What happened:** the question was set in one cell, and the evidence context was built from a `reranked_chunks` variable left over from an **earlier** question. The prompt asked about chunking and handed the model evidence about embeddings.

**Why it's the worst bug in this project:** it looks like it works. The model produces a fluent answer from whatever evidence it was handed. The retrieval stage was effectively bypassed, and nothing anywhere printed a warning.

**The fix:** one function that takes the question once and threads it through every stage:

```python
    def answer(self, question, ...):
        retrieved_chunks = self.retrieve(question, top_k=retrieve_top_k)
        reranked_chunks  = self.rerank(question, retrieved_chunks, final_top_k=final_top_k)
        evidence_context = self.build_evidence_context(reranked_chunks)
        prompt           = self.build_prompt(question, evidence_context)
```

**The lesson:** notebook state is a shared mutable global namespace, and RAG pipelines have many stages that each want to hold "the current question". If two pieces of state must agree, do not let two cells own them. Covered by `test_prompt_contains_question_and_cited_evidence`.

---

### Failure 6: `NameError` — a Cleaning Cell Called `clean_text()` Before It Existed

**What happened:** an early cell applied the cleaning pass; `clean_text` was defined thirty cells later. Running the notebook top to bottom died with `NameError: name 'clean_text' is not defined`.

**Why it happened:** the function was developed interactively, out of order. In a running kernel the name existed. In a fresh kernel it did not.

**The fix:** the early cell became an explicit no-op, and the cleaning pass now runs after the definition. In `solution.py` the problem cannot recur — module-level definitions are resolved at import.

**The lesson:** *"restart kernel and run all"* is the only real test of a notebook. A notebook that only works in the order you happened to click is not reproducible. This is a large part of why the logic now lives in `solution.py`, with the notebook importing it.

---

### Failure 7: Qdrant — "Collection already exists" and "Storage folder already accessed"

**What happened:** two failures on every re-run of a path-based (on-disk) Qdrant store.

```
ValueError: Collection daedalus_chunks already exists
RuntimeError: Storage folder ./daedalus_qdrant is already accessed by
              another instance of Qdrant client
```

**Why it happened:** local Qdrant persists to disk and holds an **exclusive lock**. Creating a collection twice is an error, and a second client cannot open a folder the first one still holds.

**The fix:** drop the stale collection before creating, and always close the client.

```python
if self.client.collection_exists(self.collection_name):
    self.client.delete_collection(self.collection_name)
...
finally:
    rag.close()          # train.py releases the lock even if generation raised
```

**The lesson:** anything that persists to disk needs an explicit lifecycle. "It worked the first time" is the standard failure signature of stateful infrastructure.

---

### Failure 8: The Ungrounded Model Answered in the Wrong Field Entirely

**What happened:** asked *"What is the difference between retrieval failure and generation failure?"* with no retrieved evidence, Qwen3-8B produced a confident, well-organised answer about **memory psychology** — encoding specificity, tip-of-the-tongue phenomena, Broca's and Wernicke's aphasia — complete with a comparison table.

**Why it happened:** with no context, the model resolves ambiguity using its pre-training prior. "Retrieval failure" is a far more common phrase in cognitive psychology than in RAG engineering.

**The fix:** grounding rules 2 and 3 in the prompt, and evidence retrieved for *this* question:

```
2. Do NOT use your general knowledge to answer the question.
3. Do NOT reinterpret the question using another field or domain.
```

With evidence, the same model answered a mechanism question about `create_overlapping_chunks_from_pages` precisely and correctly — *"increments `start` by `chunk_size - overlap` after each chunk"* — in 8.17 seconds.

**The lesson:** the failure mode of an ungrounded LLM is not "I don't know". It is a fluent, authoritative answer to a question you didn't ask. Groundedness is not a nice-to-have — it is the product.

---

## 🏭 Production Thinking

### Persisting and Versioning the Index

An index is defined by *four* things, and changing any of them invalidates it:

```python
index_metadata = {
    "corpus_file":        "Retrival_Ext_QnA_Complete.ipynb",
    "corpus_sha256":      "<hash of the corpus file>",
    "embedding_model":    "BAAI/bge-m3",
    "embedding_dim":      1024,
    "normalized":         True,
    "chunking_strategy":  "cell_aware_sentence_chunking",
    "max_words":          250,
    "sentence_overlap":   2,
    "total_chunks":       230,
    "built_at":           "2026-08-29",
}
```

> ⚠️ **Store this next to the collection and check it at query time.** If the corpus hash or the chunking parameters changed, the index is stale. If the *embedding model* changed, the index is not stale — it is **meaningless**, because the query vector now lives in a different coordinate system than the stored vectors. Scores will still come back. They will be nonsense.

### What Happens When Things Go Wrong in Production

#### Problem 1: The Corpus Changed, the Index Didn't

**Scenario:** three cells are edited and two are added. Nobody re-indexes.

**What happens:** the system answers from deleted content and cannot see new content. No error — the old vectors are still perfectly valid vectors.

**Defence:**

```python
def index_is_stale(corpus_path, index_metadata):
    current = hashlib.sha256(Path(corpus_path).read_bytes()).hexdigest()
    return current != index_metadata["corpus_sha256"]
```

Re-index on hash mismatch. For a large corpus, hash per *document* and re-embed only the documents that moved.

#### Problem 2: Query/Index Embedding Model Mismatch

**Scenario:** the embedding model is upgraded, or `local_files_only` picks up a different cached revision.

**What happens:** silent nonsense — plausible-looking cosine scores over incomparable spaces.

**Defence:** record the model name and dimension in the index metadata and refuse to query on mismatch:

```python
if embedder.model_name != index_metadata["embedding_model"]:
    raise RuntimeError("Embedding model does not match the index. Re-index before querying.")
```

#### Problem 3: The Question Has No Answer in the Corpus

**Scenario:** someone asks about a topic the corpus never covers. Vector search *always* returns `top_k` results — there is no "no match".

**What happens:** the reranker dutifully ranks five irrelevant chunks and the model is handed evidence that answers nothing.

**Defence:** a relevance floor on the reranker score, plus prompt rule 7:

```python
RELEVANCE_FLOOR = 0.1

if reranked_chunks[0]["reranker_score"] < RELEVANCE_FLOOR:
    return {"answer": "The study material does not cover this question.",
            "reranked_chunks": reranked_chunks}
```

Cross-encoder scores are comparable across queries in a way bi-encoder scores are not, which is what makes a fixed floor workable — calibrate it on real questions rather than guessing.

#### Problem 4: Latency

**Measured on CPU:** rerank 5.11s + generation 8.17s ≈ 13.5s per question. Acceptable for a study assistant, unacceptable for an interactive search box.

**Levers, cheapest first:** cache question embeddings for repeated queries · shrink `top_k` (rerank cost is linear in candidates) · move both models to GPU · use a smaller reranker · stream the generation so time-to-first-token drops even if total time doesn't.

### Production Checklist

- [ ] Index metadata stored: corpus hash, embedding model + dimension, chunking parameters, chunk count, build date
- [ ] Query-time guard: embedding model and dimension match the index
- [ ] Staleness check: corpus hash compared on startup
- [ ] Relevance floor defined and calibrated — the system can say "not in the corpus"
- [ ] Every answer returns its evidence with source and cell citations
- [ ] Retrieval and generation logged separately, so failures can be attributed
- [ ] Vector store lifecycle handled: client closed, collection drop/recreate is idempotent
- [ ] Re-index trigger defined: on corpus change, on chunking change, on model upgrade
- [ ] Temperature pinned and documented
- [ ] A regression set of questions with known-correct source cells, run after any chunking change

---

## 🚫 When NOT to Use RAG

| Scenario | Why RAG struggles | Use instead |
|----------|------------------|-------------|
| The answer needs the *whole* document (summarise this contract) | Retrieval returns fragments by design | Long-context prompting, map-reduce summarisation |
| Aggregation queries ("how many cells mention FAISS?") | Vector search is similarity, not counting | SQL / structured index / metadata filters |
| You need behaviour or format change, not knowledge | Retrieval adds facts, not style | Fine-tuning / LoRA (Project 21) |
| Multi-hop reasoning across scattered facts | One retrieval round can't chain inferences | Agentic RAG, query decomposition, iterative retrieval |
| Tiny corpus (a few thousand tokens) | Index infrastructure for something that fits in a prompt | Just put it in the prompt |
| Exact-match lookup (order #48213) | Embeddings blur exactly the distinctions IDs need | Keyword/BM25 or a database query |
| Freshness in seconds (live prices) | The index is only as fresh as the last build | Tool/API calls at query time |

### When RAG Is Right

- The corpus is large, changes independently of the model, and must be citable ✅
- Answers must be traceable to a source ✅
- Documents must be addable or removable without retraining ✅
- The knowledge is private and must stay local ✅
- You need to say "I don't know" when the corpus is silent ✅

---

## ⚠️ Common Mistakes & Gotchas

1. **Using a different embedding model for the query than for the index.** The single most destructive RAG bug: no error, plausible scores, meaningless ranking. Store the model name with the index and check it.

2. **Cleaning code with prose rules.** Collapsing whitespace destroys Python indentation. In this corpus that would corrupt 47% of the chunks — and nothing would crash.

3. **Chunking by character count with no structure awareness.** A 500-character window cuts mid-sentence and mid-function. Cell, section and paragraph boundaries are free structure — use them.

4. **Copying `chunk_size` from a tutorial.** Sweep it. On this corpus, anything from 200 to 400 is nearly identical, and 50 nearly doubles the index for no gain. On your corpus the answer will be different.

5. **Forgetting to normalise embeddings.** Cosine still works if the store computes it, but any manual dot-product similarity you write becomes wrong. Assert `‖v‖ = 1` right after encoding.

6. **`top_k == final_top_k`.** The reranker can only choose from what retrieval shortlisted. Retrieve 2–4× what you intend to keep.

7. **Trusting an empty result.** Vector search over an empty collection returns `[]`, not an error. Refuse to query an index with 0 points.

8. **Letting the question and the evidence live in separate variables.** In a notebook this produces an answer to the previous question with today's evidence — fluent and completely wrong. Thread one `question` argument through every stage.

9. **Reusing an in-memory chunk list as the source of truth for the index.** `len(chunks_data)` describes your Python process, not your vector store.

10. **Not distinguishing retrieval failure from generation failure.** "The answer is wrong" is not a diagnosis. Print the evidence first: if the right chunk isn't there, no prompt engineering will save you.

11. **High temperature in a grounded system.** Creativity is the failure mode. Pin it low (0.1) and document why.

12. **Committing the vector store.** `daedalus_qdrant/` is a build artifact — regenerate it, don't version it.

---

## 🎯 10 Interview Questions

<details>
<summary><strong>Q1: What is RAG and what problem does it solve that fine-tuning doesn't?</strong></summary>

**Answer:** RAG retrieves relevant passages from an external corpus and puts them in the prompt so the model answers from evidence rather than from its weights. It solves the *knowledge* problem; fine-tuning solves the *behaviour* problem. Facts fine-tuned into weights can't be updated without retraining, can't be cited, and can't be deleted when a document is retracted — and the model will still confidently answer about documents it never saw. With RAG, adding a document is an index update, every answer carries a source, and removing a document actually removes its influence. The rule of thumb: if you want the model to *know* something new, retrieve; if you want it to *behave* differently, fine-tune.

</details>

<details>
<summary><strong>Q2: Walk me through your chunking strategy and justify the parameters.</strong></summary>

**Answer:** Cell-aware chunking with sentence overlap. Markdown cells split into sentence units and pack up to 250 words with 2 units of overlap carried across boundaries; code cells stay whole and, when oversized, split on line boundaries so the stored chunk is still readable Python. A chunk never spans two cells, which keeps every citation pointing at exactly one location. The parameters came from a sweep, not a tutorial: `max_words` from 50 to 600 moved the index from 488 chunks to 222, but the curve is flat above 200 because only 6 of 223 cells exceed the budget — so 250 sits on the plateau. `sentence_overlap=2` costs 1.15% duplicated text; 5 costs 3.05% for no additional boundary safety. The transferable part is the method: sweep, look at the knee, choose deliberately.

</details>

<details>
<summary><strong>Q3: Why cosine similarity rather than Euclidean distance?</strong></summary>

**Answer:** Cosine measures the angle between vectors and ignores magnitude. In embedding space magnitude correlates with things like text length rather than meaning, so Euclidean distance would rank a 30-word chunk and a 200-word chunk about the same topic very differently. Cosine is bounded in [−1, 1], which makes scores interpretable and comparable across queries. There's also an efficiency argument: if you normalise every vector to unit length, the denominator becomes 1 and cosine similarity reduces to a plain dot product — so the whole index can be searched with one matrix multiply. That's why the pipeline sets `normalize_embeddings=True` and verifies the norms are exactly 1.0.

</details>

<details>
<summary><strong>Q4: What's the difference between a bi-encoder and a cross-encoder, and why use both?</strong></summary>

**Answer:** A bi-encoder embeds the question and each chunk independently, so chunk vectors are computed once at index time and a query costs one encode plus a vector search. A cross-encoder takes the question and chunk *together* in one forward pass, so attention flows between them — much more accurate, and impossible to precompute, because the chunk's representation depends on the question. Running a cross-encoder over an entire corpus is infeasible: it's one model call per chunk per query. So you use a funnel — the bi-encoder shortlists 10 candidates in milliseconds, the cross-encoder ranks those 10 in ~5 seconds and keeps 5. Measured on my pipeline, the bi-encoder's top-10 scores spanned 0.037 while the cross-encoder's spanned 0.65 on the same candidates, and it promoted the genuinely relevant chunk from rank 2 to rank 1 while pushing the bi-encoder's top hit down to rank 3.

</details>

<details>
<summary><strong>Q5: Your RAG system gives a wrong answer. How do you debug it?</strong></summary>

**Answer:** First separate retrieval failure from generation failure by printing the evidence that was actually used. If the correct chunk isn't in the evidence, it's a retrieval failure — check the chunking (was the fact split or corrupted?), widen `top_k`, check for vocabulary mismatch between question and corpus, verify the query used the same embedding model as the index. If the correct chunk *is* in the evidence and the answer still contradicts it, it's a generation failure — tighten the grounding rules, lower temperature, shorten the evidence block, or check that the prompt actually contains the evidence you think it does. This is why my `answer()` returns `retrieved_chunks`, `reranked_chunks`, `evidence_context` and `prompt` alongside the answer: a system that returns only a string cannot be debugged.

</details>

<details>
<summary><strong>Q6: How do you stop the model from hallucinating?</strong></summary>

**Answer:** You constrain the input and you constrain the instructions. The input constraint is retrieval: the model only sees passages actually retrieved for this question, and the question and its evidence are built in the same function call so they can't drift apart. The instruction constraint is a grounding prompt: the study material is the only source of truth, don't use general knowledge, don't reinterpret the question in another domain, and say explicitly when the evidence is insufficient. That third rule came from a real failure — asked about "retrieval failure vs generation failure" with no evidence, the model produced a fluent, confident answer about memory psychology. Then you add a relevance floor so the system can decline when the top reranked score is too low, and you keep temperature at 0.1. Finally you return the citations, so a human can check. None of this makes hallucination impossible — it makes it visible.

</details>

<details>
<summary><strong>Q7: Why do you store the full chunk metadata in the vector store payload?</strong></summary>

**Answer:** So that a search result is self-describing. The payload carries source file, cell number, cell type, the chunk text, a preview, and the chunking parameters that produced it. That means the querying process doesn't need the in-memory chunk list to cite a result — which matters as soon as indexing and querying are different processes, which they always are in production. It also makes results auditable after the fact: you can see which chunking configuration produced a given vector. The one thing deliberately *not* duplicated is the embedding itself — the metadata stores the row index, since keeping 1024 floats per chunk in a Python list buys nothing when the vectors already live in the store.

</details>

<details>
<summary><strong>Q8: What breaks if you change the embedding model but keep the index?</strong></summary>

**Answer:** Everything, silently. The stored vectors and the new query vectors live in different coordinate systems — the geometry that made "disadvantages" and "limitations" neighbours is specific to the model that produced it. Cosine similarity is still computable, so you get scores back, and they look like scores. The ranking is noise. Nothing raises. If the dimensions differ you get lucky and the store rejects the query; if the new model happens to share the dimension, you get a system that quietly returns irrelevant evidence forever. The defence is to record the model name and dimension in index metadata and refuse to query on mismatch. Changing the embedding model always means a full re-index.

</details>

<details>
<summary><strong>Q9: How would you evaluate this RAG system?</strong></summary>

**Answer:** Evaluate the two stages separately, because they fail for different reasons. For retrieval, build a small labelled set of questions with the source cell that should answer each, then measure recall@k (is the right chunk in the top-k?), MRR, and hit rate after reranking versus before — that last comparison is the only way to know whether reranking is earning its 5 seconds. For generation, measure faithfulness (is every claim in the answer supported by the evidence shown?), answer relevance, and the refusal rate on questions the corpus genuinely doesn't cover — a system that never says "I don't know" is not grounded, it's fluent. Frameworks like RAGAS automate some of this with an LLM judge, but on a corpus you know well, hand-labelling 30–50 questions gives a more trustworthy baseline. Then run that set as a regression suite after every chunking change.

</details>

<details>
<summary><strong>Q10: Your corpus grows from 230 chunks to 10 million. What changes?</strong></summary>

**Answer:** The architecture holds; the constants don't. Exact search becomes infeasible, so you move to an approximate index (HNSW, which Qdrant uses server-side) and accept a small recall loss for sub-linear search. Indexing becomes a batch job with incremental updates — you hash per document and re-embed only what changed, rather than rebuilding 10 million vectors because one file moved. Metadata filtering becomes essential: filter by source, date or type *before* the vector search so you're searching a relevant subset. Chunking parameters need re-tuning, because a heterogeneous corpus won't share one elbow. And retrieval quality degrades in a specific way — with 10 million chunks, many are near-duplicates, so you add deduplication and possibly hybrid search (BM25 + vectors) to recover the exact-match cases embeddings blur. The reranker becomes *more* valuable, not less: the bigger the candidate pool, the noisier the bi-encoder's ordering.

</details>

---

## 🏋️ Exercises & Challenges

**🟢 Beginner**
- [ ] Run `python train.py --dry-run --max-words 100 --overlap 0`. How many chunks now? Explain the change using the sweep table.
- [ ] Ask the same question with `--final-top-k 1` and `--final-top-k 5`. Does the answer change? Which chunks did the extra evidence add?
- [ ] Find a question the corpus genuinely doesn't cover (e.g. "How do I deploy this to Kubernetes?"). What does the system do? What *should* it do?
- [ ] Print `result["prompt"]` for one question. Count the tokens. How much of the context window is evidence vs instructions?

**🟡 Intermediate**
- [ ] Add a `--no-rerank` flag and compare the final evidence with and without reranking on five questions. Was the reranker's 5 seconds worth it every time?
- [ ] Implement the relevance floor from Production Thinking. Calibrate it: collect reranker scores for 10 answerable and 10 unanswerable questions, then pick the threshold.
- [ ] Add a `sha256` corpus hash and index metadata JSON. Make `train.py --no-build` refuse to query a stale index.
- [ ] Build a 20-question evaluation set with the correct source cell for each. Measure recall@5 before and after reranking.

**🔴 Advanced**
- [ ] Implement hybrid retrieval: BM25 keyword scores fused with vector scores using Reciprocal Rank Fusion. Measure whether it helps on exact-identifier questions (`create_overlapping_chunks_from_pages`).
- [ ] Add query decomposition: split a multi-part question into sub-questions, retrieve for each, then merge the evidence. Test on "How do embeddings and FAISS work together?"
- [ ] Swap Qdrant for FAISS behind the same `QdrantIndex` interface, without touching `DaedalusRAG`. What does that tell you about the abstraction?
- [ ] Implement incremental indexing: hash each cell, and on re-index only re-embed cells whose hash changed. Measure the time saved on a one-cell edit.
- [ ] Add answer-level citation: make the model cite `[Cell N]` inline, then verify programmatically that every cited cell was actually in the evidence.

---

## 🔗 What's Next

- **[20 — Transformer Attention →](../20-transformer-attention/)** — Open the box you've been using. Self-attention is what makes an embedding contextual and what makes a cross-encoder more accurate than a bi-encoder.
- **[21 — Fine-tune a Small LLM →](../21-finetune-llm/)** — The other half of the "make the model useful" answer. Build both and you can explain exactly when to reach for which.

---

## 📚 Further Reading

1. [Lewis et al. (2020) — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — the original RAG paper; the retriever/generator split starts here
2. [Karpukhin et al. (2020) — Dense Passage Retrieval](https://arxiv.org/abs/2004.04906) — why dense bi-encoders beat BM25, and how they're trained
3. [Chen et al. (2024) — BGE-M3](https://arxiv.org/abs/2402.03216) — the embedding model used here: multilingual, multi-granularity, multi-functional
4. [Nogueira & Cho (2019) — Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085) — the cross-encoder reranking result this pipeline's stage 2 is built on
5. [Qdrant documentation — collections, payloads and filtering](https://qdrant.tech/documentation/) — production vector store patterns beyond the local on-disk mode
6. [RAGAS — evaluation framework for RAG pipelines](https://docs.ragas.io/) — faithfulness, answer relevance and context precision, automated
7. [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — prepending chunk-specific context before embedding, a direct upgrade to the chunker here

---

<p align="center">
  <strong>ai-from-scratch</strong> · Project 22 · RAG Pipeline from Scratch<br>
  Built with curiosity · Shared with the community
</p>
