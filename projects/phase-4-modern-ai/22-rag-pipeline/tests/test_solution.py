"""
Unit tests for the Daedalus RAG pipeline.

These tests deliberately cover the stages that decide RETRIEVAL QUALITY and
that no model can rescue: cleaning, sentence splitting, chunking and prompt
construction. If a code chunk loses its indentation here, the best embedding
model in the world will index garbage.

None of these tests load a transformer or touch Qdrant, so the suite runs in
under a second on any machine.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from solution import (
    DaedalusRAG,
    NotebookChunker,
    NotebookLoader,
    TextCleaner,
)

CORPUS_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "corpus",
    "Retrival_Ext_QnA_Complete.ipynb",
)


# ==================================================================
# 1. Cleaning preserves Python indentation in code cells
# ==================================================================

def test_code_cell_indentation_is_preserved():
    """Prose whitespace rules must never touch a code cell."""
    code = (
        "def train_model(model, X_train, y_train):\n"
        "    model.fit(X_train, y_train)\n"
        "\n"
        "    if model.score(X_train, y_train) > 0.90:\n"
        "        print( 'Excellent model' )\n"
        "\n"
        "    return model"
    )

    cleaned = TextCleaner.clean(code, "code")

    assert "\n    model.fit" in cleaned
    assert "\n        print( 'Excellent model' )" in cleaned  # spacing untouched


# ==================================================================
# 2. Fenced code blocks inside Markdown survive prose cleaning
# ==================================================================

def test_fenced_block_in_markdown_is_restored_verbatim():
    """A ```python block inside a Markdown cell keeps its indentation."""
    markdown = (
        "Here is the training loop:\n"
        "\n"
        "```python\n"
        "for epoch in range(10):\n"
        "    loss = train_one_epoch(   model )\n"
        "```\n"
        "\n"
        "That is    the whole idea."
    )

    cleaned = TextCleaner.clean(markdown, "markdown")

    assert "\n    loss = train_one_epoch(   model )" in cleaned  # fence untouched
    assert "That is the whole idea." in cleaned                  # prose normalised


# ==================================================================
# 3. Tokenizer artefacts and invisible characters are removed
# ==================================================================

def test_special_tokens_and_invisible_chars_are_removed():
    """Leftover [CLS]/[SEP] tokens and zero-width chars pollute embeddings."""
    noisy = "[CLS] Transformers use​ self-attention [SEP] and\xa0scale well."

    cleaned = TextCleaner.clean(noisy, "markdown")

    for token in ["[CLS]", "[SEP]"]:
        assert token not in cleaned

    for char in ["​", "\xa0"]:
        assert char not in cleaned

    assert "self-attention" in cleaned


# ==================================================================
# 4. Code is one structural unit — never split on punctuation
# ==================================================================

def test_code_is_not_split_on_sentence_punctuation():
    """A period in `model.fit(...)` or `0.90` is not a sentence boundary."""
    code = "model.fit(X, y)\nscore = model.score(X, y)  # target is 0.90"

    units = TextCleaner.split_into_sentences(code, "code")

    assert len(units) == 1

    prose = "Embeddings encode meaning. Cosine similarity compares them."
    assert len(TextCleaner.split_into_sentences(prose, "markdown")) == 2


# ==================================================================
# 5. Chunks respect max_words
# ==================================================================

def test_chunks_respect_max_words():
    """No prose chunk may exceed the configured budget."""
    sentence = "Retrieval finds the relevant chunk before the reader runs. "
    cell = {
        "source": "unit_test.ipynb",
        "source_type": "ipynb",
        "cell_number": 1,
        "cell_type": "markdown",
        "cleaned_text": sentence * 40,
    }

    chunks = NotebookChunker(max_words=50, sentence_overlap=1).chunk([cell])

    assert len(chunks) > 1
    assert all(chunk["word_count"] <= 50 for chunk in chunks)


# ==================================================================
# 6. Consecutive chunks of a cell share overlapping sentences
# ==================================================================

def test_consecutive_chunks_overlap():
    """Overlap is what stops a fact dying on a chunk boundary."""
    sentences = [f"Sentence number {i} explains one idea clearly." for i in range(40)]
    cell = {
        "source": "unit_test.ipynb",
        "source_type": "ipynb",
        "cell_number": 1,
        "cell_type": "markdown",
        "cleaned_text": " ".join(sentences),
    }

    overlap = 2
    chunks = NotebookChunker(max_words=40, sentence_overlap=overlap).chunk([cell])

    assert len(chunks) >= 2

    tail = chunks[0]["chunk_text"].split("\n")[-overlap:]
    head = chunks[1]["chunk_text"].split("\n")[:overlap]

    assert tail == head


# ==================================================================
# 7. A chunk never spans two cells, and ids stay sequential
# ==================================================================

def test_chunks_never_span_two_cells():
    """Cell boundaries are what make a citation ("Cell 54") meaningful."""
    cells = [
        {
            "source": "unit_test.ipynb",
            "source_type": "ipynb",
            "cell_number": n,
            "cell_type": "markdown",
            "cleaned_text": f"Cell {n} discusses one specific topic in depth.",
        }
        for n in range(1, 6)
    ]

    chunks = NotebookChunker(max_words=250, sentence_overlap=2).chunk(cells)

    assert len(chunks) == len(cells)
    assert [c["chunk_id"] for c in chunks] == list(range(len(cells)))

    for chunk, cell in zip(chunks, cells):
        assert chunk["cell_number"] == cell["cell_number"]
        assert chunk["chunk_text"] == cell["cleaned_text"]


# ==================================================================
# 8. Over-long code splits on line boundaries, not word boundaries
# ==================================================================

def test_long_code_cell_keeps_its_line_structure():
    """Word-splitting code collapses it into an unreadable one-liner."""
    code_lines = [
        f"    variable_{i} = compute_value({i}, scale=2, offset=1)"
        for i in range(60)
    ]
    cell = {
        "source": "unit_test.ipynb",
        "source_type": "ipynb",
        "cell_number": 1,
        "cell_type": "code",
        "cleaned_text": "def build():\n" + "\n".join(code_lines),
    }

    chunks = NotebookChunker(max_words=100, sentence_overlap=0).chunk([cell])

    assert len(chunks) > 1

    # Line structure survives the split: no chunk is a collapsed one-liner
    # holding several statements, and inner lines keep their indentation.
    total_lines = sum(len(chunk["chunk_text"].split("\n")) for chunk in chunks)
    assert total_lines == 61                              # 1 def + 60 statements
    assert "\n    variable_" in chunks[0]["chunk_text"]   # indentation survived


# ==================================================================
# 9. The prompt carries the question and its own evidence
# ==================================================================

def test_prompt_contains_question_and_cited_evidence():
    """Question and evidence are built together so they cannot drift apart."""
    chunks = [
        {
            "source": "corpus.ipynb",
            "cell_number": 54,
            "cell_type": "markdown",
            "chunk_text": "Brute-force QnA runs the reader on every chunk.",
        }
    ]

    question = "Why is brute-force QnA not scalable?"
    evidence = DaedalusRAG.build_evidence_context(chunks)
    prompt = DaedalusRAG.build_prompt(question, evidence)

    assert "[Source: corpus.ipynb]" in evidence
    assert "[Cell: 54]" in evidence
    assert question in prompt
    assert "Brute-force QnA runs the reader on every chunk." in prompt
    assert "ONLY the provided study material" in prompt


# ==================================================================
# 10. The corpus still produces the same index (regression test)
# ==================================================================

def test_corpus_chunking_is_reproducible():
    """223 cells -> 230 chunks. A change here changes every stored vector."""
    if not os.path.exists(CORPUS_PATH):
        pytest.skip("Corpus notebook not available.")

    cells = NotebookLoader(CORPUS_PATH).load(verbose=False)
    cleaned = TextCleaner.clean_cells(cells)
    chunks = NotebookChunker(max_words=250, sentence_overlap=2).chunk(cleaned)

    assert len(cells) == 223
    assert len(chunks) == 230

    stats = NotebookChunker.describe(chunks)
    assert stats["max_words"] <= 250
    assert stats["mean_words"] == pytest.approx(73.64, abs=0.01)

    # Every chunk must be traceable back to a cell of the corpus.
    assert all(chunk["source"].endswith(".ipynb") for chunk in chunks)
    assert all(1 <= chunk["cell_number"] <= 223 for chunk in chunks)


# ==================================================================
# 11. Guard rails
# ==================================================================

def test_invalid_inputs_raise():
    """Fail loudly at the boundary instead of silently indexing nothing."""
    with pytest.raises(FileNotFoundError):
        NotebookLoader("corpus/does_not_exist.ipynb")

    with pytest.raises(ValueError):
        NotebookLoader("solution.py")

    with pytest.raises(ValueError):
        NotebookChunker(max_words=0)

    with pytest.raises(ValueError):
        NotebookChunker(sentence_overlap=-1)
