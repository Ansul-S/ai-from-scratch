"""
Project 22 — RAG Pipeline from Scratch (Daedalus)

A retrieval-augmented generation pipeline over a Jupyter Notebook corpus.

Index time:
    .ipynb  ->  cells  ->  cleaned cells  ->  chunks  ->  embeddings  ->  Qdrant

Query time:
    question -> embedding -> vector search (top_k)
             -> cross-encoder rerank (final_top_k)
             -> evidence block -> grounded prompt -> LLM answer

Design notes
------------
1. Heavy dependencies (torch, sentence-transformers, qdrant-client, requests)
   are imported lazily inside the classes that need them. Importing this
   module therefore costs nothing, and the pure-text stages (cleaning,
   sentence splitting, chunking, prompt building) are unit-testable on a
   machine with no models installed.

2. Every stage is a separate class with a single responsibility, so a stage
   can be swapped (a different embedding model, a different vector store)
   without touching the rest of the pipeline.

3. Code cells and Markdown cells are cleaned and chunked by DIFFERENT rules.
   Prose normalisation destroys Python indentation, and Python indentation is
   exactly the signal a code-retrieval corpus needs.
"""

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# ==================================================================
# Device helper
# ==================================================================

def resolve_device(device: Optional[str] = None) -> str:
    """
    Pick the compute device for the embedding and reranking models.

    Parameters
    ----------
    device : str, optional
        Explicit device ("cuda", "mps", "cpu"). If None, CUDA is used when
        available and CPU otherwise.

    Returns
    -------
    str
        The device string handed to sentence-transformers.
    """
    if device is not None:
        return device

    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


# ==================================================================
# Stage 1 — Load a notebook corpus
# ==================================================================

class NotebookLoader:
    """
    Read a .ipynb corpus file cell by cell and attach source metadata.

    A notebook is not a flat text file: it is an ordered list of typed cells.
    Keeping the cell boundary (and the cell type) all the way through the
    pipeline is what later lets an answer cite "Cell 54, markdown".
    """

    def __init__(self, notebook_path: str, this_notebook_name: Optional[str] = None):
        """
        Parameters
        ----------
        notebook_path : str
            Path to the .ipynb file to ingest.
        this_notebook_name : str, optional
            Name of the notebook doing the ingesting. Used to refuse the
            degenerate case of a notebook indexing itself.
        """
        self.notebook_path = self._validate(notebook_path, this_notebook_name)

    @staticmethod
    def _validate(notebook_path: str, this_notebook_name: Optional[str]) -> str:
        """
        Resolve and validate the corpus file before anything is read.

        Selecting the corpus by list index (``os.listdir()[1]``) is unsafe:
        directory order is arbitrary and the folder also holds .DS_Store, the
        vector store and the notebook itself. Select by name and validate.
        """
        path = Path(notebook_path)

        if not path.exists():
            raise FileNotFoundError(f"Corpus file not found: {path}")

        if not path.is_file():
            raise ValueError(f"Expected a file, but this is a directory: {path}")

        if path.suffix != ".ipynb":
            raise ValueError(f"Expected a .ipynb file, got: {path.suffix!r}")

        if this_notebook_name is not None and path.name == this_notebook_name:
            raise ValueError("Refusing to ingest this notebook into itself.")

        return str(path)

    def load(self, verbose: bool = True) -> List[Dict[str, Any]]:
        """
        Extract text and metadata from every cell of the notebook.

        Returns
        -------
        cells_data : List[Dict[str, Any]]
            One dictionary per cell: source, source_type, cell_number,
            cell_type, text, word_count, character_count.
        """
        import nbformat

        with open(self.notebook_path, "r", encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=4)

        source_name = Path(self.notebook_path).name

        if verbose:
            print(f"Total cells found: {len(notebook.cells)}")

        cells_data = []

        for cell_index, cell in enumerate(notebook.cells):
            text = cell.source

            cells_data.append({
                "source": source_name,          # e.g. Retrival_Ext_QnA_Complete.ipynb
                "source_type": "ipynb",
                "cell_number": cell_index + 1,
                "cell_type": cell.cell_type,    # markdown / code / raw
                "text": text,
                "word_count": len(text.split()),
                "character_count": len(text),
            })

        return cells_data


# ==================================================================
# Stage 2 — Clean the extracted text
# ==================================================================

class TextCleaner:
    """
    Cell-type aware cleaning and sentence splitting.

    The single most important rule in this class: code is cleaned
    conservatively, prose is cleaned aggressively. Collapsing runs of
    whitespace is correct for a paragraph and catastrophic for Python.
    """

    SPECIAL_TOKENS = [
        "[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]",
        "<s>", "</s>", "<pad>", "</pad>", "<unk>", "<mask>",
        "<bos>", "</bos>", "<eos>", "</eos>",
    ]

    INVISIBLE_CHARS = [
        "\u200b",  # zero-width space
        "\u200c",  # zero-width non-joiner
        "\u200d",  # zero-width joiner
        "\ufeff",  # byte order mark
        "\xa0",    # non-breaking space
    ]

    ABBREVIATIONS = {
        "e.g.": "e<DOT>g<DOT>",
        "i.e.": "i<DOT>e<DOT>",
        "Fig.": "Fig<DOT>",
        "fig.": "fig<DOT>",
        "Eq.": "Eq<DOT>",
        "eq.": "eq<DOT>",
        "Dr.": "Dr<DOT>",
        "Mr.": "Mr<DOT>",
        "Ms.": "Ms<DOT>",
        "Prof.": "Prof<DOT>",
        "vs.": "vs<DOT>",
        "et al.": "et al<DOT>",
    }

    # --------------------------------------------------------------
    # Cleaning
    # --------------------------------------------------------------

    @classmethod
    def clean(cls, text: Optional[str], cell_type: str = "markdown") -> str:
        """
        Clean one notebook cell while preserving its structure.

        Parameters
        ----------
        text : str
            Raw text extracted from a notebook cell.
        cell_type : {"markdown", "code", "raw"}
            Type of the cell the text came from.

        Returns
        -------
        cleaned : str
            Cleaned text.
        """
        if text is None:
            return ""

        cleaned = str(text)

        # ==========================================================
        # CODE CELL — conservative
        # ==========================================================
        # Do not normalise spaces, punctuation, brackets or tokens.
        # All of them may be part of the actual Python source.
        if cell_type == "code":
            cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

            # Trailing whitespace only. Leading indentation is preserved.
            cleaned = "\n".join(line.rstrip() for line in cleaned.split("\n"))

            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

            return cleaned.strip()

        # ==========================================================
        # MARKDOWN / RAW CELL — aggressive, but fences are protected
        # ==========================================================

        # 0. Stash fenced code blocks before any prose rule runs, and
        #    restore them untouched at the end. Without this, the
        #    whitespace rules below flatten the indentation of every
        #    ```python block inside a Markdown cell.
        fenced_blocks: List[str] = []

        def _stash_fence(match: "re.Match") -> str:
            fenced_blocks.append(match.group(0))
            return f"<FENCEDBLOCK{len(fenced_blocks) - 1}>"

        cleaned = re.sub(r"```.*?```", _stash_fence, cleaned, flags=re.DOTALL)

        # 1. Remove tokenizer/model special tokens
        for token in cls.SPECIAL_TOKENS:
            cleaned = cleaned.replace(token, " ")

        # 2. Remove invisible Unicode characters
        for char in cls.INVISIBLE_CHARS:
            cleaned = cleaned.replace(char, " ")

        # 3. Normalise line endings
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

        # 4. Repair words broken across a line ("trans-\nformer" -> "transformer")
        cleaned = re.sub(r"(\w)-\n(\w)", r"\1\2", cleaned)

        # 5. Collapse runs of blank lines
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        # 6. Strip trailing whitespace per line
        cleaned = "\n".join(line.rstrip() for line in cleaned.split("\n"))

        # 7. Collapse repeated spaces and tabs
        cleaned = re.sub(r"[ \t]+", " ", cleaned)

        # 8. Remove space before punctuation
        cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)

        # 9. Remove padding inside brackets
        cleaned = re.sub(r"([\(\[\{])[ \t]+", r"\1", cleaned)
        cleaned = re.sub(r"[ \t]+([\)\]\}])", r"\1", cleaned)

        # 10. Restore fenced blocks exactly as they were
        for fence_index, block in enumerate(fenced_blocks):
            cleaned = cleaned.replace(f"<FENCEDBLOCK{fence_index}>", block)

        return cleaned.strip()

    @classmethod
    def clean_cells(cls, cells_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean every cell and keep the raw text alongside the cleaned text.

        Keeping both is what makes the cleaning stage auditable: any cell
        whose word count moves unexpectedly can be diffed character by
        character.
        """
        cleaned_cells_data = []

        for cell in cells_data:
            raw_text = cell["text"]
            cleaned_text = cls.clean(raw_text, cell["cell_type"])

            cleaned_cells_data.append({
                "source": cell["source"],
                "source_type": cell["source_type"],
                "cell_number": cell["cell_number"],
                "cell_type": cell["cell_type"],
                "raw_text": raw_text,
                "cleaned_text": cleaned_text,
                "raw_word_count": cell["word_count"],
                "cleaned_word_count": len(cleaned_text.split()),
                "raw_character_count": cell["character_count"],
                "cleaned_character_count": len(cleaned_text),
            })

        return cleaned_cells_data

    # --------------------------------------------------------------
    # Sentence splitting
    # --------------------------------------------------------------

    @classmethod
    def split_into_sentences(
        cls,
        text: Optional[str],
        cell_type: str = "markdown",
    ) -> List[str]:
        """
        Split cleaned cell text into sentence-like units.

        A code cell is returned as ONE unit: a period inside a string, a
        float literal or an attribute access is not a sentence boundary, and
        splitting there would produce syntactically meaningless fragments.
        """
        if text is None:
            return []

        text = cls.clean(text, cell_type)

        if len(text.strip()) == 0:
            return []

        # Code is a single structural unit.
        if cell_type == "code":
            return [text.strip()]

        # Protect abbreviations so "e.g." does not end a sentence.
        protected_text = text
        for original, protected in cls.ABBREVIATIONS.items():
            protected_text = protected_text.replace(original, protected)

        # Split on terminal punctuation followed by a likely sentence start.
        sentence_candidates = re.split(
            r'(?<=[.!?])\s+(?=[A-Z0-9"\(\[])',
            protected_text,
        )

        sentences = []

        for sentence in sentence_candidates:
            sentence = sentence.replace("<DOT>", ".")
            sentence = cls.clean(sentence, cell_type)

            if len(sentence.strip()) > 0:
                sentences.append(sentence)

        return sentences


# ==================================================================
# Stage 3 — Chunk the cleaned cells
# ==================================================================

class NotebookChunker:
    """
    Structure-aware chunking with sentence overlap.

    Markdown and raw cells are split into sentence units and packed up to
    ``max_words``, carrying ``sentence_overlap`` units across the boundary so
    a fact split across two chunks survives in at least one of them.

    Code cells stay whole; when a code cell is longer than ``max_words`` it is
    split on LINE boundaries, never on word boundaries, so every stored chunk
    is still readable Python.

    Chunks never span two cells. Cell boundary is the strongest structural
    signal a notebook gives you, and it is what makes citations meaningful.
    """

    STRATEGY = "cell_aware_sentence_chunking"

    def __init__(self, max_words: int = 250, sentence_overlap: int = 2):
        if max_words <= 0:
            raise ValueError("max_words must be greater than 0.")

        if sentence_overlap < 0:
            raise ValueError("sentence_overlap cannot be negative.")

        self.max_words = max_words
        self.sentence_overlap = sentence_overlap

    # --------------------------------------------------------------

    def _make_chunk(
        self,
        chunk_id: int,
        cell: Dict[str, Any],
        units: List[str],
    ) -> Dict[str, Any]:
        """Assemble one chunk dictionary from a list of units."""
        cell_type = cell["cell_type"]

        chunk_text = TextCleaner.clean("\n".join(units), cell_type)

        return {
            "chunk_id": chunk_id,
            "source": cell["source"],
            "source_type": cell["source_type"],
            "cell_number": cell["cell_number"],
            "cell_type": cell_type,
            "chunk_text": chunk_text,
            "word_count": len(chunk_text.split()),
            "character_count": len(chunk_text),
            "preview": chunk_text[:300],
            "num_units": len(units),
            "chunking_strategy": self.STRATEGY,
            "max_words": self.max_words,
            "sentence_overlap": self.sentence_overlap,
        }

    def _split_oversized_code(self, unit: str) -> List[str]:
        """
        Split an over-long code unit on line boundaries.

        Splitting code by words (``" ".join(words[i:i + max_words])``)
        collapses the whole cell onto one line and destroys indentation.
        """
        pieces = []
        piece_lines: List[str] = []
        piece_words = 0

        for line in unit.split("\n"):
            line_words = len(line.split())

            if piece_lines and piece_words + line_words > self.max_words:
                piece = "\n".join(piece_lines)
                if len(piece.strip()) > 0:
                    pieces.append(piece)
                piece_lines = []
                piece_words = 0

            piece_lines.append(line)
            piece_words += line_words

        piece = "\n".join(piece_lines)
        if len(piece.strip()) > 0:
            pieces.append(piece)

        return pieces

    def _split_oversized_prose(self, unit: str, cell_type: str) -> List[str]:
        """Split an over-long prose unit on word boundaries."""
        words = unit.split()
        pieces = []

        for start in range(0, len(words), self.max_words):
            piece = TextCleaner.clean(
                " ".join(words[start:start + self.max_words]),
                cell_type,
            )
            if len(piece.strip()) > 0:
                pieces.append(piece)

        return pieces

    # --------------------------------------------------------------

    def chunk(self, cleaned_cells_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create chunks from cleaned cells.

        Parameters
        ----------
        cleaned_cells_data : List[Dict[str, Any]]
            Output of ``TextCleaner.clean_cells``.

        Returns
        -------
        chunks_data : List[Dict[str, Any]]
            Chunks with full source, cell and chunking metadata.
        """
        chunks_data: List[Dict[str, Any]] = []
        global_chunk_id = 0

        for cell in cleaned_cells_data:
            cell_type = cell["cell_type"]
            cell_text = cell["cleaned_text"]

            if len(cell_text.strip()) == 0:
                continue

            units = TextCleaner.split_into_sentences(cell_text, cell_type)

            if len(units) == 0:
                continue

            # ------ normalise unit sizes ------
            processed_units: List[str] = []

            for unit in units:
                if len(unit.split()) <= self.max_words:
                    processed_units.append(unit)
                elif cell_type == "code":
                    processed_units.extend(self._split_oversized_code(unit))
                else:
                    processed_units.extend(self._split_oversized_prose(unit, cell_type))

            # ------ pack units into chunks ------
            current_units: List[str] = []
            current_word_count = 0

            for unit in processed_units:
                unit = TextCleaner.clean(unit, cell_type)
                unit_word_count = len(unit.split())

                if unit_word_count == 0:
                    continue

                # Adding this unit would overflow: close the current chunk.
                if current_units and current_word_count + unit_word_count > self.max_words:
                    chunks_data.append(
                        self._make_chunk(global_chunk_id, cell, current_units)
                    )
                    global_chunk_id += 1

                    # Carry the tail of the closed chunk into the next one.
                    if self.sentence_overlap > 0:
                        current_units = current_units[-self.sentence_overlap:].copy()
                    else:
                        current_units = []

                    current_word_count = sum(len(u.split()) for u in current_units)

                    # If the overlap itself leaves no room, drop the oldest
                    # overlap units until the incoming unit fits.
                    while current_units and current_word_count + unit_word_count > self.max_words:
                        removed_unit = current_units.pop(0)
                        current_word_count -= len(removed_unit.split())

                current_units.append(unit)
                current_word_count += unit_word_count

            # ------ final chunk of the cell ------
            if current_units:
                chunks_data.append(
                    self._make_chunk(global_chunk_id, cell, current_units)
                )
                global_chunk_id += 1

        return chunks_data

    @staticmethod
    def describe(chunks_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summary statistics for a chunk set — used by the EDA notebook."""
        if len(chunks_data) == 0:
            raise ValueError("chunks_data is empty.")

        word_counts = [c["word_count"] for c in chunks_data]
        char_counts = [c["character_count"] for c in chunks_data]
        unit_counts = [c["num_units"] for c in chunks_data]

        return {
            "total_chunks": len(chunks_data),
            "min_words": int(np.min(word_counts)),
            "max_words": int(np.max(word_counts)),
            "mean_words": round(float(np.mean(word_counts)), 2),
            "median_words": round(float(np.median(word_counts)), 2),
            "min_units": int(np.min(unit_counts)),
            "max_units": int(np.max(unit_counts)),
            "mean_units": round(float(np.mean(unit_counts)), 2),
            "min_characters": int(np.min(char_counts)),
            "max_characters": int(np.max(char_counts)),
            "mean_characters": round(float(np.mean(char_counts)), 2),
        }


# ==================================================================
# Stage 4 — Embed the chunks
# ==================================================================

class ChunkEmbedder:
    """
    Wrap a sentence-transformers bi-encoder (default: BAAI/bge-m3).

    A bi-encoder embeds the question and every chunk INDEPENDENTLY, which is
    what makes retrieval cheap: chunk vectors are computed once at index time
    and only the question is encoded per query.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        local_files_only: bool = False,
        verbose: bool = True,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = resolve_device(device)

        start_time = time.time()
        self.model = SentenceTransformer(
            model_name,
            device=self.device,
            local_files_only=local_files_only,
        )
        load_time = time.time() - start_time

        if verbose:
            print("Embedding model loaded.")
            print("Model name:", model_name)
            print("Device:", self.device)
            print("Embedding dimension:", self.dimension)
            print("Loading time:", round(load_time, 2), "seconds")

    @property
    def dimension(self) -> int:
        """Embedding dimensionality (1024 for BGE-M3)."""
        return self.model.get_sentence_embedding_dimension()

    def encode_chunks(
        self,
        chunks_data: List[Dict[str, Any]],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Embed every chunk.

        Returns
        -------
        chunk_embeddings : np.ndarray
            float32 matrix of shape (n_chunks, embedding_dim).
        """
        if len(chunks_data) == 0:
            raise ValueError(
                "chunks_data is empty. Create chunks before generating embeddings."
            )

        chunk_texts = []

        for chunk in chunks_data:
            text = TextCleaner.clean(
                chunk.get("chunk_text", ""),
                chunk.get("cell_type", "markdown"),
            )
            # An empty string embeds to a meaningless vector; give it a
            # harmless placeholder instead so indices stay aligned.
            chunk_texts.append(text if len(text.strip()) > 0 else "empty chunk")

        start_time = time.time()

        chunk_embeddings = self.model.encode(
            chunk_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=verbose,
        ).astype("float32")

        embed_time = time.time() - start_time

        # Chunk metadata stores the row index, not the vector: keeping 1024
        # floats per chunk in the metadata list would bloat it for nothing.
        for idx, chunk in enumerate(chunks_data):
            chunk["embedding_index"] = idx

        if verbose:
            print("Chunk embeddings created.")
            print("Number of chunks:", len(chunks_data))
            print("Embedding matrix shape:", chunk_embeddings.shape)
            print("Embedding dtype:", chunk_embeddings.dtype)
            print("Normalized:", normalize_embeddings)
            print("Embedding time:", round(embed_time, 2), "seconds")

        return chunk_embeddings

    def encode_query(self, question: str) -> np.ndarray:
        """Embed a single question with the SAME model used for the chunks."""
        cleaned_question = TextCleaner.clean(question, "markdown")

        return self.model.encode(
            [cleaned_question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")[0]


# ==================================================================
# Stage 5 — Store and search the vectors
# ==================================================================

class QdrantIndex:
    """
    Local, on-disk Qdrant collection holding chunk vectors and payloads.

    The payload carries the full chunk metadata, so a search result is
    self-describing: no second lookup into an in-memory list is needed to
    know which cell an answer came from.
    """

    def __init__(
        self,
        path: str = "./daedalus_qdrant",
        collection_name: str = "daedalus_chunks",
        verbose: bool = True,
    ):
        from qdrant_client import QdrantClient

        self.path = path
        self.collection_name = collection_name
        self.client = QdrantClient(path=path)

        if verbose:
            print("Qdrant client connected at:", path)

    # --------------------------------------------------------------

    def create_collection(self, chunk_embeddings: np.ndarray, verbose: bool = True) -> None:
        """
        Create the collection, dropping any stale one at the same name.

        A path-based Qdrant store persists on disk, so ``create_collection``
        raises "already exists" on the second run of a notebook. Dropping
        first is what makes indexing re-runnable.
        """
        from qdrant_client.models import Distance, VectorParams

        if chunk_embeddings is None:
            raise ValueError("chunk_embeddings cannot be None.")

        if chunk_embeddings.ndim != 2:
            raise ValueError("chunk_embeddings must be a 2D matrix.")

        if chunk_embeddings.dtype != np.float32:
            chunk_embeddings = chunk_embeddings.astype("float32")

        num_chunks, embedding_dim = chunk_embeddings.shape

        if num_chunks == 0:
            raise ValueError(
                "No embeddings found. Create chunk embeddings before creating "
                "the Qdrant collection."
            )

        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
            if verbose:
                print("Dropped existing collection:", self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE,
            ),
        )

        if verbose:
            print("Qdrant collection created.")
            print("Collection name:", self.collection_name)
            print("Embedding dimension:", embedding_dim)
            print("Distance metric: COSINE")

    def upload(
        self,
        chunks_data: List[Dict[str, Any]],
        chunk_embeddings: np.ndarray,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> None:
        """Upload vectors and their metadata payloads in batches."""
        from qdrant_client.models import PointStruct

        if len(chunks_data) == 0:
            raise ValueError("chunks_data is empty.")

        if len(chunk_embeddings) == 0:
            raise ValueError("chunk_embeddings is empty.")

        if len(chunks_data) != len(chunk_embeddings):
            raise ValueError("Number of chunks and embeddings must be the same.")

        points = []

        for idx, (chunk, embedding) in enumerate(zip(chunks_data, chunk_embeddings)):
            payload = {
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "source_type": chunk["source_type"],
                "cell_number": chunk["cell_number"],
                "cell_type": chunk["cell_type"],
                "chunk_text": chunk["chunk_text"],
                "word_count": chunk["word_count"],
                "character_count": chunk["character_count"],
                "preview": chunk["preview"],
                "num_units": chunk["num_units"],
                "chunking_strategy": chunk["chunking_strategy"],
                "max_words": chunk["max_words"],
                "sentence_overlap": chunk["sentence_overlap"],
                "embedding_index": idx,
            }

            points.append(
                PointStruct(id=idx, vector=embedding.tolist(), payload=payload)
            )

            if len(points) >= batch_size:
                self.client.upsert(collection_name=self.collection_name, points=points)
                points = []

        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

        if verbose:
            print("Chunks uploaded to Qdrant.")
            print("Total vectors uploaded:", len(chunk_embeddings))

    # --------------------------------------------------------------

    def count(self) -> int:
        """Number of vectors currently in the collection."""
        return self.client.get_collection(self.collection_name).points_count

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Cosine search over the collection.

        ``top_k`` is capped against the COLLECTION, not against an in-memory
        chunk list: the list says nothing about what was actually uploaded.
        An empty collection raises instead of silently returning [].
        """
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        points_count = self.count()

        if points_count == 0:
            raise RuntimeError(
                f"Qdrant collection '{self.collection_name}' contains 0 vectors. "
                "Upload the chunk embeddings before retrieving."
            )

        top_k = min(top_k, points_count)

        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=np.asarray(query_vector).tolist(),
            limit=top_k,
            with_payload=True,
        ).points

        retrieved_chunks = []

        for rank, result in enumerate(search_results, start=1):
            payload = result.payload or {}

            retrieved_chunks.append({
                "rank": rank,
                "retrieval_score": float(result.score),
                "chunk_id": payload.get("chunk_id", result.id),
                "source": payload.get("source", ""),
                "source_type": payload.get("source_type", ""),
                "cell_number": payload.get("cell_number", None),
                "cell_type": payload.get("cell_type", ""),
                "chunk_text": payload.get("chunk_text", ""),
                "word_count": payload.get("word_count", 0),
                "character_count": payload.get("character_count", 0),
                "preview": payload.get("preview", ""),
                "num_units": payload.get("num_units", 0),
                "chunking_strategy": payload.get("chunking_strategy", ""),
            })

        return retrieved_chunks

    def close(self) -> None:
        """
        Release the storage lock.

        A local Qdrant store is exclusively locked by its client. Re-opening
        it without closing the previous client raises "Storage folder ... is
        already accessed by another instance of Qdrant client".
        """
        try:
            self.client.close()
        except Exception as exc:  # pragma: no cover - defensive
            print("Could not close Qdrant client:", exc)


# ==================================================================
# Stage 6 — Rerank the candidates
# ==================================================================

class ChunkReranker:
    """
    Cross-encoder reranker (default: BAAI/bge-reranker-v2-m3).

    The bi-encoder scores question and chunk separately; the cross-encoder
    reads the PAIR together and can therefore judge relevance far more
    precisely. It is also far more expensive, which is why it only ever sees
    the handful of candidates vector search already shortlisted.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        max_length: int = 512,
        device: Optional[str] = None,
        verbose: bool = True,
    ):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.device = resolve_device(device)

        start_time = time.time()
        self.model = CrossEncoder(model_name, max_length=max_length, device=self.device)
        load_time = time.time() - start_time

        if verbose:
            print("Reranker loaded.")
            print("Model name:", model_name)
            print("Device:", self.device)
            print("Loading time:", round(load_time, 2), "seconds")

    def rerank(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        final_top_k: int = 5,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Score every (question, chunk) pair and keep the best ``final_top_k``.

        Returns
        -------
        reranked_chunks : List[Dict[str, Any]]
            Chunks carrying both scores — ``retrieval_score`` from the vector
            stage and ``reranker_score`` from this one — plus the final rank.
        """
        if question is None or len(question.strip()) == 0:
            raise ValueError("Question cannot be empty.")

        if len(retrieved_chunks) == 0:
            return []

        if final_top_k <= 0:
            raise ValueError("final_top_k must be greater than 0.")

        final_top_k = min(final_top_k, len(retrieved_chunks))

        cleaned_question = TextCleaner.clean(question, "markdown")

        pairs = [
            [cleaned_question, chunk.get("chunk_text", "")]
            for chunk in retrieved_chunks
        ]

        start_time = time.time()
        reranker_scores = self.model.predict(pairs, show_progress_bar=verbose)
        rerank_time = time.time() - start_time

        scored_chunks = []

        for chunk, score in zip(retrieved_chunks, reranker_scores):
            reranked_chunk = chunk.copy()
            reranked_chunk["reranker_score"] = float(score)
            scored_chunks.append(reranked_chunk)

        scored_chunks.sort(key=lambda c: c["reranker_score"], reverse=True)

        reranked_chunks = []

        for rank, chunk in enumerate(scored_chunks[:final_top_k], start=1):
            chunk["reranker_rank"] = rank
            reranked_chunks.append(chunk)

        if verbose:
            print("Reranking completed.")
            print("Candidates reranked:", len(retrieved_chunks))
            print("Final chunks:", len(reranked_chunks))
            print("Reranking time:", round(rerank_time, 2), "seconds")

        return reranked_chunks


# ==================================================================
# Stage 7 — Generate the grounded answer
# ==================================================================

class OllamaGenerator:
    """
    Talk to a locally running Ollama model (default: qwen3:8b).

    Local generation keeps the corpus on the machine it belongs to, which
    matters when the corpus is private study material or internal code.
    """

    def __init__(
        self,
        model_name: str = "qwen3:8b",
        url: str = "http://localhost:11434",
        timeout: int = 300,
    ):
        self.model_name = model_name
        self.url = url.rstrip("/")
        self.timeout = timeout

    def check_connection(self, verbose: bool = True) -> str:
        """Verify Ollama is reachable and return its version."""
        import requests

        response = requests.get(f"{self.url}/api/version", timeout=10)
        response.raise_for_status()
        version = response.json()["version"]

        if verbose:
            print("Ollama connected.")
            print("Model name:", self.model_name)
            print("Ollama URL:", self.url)
            print("Ollama version:", version)

        return version

    def list_models(self) -> List[str]:
        """Names of the models Ollama currently has pulled."""
        import requests

        response = requests.get(f"{self.url}/api/tags", timeout=10)
        response.raise_for_status()

        return [model["name"] for model in response.json()["models"]]

    def generate(self, prompt: str, temperature: float = 0.1) -> Dict[str, Any]:
        """
        Send one prompt and return the answer plus wall-clock time.

        Temperature defaults to 0.1: a grounded answer should reproduce the
        evidence, not improvise around it.
        """
        import requests

        start_time = time.time()

        response = requests.post(
            f"{self.url}/api/chat",
            json={
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "think": False,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        return {
            "answer": response.json()["message"]["content"],
            "generation_time": time.time() - start_time,
        }


# ==================================================================
# The pipeline
# ==================================================================

class DaedalusRAG:
    """
    End-to-end RAG pipeline over a notebook corpus.

    Example
    -------
    >>> rag = DaedalusRAG(corpus_path="corpus/Retrival_Ext_QnA_Complete.ipynb")
    >>> rag.build_index()
    >>> result = rag.answer("What is a text embedding?")
    >>> print(result["answer"])
    """

    def __init__(
        self,
        corpus_path: str,
        qdrant_path: str = "./daedalus_qdrant",
        collection_name: str = "daedalus_chunks",
        max_words: int = 250,
        sentence_overlap: int = 2,
        embedding_model: str = "BAAI/bge-m3",
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        generation_model: str = "qwen3:8b",
        ollama_url: str = "http://localhost:11434",
        device: Optional[str] = None,
        local_files_only: bool = False,
        verbose: bool = True,
    ):
        self.corpus_path = corpus_path
        self.qdrant_path = qdrant_path
        self.collection_name = collection_name
        self.verbose = verbose

        self.chunker = NotebookChunker(
            max_words=max_words,
            sentence_overlap=sentence_overlap,
        )

        self._embedding_model_name = embedding_model
        self._reranker_model_name = reranker_model
        self._device = device
        self._local_files_only = local_files_only

        self.generator = OllamaGenerator(
            model_name=generation_model,
            url=ollama_url,
        )

        # Built lazily so that ingesting or inspecting a corpus never pays
        # the cost of loading two transformer models.
        self.embedder: Optional[ChunkEmbedder] = None
        self.reranker: Optional[ChunkReranker] = None
        self.index: Optional[QdrantIndex] = None

        self.cells_data: List[Dict[str, Any]] = []
        self.cleaned_cells_data: List[Dict[str, Any]] = []
        self.chunks_data: List[Dict[str, Any]] = []

    # --------------------------------------------------------------
    # Lazy component builders
    # --------------------------------------------------------------

    def _get_embedder(self) -> ChunkEmbedder:
        if self.embedder is None:
            self.embedder = ChunkEmbedder(
                model_name=self._embedding_model_name,
                device=self._device,
                local_files_only=self._local_files_only,
                verbose=self.verbose,
            )
        return self.embedder

    def _get_reranker(self) -> ChunkReranker:
        if self.reranker is None:
            self.reranker = ChunkReranker(
                model_name=self._reranker_model_name,
                device=self._device,
                verbose=self.verbose,
            )
        return self.reranker

    def _get_index(self) -> QdrantIndex:
        if self.index is None:
            self.index = QdrantIndex(
                path=self.qdrant_path,
                collection_name=self.collection_name,
                verbose=self.verbose,
            )
        return self.index

    # --------------------------------------------------------------
    # Index time
    # --------------------------------------------------------------

    def ingest(self) -> List[Dict[str, Any]]:
        """Load, clean and chunk the corpus. No models required."""
        loader = NotebookLoader(self.corpus_path)

        self.cells_data = loader.load(verbose=self.verbose)
        self.cleaned_cells_data = TextCleaner.clean_cells(self.cells_data)
        self.chunks_data = self.chunker.chunk(self.cleaned_cells_data)

        if self.verbose:
            print("Chunking completed.")
            print("Total chunks created:", len(self.chunks_data))

        return self.chunks_data

    def build_index(self, batch_size: int = 32) -> None:
        """Ingest the corpus, embed every chunk and load it into Qdrant."""
        if not self.chunks_data:
            self.ingest()

        embedder = self._get_embedder()
        chunk_embeddings = embedder.encode_chunks(
            self.chunks_data,
            batch_size=batch_size,
            verbose=self.verbose,
        )

        index = self._get_index()
        index.create_collection(chunk_embeddings, verbose=self.verbose)
        index.upload(self.chunks_data, chunk_embeddings, verbose=self.verbose)

    # --------------------------------------------------------------
    # Query time
    # --------------------------------------------------------------

    def retrieve(self, question: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Stage 1 — vector search for candidate chunks."""
        if question is None or len(question.strip()) == 0:
            raise ValueError("Question cannot be empty.")

        query_vector = self._get_embedder().encode_query(question)

        return self._get_index().search(query_vector, top_k=top_k)

    def rerank(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        final_top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Stage 2 — cross-encoder reranking of the candidates."""
        return self._get_reranker().rerank(
            question=question,
            retrieved_chunks=retrieved_chunks,
            final_top_k=final_top_k,
            verbose=self.verbose,
        )

    @staticmethod
    def build_evidence_context(chunks: List[Dict[str, Any]]) -> str:
        """
        Format reranked chunks into the evidence block given to the LLM.

        Every block is labelled with its source and cell so the answer can be
        traced back to a specific place in the corpus.
        """
        evidence_parts = []

        for chunk in chunks:
            evidence_parts.append(
                f"""
[Source: {chunk.get('source', 'Unknown')}]
[Cell: {chunk.get('cell_number', 'Unknown')}]
[Cell Type: {chunk.get('cell_type', 'Unknown')}]

{chunk['chunk_text']}
"""
            )

        return "\n".join(evidence_parts)

    @staticmethod
    def build_prompt(question: str, evidence_context: str) -> str:
        """
        Build the grounded-answer prompt.

        Rule 3 exists because an ungrounded model silently reinterprets a
        question in whatever domain its pre-training finds most likely —
        "retrieval failure vs generation failure" becomes a memory-psychology
        question instead of a RAG one.
        """
        return f"""

You are Daedalus, an AI/ML interview preparation assistant.

Your task is to answer the user's question STRICTLY from the
provided study material.

IMPORTANT RULES:

1. The study material is the ONLY source of truth.
2. Do NOT use your general knowledge to answer the question.
3. Do NOT reinterpret the question using another field or domain.
4. Preserve the meaning and terminology used in the study material.
5. If the study material defines a concept, use that definition.
6. Do NOT invent examples, mechanisms, definitions, or explanations
   that are not supported by the study material.
7. If the evidence is insufficient, explicitly say that the
   provided study material does not contain enough information.
8. Give a concise, interview-ready answer.

USER QUESTION:

{question}

PROVIDED STUDY MATERIAL:

{evidence_context}

Now answer the user's question using ONLY the provided study material.

"""

    def answer(
        self,
        question: str,
        retrieve_top_k: int = 10,
        final_top_k: int = 5,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Run the full pipeline for ONE question.

        Retrieval, reranking, evidence building and generation all take the
        same ``question`` argument, so the prompt can never ask question X
        while carrying evidence retrieved for question Y.

        Returns
        -------
        Dict[str, Any]
            question, retrieved_chunks, reranked_chunks, evidence_context,
            prompt, answer, generation_time.
        """
        if question is None or len(question.strip()) == 0:
            raise ValueError("Question cannot be empty.")

        retrieved_chunks = self.retrieve(question, top_k=retrieve_top_k)

        reranked_chunks = self.rerank(
            question,
            retrieved_chunks,
            final_top_k=final_top_k,
        )

        if len(reranked_chunks) == 0:
            raise RuntimeError(
                "No chunks survived retrieval and reranking. "
                "Check that the Qdrant collection is populated."
            )

        evidence_context = self.build_evidence_context(reranked_chunks)
        prompt = self.build_prompt(question, evidence_context)

        generation = self.generator.generate(prompt, temperature=temperature)

        return {
            "question": question,
            "retrieved_chunks": retrieved_chunks,
            "reranked_chunks": reranked_chunks,
            "evidence_context": evidence_context,
            "prompt": prompt,
            "answer": generation["answer"],
            "generation_time": generation["generation_time"],
        }

    def close(self) -> None:
        """Release the Qdrant storage lock."""
        if self.index is not None:
            self.index.close()
            self.index = None
