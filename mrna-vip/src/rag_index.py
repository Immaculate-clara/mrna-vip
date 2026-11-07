"""Module 3 - retrieval index over the regulatory guideline document.

Pipeline: load text -> token-aware chunking (300-500 tokens, ~50 overlap) ->
dense/sparse embeddings -> persistent vector store -> top-k similarity search.

Two interchangeable backends (chosen automatically, recorded in a manifest so a
reloaded index always uses the one it was built with):

* embedder : sentence-transformers ``all-MiniLM-L6-v2``  (preferred)
             TF-IDF (scikit-learn)                       (offline fallback)
* store    : ChromaDB persistent collection              (preferred)
             NumPy ``.npz`` file                         (offline fallback)

Satisfies FR-3.1 .. FR-3.5.
"""
from __future__ import annotations

import json
import re
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from src import config
from src.text_utils import stem_tokenize

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


# --------------------------------------------------------------------------
# Chunking (FR-3.1)
# --------------------------------------------------------------------------
def count_tokens(text: str) -> int:
    """Approximate token count: words and punctuation marks."""
    return len(_TOKEN_RE.findall(text))


def _split_units(text: str, max_tokens: int) -> list[tuple[str, int, bool]]:
    """Split into sentence-level units: (text, n_tokens, starts_new_paragraph).

    A short heading line (no terminal period) is glued to the paragraph that follows
    it, so a chunk boundary can never strand a heading at the end of a chunk.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    merged: list[str] = []
    carry = ""
    for p in paragraphs:
        is_heading = len(p) < 90 and not p.endswith((".", "!", "?"))
        if is_heading:
            carry = f"{carry}\n{p}".strip()
        else:
            merged.append(f"{carry}\n{p}".strip() if carry else p)
            carry = ""
    if carry:
        merged.append(carry)

    units: list[tuple[str, int, bool]] = []
    for paragraph in merged:
        first = True
        for sentence in _SENTENCE_RE.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            # Pathological oversize sentence: hard-split on words.
            while count_tokens(sentence) > max_tokens:
                words = sentence.split()
                part = " ".join(words[: max(1, len(words) // 2)])
                units.append((part, count_tokens(part), first))
                first, sentence = False, " ".join(words[len(part.split()):])
            units.append((sentence, count_tokens(sentence), first))
            first = False
    return units


def chunk_text(text: str, min_tokens: int = config.CHUNK_MIN_TOKENS,
               max_tokens: int = config.CHUNK_TARGET_MAX_TOKENS,
               overlap_tokens: int = config.CHUNK_OVERLAP_TOKENS) -> list[str]:
    """Greedy sentence packing into chunks of min..max tokens with ~overlap tokens.

    Unlike a plain character splitter this *guarantees* the size window (except
    when the whole document is shorter than ``min_tokens``) and never cuts a
    sentence in half. Overlap is whole trailing sentences totalling >= overlap_tokens.
    """
    units = _split_units(text, max_tokens)
    n = len(units)
    if n == 0:
        return []
    tok = [u[1] for u in units]

    ranges: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end, total = start, 0
        while end < n and total + tok[end] <= max_tokens:
            total += tok[end]
            end += 1
        end = max(end, start + 1)
        ranges.append((start, end))
        if end >= n:
            break
        next_start, overlap = end, 0
        while next_start - 1 > start and overlap < overlap_tokens:
            next_start -= 1
            overlap += tok[next_start]
        start = next_start

    # A short tail chunk is grown backwards (more overlap) until it reaches min_tokens.
    s, e = ranges[-1]
    total = sum(tok[s:e])
    while len(ranges) > 1 and total < min_tokens and s > 0 and total + tok[s - 1] <= max_tokens:
        s -= 1
        total += tok[s]
    ranges[-1] = (s, e)

    def render(a: int, b: int) -> str:
        parts: list[str] = []
        for i in range(a, b):
            if parts:
                parts.append("\n" if units[i][2] else " ")
            parts.append(units[i][0])
        return "".join(parts)

    return [render(a, b) for a, b in ranges]


# --------------------------------------------------------------------------
# Embedders
# --------------------------------------------------------------------------
def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1.0, norms)


class TfidfEmbedder:
    """Offline sparse-vector embedder (cosine similarity on L2-normalised TF-IDF)."""

    name = "tfidf"

    def __init__(self, vectorizer=None):
        self.vectorizer = vectorizer

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(tokenizer=stem_tokenize, token_pattern=None,
                                          ngram_range=(1, 2), sublinear_tf=True,
                                          stop_words=None)
        self.vectorizer.fit(texts)
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalize(self.vectorizer.transform(texts).toarray().astype(np.float32))

    def save(self, directory: Path) -> None:
        joblib.dump(self.vectorizer, directory / "tfidf_vectorizer.joblib")

    @classmethod
    def load(cls, directory: Path) -> "TfidfEmbedder":
        return cls(joblib.load(directory / "tfidf_vectorizer.joblib"))


class SentenceTransformerEmbedder:
    """Dense embeddings from a sentence-transformers model (needs the package + model download)."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL_NAME):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.name = f"sentence-transformers:{model_name}"
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)

    def save(self, directory: Path) -> None:  # model is re-loaded by name
        pass

    @classmethod
    def load(cls, directory: Path, model_name: str) -> "SentenceTransformerEmbedder":
        return cls(model_name)


# --------------------------------------------------------------------------
# Vector stores (FR-3.3: persisted, survive a process restart)
# --------------------------------------------------------------------------
class NumpyVectorStore:
    name = "numpy"

    def __init__(self, directory: Path):
        self.path = Path(directory) / "vectors.npz"
        self.texts: list[str] = []
        self.vectors = np.zeros((0, 0), dtype=np.float32)

    def reset_and_add(self, texts: list[str], vectors: np.ndarray) -> None:
        self.texts, self.vectors = list(texts), vectors.astype(np.float32)
        np.savez(self.path, vectors=self.vectors, texts=np.array(self.texts, dtype=object))

    def load(self) -> None:
        data = np.load(self.path, allow_pickle=True)
        self.vectors, self.texts = data["vectors"], [str(t) for t in data["texts"]]

    def count(self) -> int:
        return len(self.texts)

    def search(self, vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        sims = self.vectors @ vector  # vectors are unit length -> cosine similarity
        order = np.argsort(-sims, kind="stable")[:k]
        return [(int(i), float(sims[i])) for i in order]


class ChromaVectorStore:
    name = "chroma"

    def __init__(self, directory: Path):
        import chromadb

        self.client = chromadb.PersistentClient(path=str(Path(directory) / "chroma"))
        self.collection = None

    def reset_and_add(self, texts: list[str], vectors: np.ndarray) -> None:
        try:
            self.client.delete_collection(config.COLLECTION_NAME)
        except Exception:  # collection did not exist yet
            pass
        self.collection = self.client.create_collection(
            config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
        self.collection.add(ids=[f"chunk-{i}" for i in range(len(texts))],
                            documents=list(texts), embeddings=vectors.tolist())

    def load(self) -> None:
        self.collection = self.client.get_collection(config.COLLECTION_NAME)

    def count(self) -> int:
        return self.collection.count()

    def search(self, vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        res = self.collection.query(query_embeddings=[vector.tolist()],
                                    n_results=min(k, self.count()))
        return [(int(cid.split("-")[1]), 1.0 - float(dist))
                for cid, dist in zip(res["ids"][0], res["distances"][0])]


# --------------------------------------------------------------------------
# The index
# --------------------------------------------------------------------------
@dataclass
class RetrievedChunk:
    chunk_id: int
    text: str
    score: float


class GuidelineIndex:
    def __init__(self, embedder, store, chunks: list[str], persist_dir: Path,
                 log_path: Path | None = config.RETRIEVAL_LOG_PATH):
        self.embedder, self.store, self.chunks = embedder, store, chunks
        self.persist_dir, self.log_path = Path(persist_dir), log_path

    @property
    def backend(self) -> str:
        return f"{self.embedder.name}+{self.store.name}"

    # ---- build / load -------------------------------------------------
    @classmethod
    def build(cls, doc_path: Path = config.GUIDELINES_TXT,
              persist_dir: Path = config.VECTOR_STORE_DIR, backend: str = "auto",
              log_path: Path | None = config.RETRIEVAL_LOG_PATH) -> "GuidelineIndex":
        """backend: 'auto' (best available), 'full' (require ST + Chroma) or 'offline'."""
        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        chunks = chunk_text(Path(doc_path).read_text(encoding="utf-8"))
        embedder = cls._make_embedder(backend, chunks)
        store = cls._make_store(backend, persist_dir)

        vectors = embedder.embed(chunks)
        assert len(vectors) == len(chunks)  # FR-3.2
        store.reset_and_add(chunks, vectors)
        embedder.save(persist_dir)
        (persist_dir / "manifest.json").write_text(json.dumps({
            "embedder": embedder.name, "store": store.name, "n_chunks": len(chunks),
            "chunk_tokens": [count_tokens(c) for c in chunks],
            "model_name": config.EMBEDDING_MODEL_NAME,
        }, indent=2), encoding="utf-8")
        return cls(embedder, store, chunks, persist_dir, log_path)

    @staticmethod
    def _make_embedder(backend: str, chunks: list[str]):
        if backend != "offline":
            try:
                return SentenceTransformerEmbedder()
            except Exception as exc:  # missing package, or model cannot be downloaded
                if backend == "full":
                    raise RuntimeError(f"sentence-transformers backend unavailable: {exc}") from exc
                warnings.warn(f"Falling back to TF-IDF embeddings ({type(exc).__name__}).")
        return TfidfEmbedder().fit(chunks)

    @staticmethod
    def _make_store(backend: str, persist_dir: Path):
        if backend != "offline":
            try:
                return ChromaVectorStore(persist_dir)
            except Exception as exc:
                if backend == "full":
                    raise RuntimeError(f"ChromaDB backend unavailable: {exc}") from exc
                warnings.warn(f"Falling back to NumPy vector store ({type(exc).__name__}).")
        return NumpyVectorStore(persist_dir)

    @classmethod
    def load(cls, persist_dir: Path = config.VECTOR_STORE_DIR,
             log_path: Path | None = config.RETRIEVAL_LOG_PATH) -> "GuidelineIndex":
        """Re-open a persisted index without re-embedding (FR-3.3)."""
        persist_dir = Path(persist_dir)
        manifest = json.loads((persist_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest["embedder"] == "tfidf":
            embedder = TfidfEmbedder.load(persist_dir)
        else:
            embedder = SentenceTransformerEmbedder.load(persist_dir, manifest["model_name"])
        store = ChromaVectorStore(persist_dir) if manifest["store"] == "chroma" \
            else NumpyVectorStore(persist_dir)
        store.load()
        chunks = _chunks_from_store(store, manifest["n_chunks"])
        return cls(embedder, store, chunks, persist_dir, log_path)

    # ---- query (FR-3.4, FR-3.5) -----------------------------------------
    def query(self, question: str, k: int = config.DEFAULT_TOP_K) -> tuple[list[RetrievedChunk], float]:
        if not question.strip():
            raise ValueError("question must not be empty")
        start = time.perf_counter()
        vector = self.embedder.embed([question])[0]
        hits = self.store.search(vector, k)
        latency = max(time.perf_counter() - start, 1e-9)
        results = [RetrievedChunk(i, self.chunks[i], s) for i, s in hits]
        self._log(question, k, latency, results)
        return results, latency

    def _log(self, question: str, k: int, latency: float, results: list[RetrievedChunk]) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "question": question,
                 "k": k, "latency_s": latency, "backend": self.backend,
                 "chunk_ids": [r.chunk_id for r in results],
                 "scores": [round(r.score, 4) for r in results]}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def _chunks_from_store(store, n_chunks: int) -> list[str]:
    if isinstance(store, NumpyVectorStore):
        return store.texts
    got = store.collection.get(ids=[f"chunk-{i}" for i in range(n_chunks)])
    by_id = dict(zip(got["ids"], got["documents"]))
    return [by_id[f"chunk-{i}"] for i in range(n_chunks)]


def query_guidelines(index: GuidelineIndex, question: str,
                     k: int = config.DEFAULT_TOP_K) -> tuple[list[str], float]:
    """Template-compatible wrapper: returns (top-k chunk strings, latency in seconds)."""
    results, latency = index.query(question, k)
    return [r.text for r in results], latency


# --------------------------------------------------------------------------
# Demo: build, sanity-check the chunks, run test questions, write a markdown table
# --------------------------------------------------------------------------
TEST_QUESTIONS = [
    "What are the storage requirements below -20C?",
    "What dose range applies to patients aged 65 and older?",
    "Within how many hours must a moderate adverse event be documented?",
    "Who must never receive the vaccine? What are the absolute contraindications?",
    "How long should a patient be monitored after administration?",
    "At what dose does a vaccine require secondary clinician review?",
    # Deliberate out-of-scope probe: the guideline is silent on pregnancy.
    "Is the vaccine safe to give during pregnancy?",
]


def main() -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if config.RETRIEVAL_LOG_PATH.exists():
        config.RETRIEVAL_LOG_PATH.unlink()
    index = GuidelineIndex.build()
    print(f"Backend: {index.backend}; {len(index.chunks)} chunks; "
          f"token counts: {[count_tokens(c) for c in index.chunks]}")

    reloaded = GuidelineIndex.load()  # proves persistence works without re-embedding
    rows = ["| # | Question | Top chunk | Score | Latency (ms) |",
            "|---|----------|-----------|-------|--------------|"]
    for i, q in enumerate(TEST_QUESTIONS, 1):
        results, latency = reloaded.query(q, k=config.DEFAULT_TOP_K)
        top = f"#{results[0].chunk_id}: {results[0].text.splitlines()[0][:50]}..."
        rows.append(f"| {i} | {q} | {top} | {results[0].score:.3f} | {latency * 1000:.2f} |")
    table = "\n".join(rows)
    (config.OUTPUT_DIR / "retrieval_report.md").write_text(
        f"# Retrieval test queries\n\nBackend: `{index.backend}`\n\n{table}\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
