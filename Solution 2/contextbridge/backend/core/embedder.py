"""Embedding generation with a four-tier backend.

1. Remote OpenAI-compatible ``/embeddings`` endpoint — used when
   ``EMBEDDING_BASE_URL`` is set. No local model download.
2. ``sentence-transformers`` (all-MiniLM-L6-v2) — best local quality, needs torch.
3. ChromaDB's bundled ONNX MiniLM — same model and dimensions, no torch.
4. Deterministic hashing encoder — no downloads, keeps the demo runnable offline.

Every backend emits L2-normalised vectors so cosine similarity and the retrieval
threshold behave identically. Dimensions differ per backend, though — switching
backends against an existing ChromaDB collection is a breaking change (see the
guard in ``vector_store.add_chunks``).
"""

from __future__ import annotations

import hashlib
import math
import time

from backend import config
from backend.utils.logger import logger

_PROBE_ATTEMPTS = 3
_PROBE_BACKOFF = 1.5


class EmbeddingError(RuntimeError):
    """Raised when every embedding backend fails."""


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]


class EmbeddingEngine:
    """Loads one embedding model at startup and reuses it for every request."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self.dimension = config.EMBEDDING_DIMENSION
        self.backend = "hash"
        self._model = None
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if config.EMBEDDING_BASE_URL:
            # Retry the probe: a transient 503 must not silently downgrade us to a
            # different-dimension backend, which would corrupt an existing index.
            last_error: Exception | None = None
            for attempt in range(1, _PROBE_ATTEMPTS + 1):
                try:
                    probe = self._remote_embed(["dimension probe"])
                    self.dimension = len(probe[0])
                    self.backend = "remote"
                    logger.info(
                        f"Embeddings: remote {config.EMBEDDING_REMOTE_MODEL} "
                        f"({self.dimension}d) via {config.EMBEDDING_BASE_URL}"
                    )
                    return
                except Exception as exc:
                    last_error = exc
                    if attempt < _PROBE_ATTEMPTS:
                        logger.warning(
                            f"Remote embeddings probe failed "
                            f"(attempt {attempt}/{_PROBE_ATTEMPTS}): {exc}"
                        )
                        time.sleep(_PROBE_BACKOFF * attempt)

            logger.error(
                f"Remote embeddings unreachable after {_PROBE_ATTEMPTS} attempts "
                f"({last_error}). Falling back to a LOCAL backend with a DIFFERENT "
                "vector dimension. Any collection already built with remote "
                "embeddings will reject writes until you delete data/chroma_db "
                "and re-ingest."
            )

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self.dimension = int(self._model.get_sentence_embedding_dimension())
            self.backend = "sentence-transformers"
            logger.info(f"Embeddings: sentence-transformers/{self.model_name}")
            return
        except Exception as exc:
            logger.debug(f"sentence-transformers unavailable: {exc}")

        try:
            from chromadb.utils import embedding_functions

            self._model = embedding_functions.ONNXMiniLM_L6_V2()
            self.backend = "chroma-onnx-minilm"
            self.dimension = 384
            logger.info("Embeddings: ChromaDB ONNX MiniLM-L6-v2")
            return
        except Exception as exc:
            logger.debug(f"ONNX MiniLM unavailable: {exc}")

        self.backend = "hash"
        self.dimension = config.EMBEDDING_DIMENSION
        logger.warning(
            "Embeddings: deterministic hashing fallback — retrieval quality will be "
            "reduced. Install sentence-transformers for full semantic search."
        )

    # ------------------------------------------------------------------
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed and L2-normalise. Retries once, then falls back to hashing."""
        if not texts:
            return []

        try:
            return self._embed(texts)
        except Exception as exc:
            logger.warning(f"Embedding failed ({exc}); retrying once")
            try:
                return self._embed(texts)
            except Exception as exc2:
                logger.error(f"Embedding failed twice ({exc2}); using hash fallback")
                return [self._hash_embed(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        """Single-query embedding, same normalisation as the corpus."""
        return self.embed_texts([query])[0]

    # ------------------------------------------------------------------
    def _remote_embed(self, texts: list[str]) -> list[list[float]]:
        """POST to an OpenAI-compatible /embeddings endpoint."""
        import httpx

        headers = {"content-type": "application/json"}
        if config.EMBEDDING_API_KEY:
            headers["Authorization"] = f"Bearer {config.EMBEDDING_API_KEY}"

        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{config.EMBEDDING_BASE_URL}/embeddings",
                headers=headers,
                json={"model": config.EMBEDDING_REMOTE_MODEL, "input": texts},
            )
        response.raise_for_status()
        body = response.json()

        rows = body.get("data")
        if not rows:
            raise EmbeddingError(f"embeddings endpoint returned no data: {body}")

        # The spec does not guarantee response order — sort by index when present.
        rows = sorted(rows, key=lambda r: r.get("index", 0))
        vectors = [_l2_normalize([float(v) for v in row["embedding"]]) for row in rows]
        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"expected {len(texts)} embeddings, got {len(vectors)}"
            )
        return vectors

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self.backend == "remote":
            out: list[list[float]] = []
            batch = max(1, config.EMBEDDING_BATCH_SIZE)
            for start in range(0, len(texts), batch):
                out.extend(self._remote_embed(texts[start : start + batch]))
            return out

        if self.backend == "sentence-transformers":
            vectors = self._model.encode(  # type: ignore[union-attr]
                texts,
                batch_size=config.EMBEDDING_BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return [list(map(float, v)) for v in vectors]

        if self.backend == "chroma-onnx-minilm":
            out: list[list[float]] = []
            batch = max(1, config.EMBEDDING_BATCH_SIZE)
            for start in range(0, len(texts), batch):
                window = texts[start : start + batch]
                vectors = self._model(window)  # type: ignore[misc]
                out.extend(_l2_normalize([float(x) for x in v]) for v in vectors)
            return out

        return [self._hash_embed(t) for t in texts]

    def _hash_embed(self, text: str) -> list[float]:
        """Hashed bag-of-words vector — deterministic, order-insensitive, normalised."""
        dim = self.dimension
        vector = [0.0] * dim
        tokens = [t for t in text.lower().split() if t]
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8", "ignore")).digest()
            index = int.from_bytes(digest[:4], "big") % dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return _l2_normalize(vector)

    def info(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "model": (
                config.EMBEDDING_REMOTE_MODEL
                if self.backend == "remote"
                else self.model_name
            ),
            "dimension": self.dimension,
        }


_engine: EmbeddingEngine | None = None


def get_embedder() -> EmbeddingEngine:
    """Process-wide singleton — the model is loaded exactly once."""
    global _engine
    if _engine is None:
        _engine = EmbeddingEngine()
    return _engine
