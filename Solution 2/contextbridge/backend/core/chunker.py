"""Document chunking: flat recursive chunking and section-aware chunking."""

from __future__ import annotations

from typing import Any

from backend import config
from backend.core.models import ChunkResult
from backend.core.token_counter import token_counter
from backend.utils.helpers import is_heading, page_for_offset
from backend.utils.logger import logger

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class EmptyChunkError(ValueError):
    """Raised when a document produces no usable chunks."""


def _splitter(chunk_size_tokens: int, overlap_tokens: int):
    """RecursiveCharacterTextSplitter sized in characters, or a manual fallback."""
    chunk_chars = chunk_size_tokens * config.CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * config.CHARS_PER_TOKEN
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_chars,
            chunk_overlap=overlap_chars,
            separators=_SEPARATORS,
            length_function=len,
            keep_separator=True,
        )
    except ImportError:  # pragma: no cover - langchain is a declared dependency
        logger.warning("langchain-text-splitters missing; using naive splitter")
        return _NaiveSplitter(chunk_chars, overlap_chars)


class _NaiveSplitter:
    """Minimal stand-in that honours the same separator priority."""

    def __init__(self, chunk_chars: int, overlap_chars: int) -> None:
        self.chunk_chars = max(1, chunk_chars)
        self.overlap_chars = max(0, min(overlap_chars, self.chunk_chars - 1))

    def split_text(self, text: str) -> list[str]:
        pieces: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_chars, len(text))
            window = text[start:end]
            if end < len(text):
                for sep in ("\n\n", "\n", ". ", " "):
                    cut = window.rfind(sep)
                    if cut > self.chunk_chars * 0.4:
                        window = window[: cut + len(sep)]
                        end = start + len(window)
                        break
            pieces.append(window)
            if end >= len(text):
                break
            start = max(end - self.overlap_chars, start + 1)
        return [p for p in pieces if p.strip()]


class DocumentChunker:
    """Turns raw document text into token-bounded, metadata-rich chunks."""

    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
        min_chunk_size: int = config.MIN_CHUNK_SIZE,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.splitter = _splitter(chunk_size, chunk_overlap)

    # ------------------------------------------------------------------
    # Flat chunking
    # ------------------------------------------------------------------
    def chunk_text(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[ChunkResult]:
        """Recursively split ``text`` and drop anything under ``min_chunk_size``."""
        metadata = dict(metadata or {})
        if not text or not text.strip():
            raise EmptyChunkError("document text is empty")

        pieces = self.splitter.split_text(text)
        return self._finalize(pieces, text, doc_id, metadata, base_offset=0)

    # ------------------------------------------------------------------
    # Section-aware chunking
    # ------------------------------------------------------------------
    def chunk_by_section(
        self,
        text: str,
        doc_id: str,
        section_markers: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[ChunkResult]:
        """Split on detected headings first, then sub-chunk oversized sections."""
        metadata = dict(metadata or {})
        if not text or not text.strip():
            raise EmptyChunkError("document text is empty")

        sections = self._split_sections(text, section_markers)
        if not sections:
            return self.chunk_text(text, doc_id, metadata)

        chunks: list[ChunkResult] = []
        for title, body, offset in sections:
            if not body.strip():
                continue
            section_meta = {**metadata, "section_name": title}
            if token_counter.count(body) <= self.chunk_size:
                pieces = [body]
            else:
                pieces = self.splitter.split_text(body)
            chunks.extend(
                self._finalize(
                    pieces,
                    body,
                    doc_id,
                    section_meta,
                    base_offset=offset,
                    full_text=text,
                    start_index=len(chunks),
                    renumber=False,
                )
            )

        if not chunks:
            return self.chunk_text(text, doc_id, metadata)

        total = len(chunks)
        for index, chunk in enumerate(chunks):
            chunk.chunk_index = index
            chunk.total_chunks = total
            chunk.chunk_id = f"{doc_id}_chunk_{index:04d}"
            chunk.metadata["chunk_index"] = index

        self._log_stats(doc_id, chunks, mode="section")
        return chunks

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _split_sections(
        self, text: str, section_markers: list[str] | None
    ) -> list[tuple[str, str, int]]:
        """Return ``(title, body, char_offset)`` triples."""
        lines = text.splitlines(keepends=True)
        markers = {m.strip().lower() for m in (section_markers or [])}

        boundaries: list[tuple[int, str]] = []
        offset = 0
        for line in lines:
            stripped = line.strip()
            title = None
            if markers and stripped.lower() in markers:
                title = stripped
            else:
                title = is_heading(line)
            if title:
                boundaries.append((offset, title))
            offset += len(line)

        if not boundaries:
            return []

        sections: list[tuple[str, str, int]] = []
        if boundaries[0][0] > 0:
            sections.append(("Preamble", text[: boundaries[0][0]], 0))

        for i, (start, title) in enumerate(boundaries):
            end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
            sections.append((title, text[start:end], start))
        return sections

    def _finalize(
        self,
        pieces: list[str],
        source_text: str,
        doc_id: str,
        metadata: dict[str, Any],
        base_offset: int,
        full_text: str | None = None,
        start_index: int = 0,
        renumber: bool = True,
    ) -> list[ChunkResult]:
        # In section mode `source_text` is one section and `base_offset` locates it
        # in the whole document, so char offsets are always document-global.
        page_source = full_text if full_text is not None else source_text

        kept: list[ChunkResult] = []
        dropped = 0
        cursor = 0

        for piece in pieces:
            body = piece.strip()
            if not body:
                continue

            tokens = token_counter.count(body)
            if tokens < self.min_chunk_size:
                dropped += 1
                continue

            local_start = source_text.find(piece, cursor)
            if local_start == -1:
                local_start = cursor
            cursor = local_start + max(1, len(piece) // 2)

            char_start = base_offset + local_start
            char_end = char_start + len(piece)
            page = page_for_offset(page_source, char_start)

            index = start_index + len(kept)
            chunk_meta = {
                **metadata,
                "doc_id": doc_id,
                "page": page,
                "chunk_index": index,
                "token_count": tokens,
            }
            kept.append(
                ChunkResult(
                    chunk_id=f"{doc_id}_chunk_{index:04d}",
                    text=body,
                    token_count=tokens,
                    char_start=char_start,
                    char_end=char_end,
                    chunk_index=index,
                    total_chunks=0,  # patched below / by caller
                    metadata=chunk_meta,
                )
            )

        if dropped:
            logger.debug(f"{doc_id}: dropped {dropped} chunk(s) below min size")

        if renumber:
            total = len(kept)
            for chunk in kept:
                chunk.total_chunks = total
            self._log_stats(doc_id, kept, mode="flat", dropped=dropped)

        return kept

    @staticmethod
    def _log_stats(
        doc_id: str, chunks: list[ChunkResult], mode: str, dropped: int = 0
    ) -> None:
        if not chunks:
            logger.warning(f"{doc_id}: chunking produced 0 chunks ({mode})")
            return
        counts = [c.token_count for c in chunks]
        logger.info(
            f"{doc_id}: {len(chunks)} chunks ({mode}) | "
            f"avg {sum(counts) // len(counts)} tok | "
            f"min {min(counts)} | max {max(counts)}"
            + (f" | dropped {dropped}" if dropped else "")
        )
