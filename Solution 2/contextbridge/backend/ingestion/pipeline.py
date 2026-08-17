"""Full ingestion orchestration: parse -> chunk -> embed -> store -> summarize -> extract."""

from __future__ import annotations

from pathlib import Path

from backend import config
from backend.core.chunker import DocumentChunker, EmptyChunkError
from backend.core.embedder import get_embedder
from backend.core.models import IngestionResult, ParsedDocument, SummaryResult
from backend.core.summarizer import HierarchicalSummarizer
from backend.core.token_counter import token_counter
from backend.core.vector_store import VectorStoreError, get_vector_store
from backend.domain.entity_extractor import EntityExtractor
from backend.ingestion.docx_parser import DOCXParser
from backend.ingestion.pdf_parser import PDFParser
from backend.ingestion.text_parser import FileParseError, TextParser
from backend.utils.helpers import Timer, make_doc_id
from backend.utils.logger import logger


class IngestionPipeline:
    """Turns a file on disk into indexed, summarized, entity-tagged chunks."""

    def __init__(self) -> None:
        self.chunker = DocumentChunker()
        self.embedder = get_embedder()
        self.vector_store = get_vector_store()
        self.summarizer = HierarchicalSummarizer()
        self.entity_extractor = EntityExtractor()
        self._parsers = {
            **{ext: PDFParser() for ext in PDFParser.SUPPORTED},
            **{ext: DOCXParser() for ext in DOCXParser.SUPPORTED},
            **{ext: TextParser() for ext in TextParser.SUPPORTED},
        }

    # ------------------------------------------------------------------
    def supported_extensions(self) -> list[str]:
        return sorted(self._parsers)

    async def ingest(
        self,
        file_path: str,
        doc_type: str = "general",
        run_summarization: bool = True,
        doc_id: str | None = None,
        original_name: str | None = None,
    ) -> IngestionResult:
        """``original_name`` preserves the user's filename when the upload was
        saved under a generated, path-safe name — it is what the UI displays."""
        path = Path(file_path)
        doc_id = doc_id or make_doc_id(path.name)
        warnings: list[str] = []

        with Timer() as timer:
            # 1. parse -------------------------------------------------
            try:
                parsed = self._parse(path, doc_type)
            except FileParseError as exc:
                logger.error(f"Parse failed for {path.name}: {exc}")
                return IngestionResult(
                    doc_id=doc_id,
                    file_name=path.name,
                    doc_type=doc_type,
                    total_pages=0,
                    total_chars=0,
                    total_tokens=0,
                    total_chunks=0,
                    chunks_stored=0,
                    ingestion_time_seconds=timer.seconds,
                    status="failed",
                    warnings=[str(exc)],
                )
            warnings.extend(parsed.warnings)
            if original_name:
                parsed.file_name = original_name

            total_tokens = token_counter.count(parsed.text)

            # 2. chunk -------------------------------------------------
            base_metadata = {
                "doc_id": doc_id,
                "file_name": parsed.file_name,
                "doc_type": doc_type,
                "source_format": parsed.metadata.get("source_format", ""),
                "title": parsed.metadata.get("title", parsed.file_name),
            }
            try:
                chunks = self.chunker.chunk_by_section(
                    parsed.text, doc_id, metadata=base_metadata
                )
            except EmptyChunkError as exc:
                return IngestionResult(
                    doc_id=doc_id,
                    file_name=parsed.file_name,
                    doc_type=doc_type,
                    total_pages=parsed.page_count,
                    total_chars=parsed.total_chars,
                    total_tokens=total_tokens,
                    total_chunks=0,
                    chunks_stored=0,
                    ingestion_time_seconds=timer.seconds,
                    status="failed",
                    warnings=warnings + [f"Chunking produced nothing: {exc}"],
                )

            if not chunks:
                warnings.append(
                    "All chunks fell below the minimum size threshold; "
                    "falling back to flat chunking."
                )
                chunks = self.chunker.chunk_text(parsed.text, doc_id, base_metadata)

            # 3-5. embed + store ---------------------------------------
            chunks_stored = 0
            try:
                embeddings = self.embedder.embed_texts([c.text for c in chunks])
                chunks_stored = self.vector_store.add_chunks(chunks, embeddings)
            except VectorStoreError as exc:
                warnings.append(f"Vector storage failed: {exc}")
                logger.error(f"Vector storage failed for {doc_id}: {exc}")
            except Exception as exc:
                warnings.append(f"Embedding/storage failed: {exc}")
                logger.error(f"Embedding failed for {doc_id}: {exc}")

            # Keyword search reads its corpus from the store — refresh it.
            try:
                from backend.core.retriever import get_retriever

                get_retriever().invalidate_cache(doc_id)
            except Exception:  # pragma: no cover - cache refresh is best-effort
                pass

            # 6. summarize ---------------------------------------------
            summary: SummaryResult | None = None
            if run_summarization:
                try:
                    summary = await self.summarizer.summarize_document(
                        chunks, doc_type=doc_type
                    )
                    warnings.extend(summary.warnings)
                except Exception as exc:
                    warnings.append(f"Summarization failed: {exc}")
                    logger.error(f"Summarization failed for {doc_id}: {exc}")

            # 7. extract entities --------------------------------------
            entities: dict[str, list[str]] | None = None
            try:
                source = (
                    summary.master_summary
                    if summary and summary.master_summary
                    else parsed.text
                )
                entities = await self.entity_extractor.extract(source)
            except Exception as exc:
                warnings.append(f"Entity extraction failed: {exc}")
                logger.error(f"Entity extraction failed for {doc_id}: {exc}")

        status = "success"
        if chunks_stored == 0:
            status = "failed"
        elif warnings or chunks_stored < len(chunks):
            status = "partial"

        result = IngestionResult(
            doc_id=doc_id,
            file_name=parsed.file_name,
            doc_type=doc_type,
            total_pages=parsed.page_count,
            total_chars=parsed.total_chars,
            total_tokens=total_tokens,
            total_chunks=len(chunks),
            chunks_stored=chunks_stored,
            summary=summary,
            entities=entities,
            ingestion_time_seconds=timer.seconds,
            status=status,
            warnings=warnings,
        )
        logger.info(
            f"Ingested {parsed.file_name} as {doc_id}: {len(chunks)} chunks, "
            f"{total_tokens} tokens, {timer.seconds}s, status={status}"
        )
        return result

    # ------------------------------------------------------------------
    def _parse(self, path: Path, doc_type: str) -> ParsedDocument:
        if not path.exists():
            raise FileParseError(f"File not found: {path}")

        parser = self._parsers.get(path.suffix.lower())
        if parser is None:
            raise FileParseError(
                f"Unsupported file type '{path.suffix}'. "
                f"Supported: {', '.join(self.supported_extensions())}"
            )

        parsed = parser.parse(str(path))
        parsed.doc_type = doc_type
        return parsed

    # ------------------------------------------------------------------
    def context_overflow_factor(self, total_tokens: int) -> float:
        """How many times over a baseline context window this document runs."""
        return round(total_tokens / max(1, config.BASELINE_CONTEXT_TOKENS), 1)


_pipeline: IngestionPipeline | None = None


def get_pipeline() -> IngestionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline()
    return _pipeline
