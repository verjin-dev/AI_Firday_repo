"""PDF parser built on PyMuPDF (fitz).

Preserves page boundaries as ``--- PAGE n ---`` so every retrieved chunk can cite
a real page number, and extracts tables separately as markdown.
"""

from __future__ import annotations

from pathlib import Path

from backend.core.models import ParsedDocument, TableResult
from backend.ingestion.text_parser import FileParseError
from backend.utils.helpers import is_heading
from backend.utils.logger import logger


def _pymupdf():
    """Import PyMuPDF under its current name, falling back to the legacy alias."""
    try:
        import pymupdf

        return pymupdf
    except ImportError:
        try:
            import fitz  # legacy alias, deprecated upstream

            return fitz
        except ImportError as exc:  # pragma: no cover - declared dependency
            raise FileParseError(f"PyMuPDF not installed: {exc}") from exc


class PDFParser:
    SUPPORTED = {".pdf"}

    def parse(self, file_path: str) -> ParsedDocument:
        path = Path(file_path)
        fitz = _pymupdf()

        try:
            document = fitz.open(str(path))
        except Exception as exc:
            raise FileParseError(f"Cannot open {path.name}: {exc}") from exc

        warnings: list[str] = []
        parts: list[str] = []
        headings: list[str] = []
        empty_pages = 0

        try:
            for page_number in range(document.page_count):
                parts.append(f"\n\n--- PAGE {page_number + 1} ---\n\n")
                try:
                    page = document.load_page(page_number)
                    text = page.get_text("text") or ""
                except Exception as exc:
                    warnings.append(f"Page {page_number + 1} failed to parse: {exc}")
                    continue

                if not text.strip():
                    empty_pages += 1
                    continue

                parts.append(text)
                headings.extend(
                    title
                    for line in text.splitlines()
                    if (title := is_heading(line))
                )

            tables = self._extract_tables(document, warnings)
            for table in tables:
                parts.append(f"\n\n[TABLE — page {table.page}]\n{table.markdown}\n")

            metadata = document.metadata or {}
            page_count = document.page_count
        finally:
            document.close()

        text = "".join(parts)
        if not text.strip():
            raise FileParseError(
                f"{path.name} yielded no extractable text — it may be a scanned "
                "document requiring OCR."
            )

        if empty_pages:
            warnings.append(
                f"{empty_pages} of {page_count} pages had no extractable text "
                "(likely images or scans)."
            )

        logger.info(
            f"Parsed pdf {path.name}: {page_count} pages, {len(text)} chars, "
            f"{len(tables)} tables"
        )
        return ParsedDocument(
            text=text,
            file_name=path.name,
            page_count=page_count,
            metadata={
                "title": metadata.get("title") or path.stem,
                "author": metadata.get("author", ""),
                "creation_date": metadata.get("creationDate", ""),
                "source_format": "pdf",
                "page_count": page_count,
                "heading_count": len(headings),
                "section_names": headings[:60],
                "table_count": len(tables),
            },
            tables=tables,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    def extract_tables(self, file_path: str) -> list[TableResult]:
        """Standalone table extraction (also called internally by ``parse``)."""
        document = _pymupdf().open(str(file_path))
        try:
            return self._extract_tables(document, [])
        finally:
            document.close()

    @staticmethod
    def _extract_tables(document, warnings: list[str]) -> list[TableResult]:
        results: list[TableResult] = []
        for page_number in range(document.page_count):
            try:
                page = document.load_page(page_number)
                finder = page.find_tables()
            except Exception:
                # find_tables() needs a recent PyMuPDF; absence is not an error.
                continue

            for table in getattr(finder, "tables", []) or []:
                try:
                    rows = table.extract()
                except Exception as exc:
                    warnings.append(
                        f"Table on page {page_number + 1} failed to extract: {exc}"
                    )
                    continue
                markdown = _rows_to_markdown(rows)
                if markdown:
                    results.append(
                        TableResult(
                            page=page_number + 1,
                            markdown=markdown,
                            rows=len(rows),
                            cols=len(rows[0]) if rows else 0,
                        )
                    )
        return results


def _rows_to_markdown(rows: list[list]) -> str:
    cleaned = [
        [str(cell or "").strip().replace("\n", " ") for cell in row] for row in rows
    ]
    cleaned = [row for row in cleaned if any(cell for cell in row)]
    if not cleaned:
        return ""

    header = cleaned[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in cleaned[1:])
    return "\n".join(lines)
