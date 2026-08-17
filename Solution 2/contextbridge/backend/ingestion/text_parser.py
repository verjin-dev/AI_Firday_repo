"""Plain-text / markdown parser."""

from __future__ import annotations

from pathlib import Path

from backend.core.models import ParsedDocument
from backend.utils.helpers import is_heading
from backend.utils.logger import logger

# Sample docs already carry "--- PAGE n ---" markers; for files without them we
# synthesise a page every N characters so citations still have a page number.
SYNTHETIC_PAGE_CHARS = 3000


class FileParseError(RuntimeError):
    """Raised when a file cannot be read at all."""


class TextParser:
    """Reads .txt/.md and normalises line endings + page markers."""

    SUPPORTED = {".txt", ".md", ".markdown", ".log", ".csv"}

    def parse(self, file_path: str) -> ParsedDocument:
        path = Path(file_path)
        warnings: list[str] = []

        text = ""
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                text = path.read_text(encoding=encoding)
                if encoding != "utf-8":
                    warnings.append(f"Decoded using {encoding} rather than utf-8.")
                break
            except UnicodeDecodeError:
                continue
            except OSError as exc:
                raise FileParseError(f"Cannot read {path.name}: {exc}") from exc

        if not text:
            raise FileParseError(f"{path.name} is empty or unreadable")

        text = text.replace("\r\n", "\n").replace("\r", "\n")

        if "--- PAGE" not in text:
            text = _insert_synthetic_pages(text)
            warnings.append(
                "No page markers found; synthetic page boundaries were inserted "
                f"every ~{SYNTHETIC_PAGE_CHARS} characters for citation purposes."
            )

        page_count = text.count("--- PAGE ") or 1
        headings = [
            title for line in text.splitlines() if (title := is_heading(line))
        ]

        logger.info(
            f"Parsed text {path.name}: {len(text)} chars, ~{page_count} pages, "
            f"{len(headings)} headings"
        )
        return ParsedDocument(
            text=text,
            file_name=path.name,
            page_count=page_count,
            metadata={
                "title": path.stem,
                "source_format": path.suffix.lstrip("."),
                "heading_count": len(headings),
                "section_names": headings[:60],
            },
            warnings=warnings,
        )


def _insert_synthetic_pages(text: str) -> str:
    paragraphs = text.split("\n\n")
    out: list[str] = ["\n\n--- PAGE 1 ---\n"]
    page = 1
    since_break = 0

    for paragraph in paragraphs:
        if since_break >= SYNTHETIC_PAGE_CHARS:
            page += 1
            out.append(f"\n\n--- PAGE {page} ---\n")
            since_break = 0
        out.append(paragraph)
        out.append("\n\n")
        since_break += len(paragraph)

    return "".join(out)
