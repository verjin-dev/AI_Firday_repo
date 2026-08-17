"""Small shared utilities."""

from __future__ import annotations

import hashlib
import re
import sys
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Iterable, Sequence, TypeVar

T = TypeVar("T")

_SLUG_RE = re.compile(r"[^a-z0-9]+")

# A heading is: an ALL CAPS line, a numbered section, "Section N", or a markdown "##".
_HEADING_PATTERNS = [
    re.compile(r"^\s*#{1,6}\s+(?P<title>.+?)\s*$"),
    re.compile(r"^\s*SECTION\s+\d+[.:\-\s]*(?P<title>.*?)\s*$", re.IGNORECASE),
    # "3. Interpretation", "3) Interpretation", and the bare "1.2 Definitions"
    # style common in contracts. Trailing punctuation is optional, but a line
    # ending in a full stop is prose, not a heading.
    re.compile(r"^\s*\d+(?:\.\d+)*[.)]?\s+(?P<title>[A-Z][^\n]{2,80}[^.\s])\s*$"),
]

_PAGE_MARKER_RE = re.compile(r"---\s*PAGE\s+(\d+)\s*---")


def enable_utf8_console() -> None:
    """Make stdout/stderr UTF-8 tolerant.

    Windows consoles default to cp1252, which raises UnicodeEncodeError on
    characters LLMs emit routinely (narrow no-break space, em dashes, emoji).
    Console scripts call this first so a stray character can't kill a demo.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - platform dependent
                pass


def slugify(text: str, max_len: int = 48) -> str:
    """Filesystem/ID-safe lowercase slug."""
    normalised = unicodedata.normalize("NFKD", text)
    ascii_only = normalised.encode("ascii", "ignore").decode("ascii").lower()
    slug = _SLUG_RE.sub("-", ascii_only).strip("-")
    return (slug[:max_len].rstrip("-")) or "doc"


def make_doc_id(file_name: str) -> str:
    """Stable-prefixed, unique document id: ``<slug>-<8 hex>``."""
    return f"{slugify(Path(file_name).stem)}-{uuid.uuid4().hex[:8]}"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()[:16]


def is_heading(line: str) -> str | None:
    """Return the heading title if ``line`` looks like a section heading, else None."""
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return None

    for pattern in _HEADING_PATTERNS:
        match = pattern.match(stripped)
        if match:
            title = (match.group("title") or stripped).strip()
            return title or stripped

    # ALL CAPS line with at least two word characters and no terminal period.
    letters = [c for c in stripped if c.isalpha()]
    if (
        len(letters) >= 3
        and all(c.isupper() for c in letters)
        and not stripped.endswith(".")
    ):
        return stripped

    return None


def page_for_offset(text: str, offset: int) -> int:
    """Page number for a character offset, using ``--- PAGE n ---`` markers."""
    page = 1
    for match in _PAGE_MARKER_RE.finditer(text):
        if match.start() > offset:
            break
        try:
            page = int(match.group(1))
        except ValueError:  # pragma: no cover - regex guarantees digits
            continue
    return page


def truncate_at_sentence(text: str, max_chars: int) -> str:
    """Cut ``text`` to ``max_chars``, backing up to the last sentence boundary."""
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    boundary = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
    if boundary > max_chars * 0.5:
        return window[: boundary + 1]
    space = window.rfind(" ")
    return window[:space] if space > 0 else window


def batched(items: Sequence[T], size: int) -> Iterable[list[T]]:
    """Yield consecutive lists of at most ``size`` items."""
    size = max(1, size)
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def dedupe_preserving_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """ChromaDB only stores scalar metadata — coerce everything else to str."""
    clean: dict[str, str | int | float | bool] = {}
    for key, value in (metadata or {}).items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[str(key)] = value
        elif isinstance(value, (list, tuple, set)):
            clean[str(key)] = ", ".join(str(v) for v in value)
        else:
            clean[str(key)] = str(value)
    return clean


def extract_json(text: str) -> Any | None:
    """Pull the first JSON object/array out of an LLM response. None if unparseable."""
    import json

    if not text:
        return None

    candidate = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", candidate, re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Anchor on the *root* container only. Scanning for any "[...]" would happily
    # return a nested array out of a truncated object, losing the outer keys.
    positions = [(candidate.find(c), c) for c in "{[" if candidate.find(c) != -1]
    if positions:
        _, opener = min(positions)
        closer = "}" if opener == "{" else "]"
        start = candidate.find(opener)
        end = candidate.rfind(closer)
        if end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                pass

    return _repair_truncated_json(candidate)


def _repair_truncated_json(text: str) -> Any | None:
    """Recover what we can from JSON cut off by an output-token limit.

    A truncated extraction is common on long inputs. Discarding it loses every
    entity the model *did* find, so instead we drop the incomplete trailing item
    and close the open brackets.
    """
    import json

    start = min(
        (pos for pos in (text.find("{"), text.find("[")) if pos != -1), default=-1
    )
    if start == -1:
        return None

    body = text[start:]

    # Walk the text tracking nesting, ignoring brackets inside string literals.
    # Record every comma at depth 1 — those separate top-level members and are
    # the cut points that keep the root container intact.
    stack: list[str] = []
    in_string = False
    escaped = False
    cut_points: list[int] = []
    nested_cut_points: list[int] = []

    for index, char in enumerate(body):
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if stack:
                stack.pop()
        elif char == ",":
            if len(stack) == 1:
                cut_points.append(index)
            elif len(stack) == 2:
                nested_cut_points.append(index)

    def _closers(open_stack: list[str]) -> str:
        return "".join(reversed(open_stack))

    # Try the fullest candidate first, then progressively drop trailing members.
    candidates = [body.rstrip().rstrip(",")]
    candidates += [body[:point] for point in reversed(cut_points)]
    # Last resort for a truncated array-of-objects: cut inside the array.
    candidates += [body[:point] for point in reversed(nested_cut_points)]

    for candidate in candidates:
        # Recompute what is still open for this candidate.
        depth: list[str] = []
        inside = False
        esc = False
        for char in candidate:
            if esc:
                esc = False
                continue
            if char == "\\" and inside:
                esc = True
                continue
            if char == '"':
                inside = not inside
                continue
            if inside:
                continue
            if char in "{[":
                depth.append("}" if char == "{" else "]")
            elif char in "}]" and depth:
                depth.pop()

        if inside:
            continue  # candidate ends mid-string; the next cut point is cleaner

        try:
            parsed = json.loads(candidate + _closers(depth))
        except json.JSONDecodeError:
            continue
        if parsed:  # ignore empty shells like {} produced by over-trimming
            return parsed

    return None


class Timer:
    """``with Timer() as t: ...`` then read ``t.seconds``."""

    def __init__(self) -> None:
        self.seconds = 0.0
        self._start = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.seconds = round(time.perf_counter() - self._start, 3)
